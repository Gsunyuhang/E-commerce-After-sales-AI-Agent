"""
LangGraph 节点实现
包含：政策答疑、订单查询、退换货处理、情绪安抚、超范围处理、结果校验
"""

import re
import time
from typing import Optional

from langchain_core.messages import AIMessage, HumanMessage

from agent.intent import (
    INTENT_COMPLAINT,
    INTENT_ORDER_QUERY,
    INTENT_OUT_OF_SCOPE,
    INTENT_POLICY_QA,
    INTENT_RETURN_PROCESS,
)
from agent.memory import get_memory
from agent.state import AgentState
from config.prompts import (
    EMOTION_COMFORT_PROMPT,
    OUT_OF_SCOPE_PROMPT,
    POLICY_QA_PROMPT,
    RETURN_PROCESS_PROMPT,
    SYSTEM_PROMPT,
    format_prompt,
)
from config.settings import get_settings
from knowledge.retriever import get_retriever
from llm.qwen_client import get_qwen_client
from tools.logistics_query import LogisticsQueryTool
from tools.order_query import OrderQueryTool
from tools.refund_calc import RefundCalcTool
from tools.ticket_create import TicketCreateTool
from utils.exception import handle_agent_error
from utils.logger import logger


# ============================================
# 工具实例（延迟初始化）
# ============================================
_order_tool: Optional[OrderQueryTool] = None
_logistics_tool: Optional[LogisticsQueryTool] = None
_ticket_tool: Optional[TicketCreateTool] = None
_refund_tool: Optional[RefundCalcTool] = None


def _get_order_tool() -> OrderQueryTool:
    global _order_tool
    if _order_tool is None:
        _order_tool = OrderQueryTool()
    return _order_tool


def _get_logistics_tool() -> LogisticsQueryTool:
    global _logistics_tool
    if _logistics_tool is None:
        _logistics_tool = LogisticsQueryTool()
    return _logistics_tool


def _get_ticket_tool() -> TicketCreateTool:
    global _ticket_tool
    if _ticket_tool is None:
        _ticket_tool = TicketCreateTool()
    return _ticket_tool


def _get_refund_tool() -> RefundCalcTool:
    global _refund_tool
    if _refund_tool is None:
        _refund_tool = RefundCalcTool()
    return _refund_tool


def _add_trace(state: AgentState, step: str) -> list[str]:
    """添加工作流轨迹"""
    trace = state.get("workflow_trace", [])
    trace.append(step)
    return trace


# ============================================
# 政策答疑节点（优化点2：反幻觉机制）
# ============================================
def policy_qa_node(state: AgentState) -> AgentState:
    """
    政策答疑节点
    流程：知识库检索 → 置信度判断 → 答案生成（或转人工）
    """
    user_input = state.get("user_input", "")
    trace = _add_trace(state, "policy_qa_node: start")

    try:
        retriever = get_retriever()
        retrieval_result = retriever.retrieve(user_input, use_rewrite=True)

        # 记录检索结果到状态
        retrieved_docs = retrieval_result.to_dict()

        # 反幻觉机制：检索结果不可靠时，直接转人工
        if not retrieval_result.is_reliable or not retrieval_result.documents:
            trace.append("policy_qa_node: retrieval not reliable, handoff to human")
            response = (
                "抱歉，我暂时无法找到与您问题相关的售后政策信息。"
                "已为您转接人工客服进行核实，请稍候。"
            )
            return {
                **state,
                "response": response,
                "need_human": True,
                "retrieved_docs": [retrieved_docs],
                "is_retrieval_reliable": False,
                "retrieval_max_score": retrieval_result.max_score,
                "confidence": retrieval_result.max_score,
                "workflow_trace": trace,
            }

        # 检索可靠，生成答案
        context = retrieval_result.to_context()
        prompt = format_prompt(
            POLICY_QA_PROMPT,
            retrieved_context=context,
            user_question=user_input,
        )

        # 流式模式：存储 prompt，不调用 LLM
        if state.get("_stream_mode"):
            trace.append(f"policy_qa_node: stream mode, prompt stored, score={retrieval_result.max_score:.2f}")
            return {
                **state,
                "response": "",
                "_stream_prompt": prompt,
                "_stream_system_prompt": SYSTEM_PROMPT,
                "need_human": False,
                "retrieved_docs": [retrieved_docs],
                "is_retrieval_reliable": True,
                "retrieval_max_score": retrieval_result.max_score,
                "confidence": retrieval_result.max_score,
                "workflow_trace": trace,
            }

        qwen_client = get_qwen_client()
        response = qwen_client.chat(
            messages=[HumanMessage(content=prompt)],
            system_prompt=SYSTEM_PROMPT,
        )

        trace.append(f"policy_qa_node: answer generated, score={retrieval_result.max_score:.2f}")

        return {
            **state,
            "response": response,
            "need_human": False,
            "retrieved_docs": [retrieved_docs],
            "is_retrieval_reliable": True,
            "retrieval_max_score": retrieval_result.max_score,
            "confidence": retrieval_result.max_score,
            "workflow_trace": trace,
        }

    except Exception as e:
        logger.error(f"政策答疑节点异常: {e}")
        trace.append(f"policy_qa_node: error={e}")
        return {
            **state,
            "response": handle_agent_error(e),
            "need_human": True,
            "error": str(e),
            "workflow_trace": trace,
        }


# ============================================
# 订单/物流查询节点（优化点3：工具前置条件校验）
# ============================================
def order_query_node(state: AgentState) -> AgentState:
    """
    订单/物流查询节点
    优化点3：未检测到有效订单号时先追问，不直接调用工具
    """
    user_input = state.get("user_input", "")
    slots = state.get("slots", {})
    context_summary = state.get("context_summary", "")
    messages = state.get("messages", [])
    trace = _add_trace(state, "order_query_node: start")

    try:
        # 优化点3：前置条件检查 - 从 slots 和用户输入中提取订单号
        order_id = slots.get("order_id")

        # 如果 slots 中没有，尝试从用户输入中正则提取
        if not order_id:
            order_match = re.search(r"DD\d+", user_input)
            if order_match:
                order_id = order_match.group(0)
                trace.append(f"order_query_node: extracted order_id={order_id} from input")

        # 如果上下文摘要中有订单号，也尝试使用
        if not order_id and context_summary:
            order_match = re.search(r"DD\d+", context_summary)
            if order_match:
                order_id = order_match.group(0)
                trace.append(f"order_query_node: found order_id={order_id} in context summary")

        # 没有订单号，追问用户
        if not order_id:
            trace.append("order_query_node: no order_id found, asking user")
            response = (
                "您好，请问您要查询的订单号是多少呢？\n"
                "订单号格式为DD开头的数字串（如DD20250701001），"
                "您可以在'我的订单'页面找到。"
            )
            return {
                **state,
                "response": response,
                "need_human": False,
                "workflow_trace": trace,
            }

        # 有订单号，判断用户是想查订单还是查物流
        logistics_keywords = ["物流", "快递", "到哪", "什么时候到", "配送", "签收", "运输"]
        is_logistics_query = any(kw in user_input for kw in logistics_keywords)

        tool_calls = []
        tool_results = []

        if is_logistics_query:
            # 查询物流
            trace.append(f"order_query_node: querying logistics for {order_id}")
            logistics_tool = _get_logistics_tool()
            result = logistics_tool.execute(order_id=order_id)
            tool_calls.append({
                "tool_name": "logistics_query",
                "tool_input": {"order_id": order_id},
                "success": result["success"],
            })
            tool_results.append(result)

            if result["success"] and result["data"].get("has_logistics"):
                data = result["data"]
                tracking_info = data.get("tracking_info", [])
                tracking_text = "\n".join(
                    f"  {t['time']} | {t['location']} | {t['desc']}"
                    for t in tracking_info
                )
                response = (
                    f"您的订单物流信息如下：\n\n"
                    f"订单号：{order_id}\n"
                    f"物流公司：{data.get('logistics_company', '')}\n"
                    f"物流单号：{data.get('tracking_number', '')}\n"
                    f"当前状态：{data.get('status_text', '')}\n"
                    f"预计到达：{data.get('estimated_arrival', '未知')}\n\n"
                    f"物流轨迹：\n{tracking_text}"
                )
            elif result["success"] and not result["data"].get("has_logistics"):
                response = result["data"].get("message", "该订单尚未发货，暂无物流信息。")
            else:
                response = result.get("fallback", "物流查询服务暂时不可用，请稍后重试。")
        else:
            # 查询订单
            trace.append(f"order_query_node: querying order {order_id}")
            order_tool = _get_order_tool()
            result = order_tool.execute(order_id=order_id)
            tool_calls.append({
                "tool_name": "order_query",
                "tool_input": {"order_id": order_id},
                "success": result["success"],
            })
            tool_results.append(result)

            if result["success"] and result["data"].get("found"):
                data = result["data"]
                order = data["order"]
                insurance_text = "已购买" if data.get("has_insurance") else "未购买"
                response = (
                    f"您的订单信息如下：\n\n"
                    f"订单号：{order['order_id']}\n"
                    f"商品名称：{order['product_name']}\n"
                    f"商品价格：¥{order['product_price']:.2f}\n"
                    f"订单状态：{data.get('status_text', order['order_status'])}\n"
                    f"下单时间：{order['order_time']}\n"
                    f"运费：¥{order['shipping_fee']:.2f}\n"
                    f"运费险：{insurance_text}\n"
                )
                if data.get("can_return"):
                    response += "\n该订单支持退换货申请，如需退换货请告诉我。"
                elif data.get("can_cancel"):
                    response += "\n该订单尚未发货，如需取消请告诉我。"
            elif result["success"] and not result["data"].get("found"):
                response = result["data"].get("message", "未找到该订单。")
            else:
                response = result.get("fallback", "订单查询服务暂时不可用，请稍后重试。")

        return {
            **state,
            "response": response,
            "need_human": False,
            "tool_calls": tool_calls,
            "tool_results": tool_results,
            "confidence": 0.9,
            "workflow_trace": trace,
        }

    except Exception as e:
        logger.error(f"订单查询节点异常: {e}")
        trace.append(f"order_query_node: error={e}")
        return {
            **state,
            "response": handle_agent_error(e),
            "need_human": True,
            "error": str(e),
            "workflow_trace": trace,
        }


# ============================================
# 退换货处理节点（优化点3：前置条件校验）
# ============================================
def return_process_node(state: AgentState) -> AgentState:
    """
    退换货处理节点
    流程：订单校验 → 规则匹配 → 资格判定 → 工单生成
    """
    user_input = state.get("user_input", "")
    slots = state.get("slots", {})
    context_summary = state.get("context_summary", "")
    trace = _add_trace(state, "return_process_node: start")

    try:
        # 优化点3：前置条件检查 - 必须有订单号
        order_id = slots.get("order_id")
        if not order_id:
            order_match = re.search(r"DD\d+", user_input)
            if order_match:
                order_id = order_match.group(0)

        if not order_id and context_summary:
            order_match = re.search(r"DD\d+", context_summary)
            if order_match:
                order_id = order_match.group(0)

        if not order_id:
            trace.append("return_process_node: no order_id, asking user")
            response = (
                "您好，请问您要退换货的订单号是多少呢？\n"
                "请提供订单号（格式如DD20250701001），"
                "我将为您核实退换货资格。"
            )
            return {
                **state,
                "response": response,
                "need_human": False,
                "workflow_trace": trace,
            }

        # 确定退换货类型
        return_type = slots.get("return_type", "")
        if not return_type:
            if any(kw in user_input for kw in ["换货", "换一个", "更换"]):
                return_type = "exchange"
            elif any(kw in user_input for kw in ["退款", "退钱"]):
                return_type = "refund"
            else:
                return_type = "return"

        # 获取退换货原因
        reason = slots.get("product_issue", "")
        if not reason:
            if any(kw in user_input for kw in ["质量", "坏了", "破损", "故障", "缺陷"]):
                reason = "商品质量问题"
            elif any(kw in user_input for kw in ["不想要", "不喜欢", "不合适"]):
                reason = "不想要了"
            elif any(kw in user_input for kw in ["尺码", "大小", "颜色"]):
                reason = "尺码/规格不合适"
            else:
                reason = "用户申请退换货"

        trace.append(f"return_process_node: order={order_id}, type={return_type}, reason={reason}")

        tool_calls = []
        tool_results = []

        # 1. 查询订单
        order_tool = _get_order_tool()
        order_result = order_tool.execute(order_id=order_id)
        tool_calls.append({
            "tool_name": "order_query",
            "tool_input": {"order_id": order_id},
            "success": order_result["success"],
        })
        tool_results.append(order_result)

        if not order_result["success"] or not order_result["data"].get("found"):
            response = order_result["data"].get("message", "未找到该订单，请确认订单号。") if order_result["success"] else order_result.get("fallback", "订单查询失败")
            return {
                **state,
                "response": response,
                "need_human": False,
                "tool_calls": tool_calls,
                "tool_results": tool_results,
                "workflow_trace": trace,
            }

        # 2. 创建工单
        ticket_tool = _get_ticket_tool()
        ticket_result = ticket_tool.execute(
            order_id=order_id,
            ticket_type=return_type,
            reason=reason,
        )
        tool_calls.append({
            "tool_name": "ticket_create",
            "tool_input": {"order_id": order_id, "ticket_type": return_type, "reason": reason},
            "success": ticket_result["success"],
        })
        tool_results.append(ticket_result)

        if not ticket_result["success"]:
            # 工单创建失败（可能是不符合条件或已有工单）
            data = ticket_result.get("data", {})
            response = data.get("message", "退换货申请失败，请联系人工客服。")
            if data.get("eligible") is False:
                response += "\n\n如需进一步帮助，我可以为您转接人工客服。"
            return {
                **state,
                "response": response,
                "need_human": data.get("eligible") is False,
                "tool_calls": tool_calls,
                "tool_results": tool_results,
                "workflow_trace": trace,
            }

        # 3. 计算退款金额（退货/退款时）
        ticket_data = ticket_result["data"]
        refund_info = ""
        if return_type in ["return", "refund"]:
            return_reason_param = "quality_issue" if "质量" in reason else "no_reason"
            refund_tool = _get_refund_tool()
            refund_result = refund_tool.execute(
                order_id=order_id,
                return_reason=return_reason_param,
            )
            tool_calls.append({
                "tool_name": "refund_calc",
                "tool_input": {"order_id": order_id, "return_reason": return_reason_param},
                "success": refund_result["success"],
            })
            tool_results.append(refund_result)

            if refund_result["success"]:
                refund_data = refund_result["data"]
                breakdown = refund_data.get("refund_breakdown", {})
                breakdown_text = "、".join(f"{k}: ¥{v:.2f}" for k, v in breakdown.items() if v > 0)
                refund_info = (
                    f"\n\n退款明细：\n"
                    f"  {breakdown_text}\n"
                    f"  预计退款总额：¥{refund_data.get('total_with_insurance', 0):.2f}\n"
                    f"  {refund_data.get('shipping_note', '')}\n"
                    f"  {refund_data.get('note', '')}"
                )

        # 生成回复
        type_text = ticket_data.get("ticket_type_text", return_type)
        response = (
            f"您的{type_text}申请已受理！\n\n"
            f"工单号：{ticket_data['ticket_id']}\n"
            f"订单号：{order_id}\n"
            f"商品名称：{ticket_data.get('product_name', '')}\n"
            f"退换货类型：{type_text}\n"
            f"原因：{reason}"
            f"{refund_info}"
        )

        if ticket_data.get("warning"):
            response += f"\n\n注意：{ticket_data['warning']}"

        response += "\n\n我们将在1-3个工作日内处理您的申请，请保持手机畅通。"

        trace.append("return_process_node: ticket created successfully")

        return {
            **state,
            "response": response,
            "need_human": False,
            "tool_calls": tool_calls,
            "tool_results": tool_results,
            "confidence": 0.95,
            "workflow_trace": trace,
        }

    except Exception as e:
        logger.error(f"退换货处理节点异常: {e}")
        trace.append(f"return_process_node: error={e}")
        return {
            **state,
            "response": handle_agent_error(e),
            "need_human": True,
            "error": str(e),
            "workflow_trace": trace,
        }


# ============================================
# 情绪安抚节点
# ============================================
def complaint_node(state: AgentState) -> AgentState:
    """情绪安抚 + 自动转人工"""
    user_input = state.get("user_input", "")
    slots = state.get("slots", {})
    emotion = slots.get("user_emotion", "unhappy")
    trace = _add_trace(state, "complaint_node: start")

    try:
        prompt = format_prompt(
            EMOTION_COMFORT_PROMPT,
            user_input=user_input,
            emotion_state=emotion,
        )

        # 流式模式：存储 prompt，不调用 LLM（尾部追加的转人工话术在 run_agent_stream 中处理）
        if state.get("_stream_mode"):
            trace.append("complaint_node: stream mode, prompt stored")
            return {
                **state,
                "response": "",
                "_stream_prompt": prompt,
                "_stream_system_prompt": SYSTEM_PROMPT,
                "need_human": True,
                "confidence": 0.8,
                "workflow_trace": trace,
            }

        qwen_client = get_qwen_client()
        response = qwen_client.chat(
            messages=[HumanMessage(content=prompt)],
            system_prompt=SYSTEM_PROMPT,
        )

        # 情绪问题一律转人工
        response += "\n\n已为您转接人工客服，客服人员将尽快为您处理。"

        trace.append("complaint_node: comfort response generated, handoff to human")

        return {
            **state,
            "response": response,
            "need_human": True,
            "confidence": 0.8,
            "workflow_trace": trace,
        }

    except Exception as e:
        logger.error(f"情绪安抚节点异常: {e}")
        trace.append(f"complaint_node: error={e}")
        return {
            **state,
            "response": (
                "非常抱歉给您带来了不好的体验，我已为您转接人工客服，"
                "客服人员将尽快为您处理。"
            ),
            "need_human": True,
            "error": str(e),
            "workflow_trace": trace,
        }


# ============================================
# 超范围处理节点
# ============================================
def out_of_scope_node(state: AgentState) -> AgentState:
    """礼貌告知超范围 + 引导"""
    user_input = state.get("user_input", "")
    trace = _add_trace(state, "out_of_scope_node: start")

    try:
        prompt = format_prompt(OUT_OF_SCOPE_PROMPT, user_input=user_input)

        # 流式模式：存储 prompt，不调用 LLM
        if state.get("_stream_mode"):
            trace.append("out_of_scope_node: stream mode, prompt stored")
            return {
                **state,
                "response": "",
                "_stream_prompt": prompt,
                "_stream_system_prompt": SYSTEM_PROMPT,
                "need_human": False,
                "confidence": 0.85,
                "workflow_trace": trace,
            }

        qwen_client = get_qwen_client()
        response = qwen_client.chat(
            messages=[HumanMessage(content=prompt)],
            system_prompt=SYSTEM_PROMPT,
        )

        trace.append("out_of_scope_node: response generated")

        return {
            **state,
            "response": response,
            "need_human": False,
            "confidence": 0.85,
            "workflow_trace": trace,
        }

    except Exception as e:
        logger.error(f"超范围处理节点异常: {e}")
        trace.append(f"out_of_scope_node: error={e}")
        return {
            **state,
            "response": (
                "抱歉，该问题超出了我的服务范围。我是售后智能助手，"
                "可以为您处理退换货、物流查询、售后政策咨询等问题。"
                "如需其他帮助，请联系对应客服。"
            ),
            "need_human": False,
            "error": str(e),
            "workflow_trace": trace,
        }


# ============================================
# 结果校验节点（优化点2：反幻觉校验）
# ============================================
def validate_node(state: AgentState) -> AgentState:
    """
    结果校验节点
    - 政策类回复必须绑定知识库来源
    - 置信度不足时转人工
    """
    trace = _add_trace(state, "validate_node: start")
    settings = get_settings()

    intent = state.get("intent", "")
    confidence = state.get("confidence", 0.0)
    is_retrieval_reliable = state.get("is_retrieval_reliable", True)
    need_human = state.get("need_human", False)

    # 政策类回复校验
    if intent == INTENT_POLICY_QA:
        if not is_retrieval_reliable:
            trace.append("validate_node: policy answer not reliable, handoff to human")
            return {
                **state,
                "response": (
                    "抱歉，我暂时无法确认该问题的准确答案。"
                    "已为您转接人工客服进行核实，请稍候。"
                ),
                "need_human": True,
                "_stream_prompt": None,
                "workflow_trace": trace,
            }

    # 整体置信度校验
    if confidence < settings.human_handoff_threshold and not need_human:
        trace.append(f"validate_node: low confidence ({confidence:.2f}), handoff to human")
        return {
            **state,
            "response": (
                "抱歉，我对这个问题的处理信心不足，"
                "为了给您提供准确的服务，已为您转接人工客服。"
            ),
            "need_human": True,
            "_stream_prompt": None,
            "workflow_trace": trace,
        }

    trace.append(f"validate_node: passed (confidence={confidence:.2f})")
    return {
        **state,
        "workflow_trace": trace,
    }


# ============================================
# 记忆摘要节点（优化点4：多轮记忆）
# ============================================
def memory_summary_node(state: AgentState) -> AgentState:
    """每 N 轮对话触发上下文摘要"""
    messages = state.get("messages", [])
    context_summary = state.get("context_summary", "")
    turn_count = state.get("turn_count", 0) + 1
    trace = _add_trace(state, "memory_summary_node: start")

    try:
        memory = get_memory()

        # 更新摘要
        new_summary = memory.update_summary(messages, context_summary, turn_count)

        if new_summary != context_summary:
            trace.append("memory_summary_node: summary updated")
        else:
            trace.append(f"memory_summary_node: no update (turn={turn_count})")

        return {
            **state,
            "context_summary": new_summary,
            "turn_count": turn_count,
            "workflow_trace": trace,
        }

    except Exception as e:
        logger.error(f"记忆摘要节点异常: {e}")
        trace.append(f"memory_summary_node: error={e}")
        return {
            **state,
            "turn_count": turn_count,
            "workflow_trace": trace,
        }

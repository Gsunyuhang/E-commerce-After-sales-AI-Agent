"""
LangGraph 工作流编排
"感知-规划-执行-校验"完整 Agent 工作流
"""

from typing import Optional

from langgraph.graph import END, StateGraph

from agent.intent import (
    INTENT_COMPLAINT,
    INTENT_ORDER_QUERY,
    INTENT_OUT_OF_SCOPE,
    INTENT_POLICY_QA,
    INTENT_RETURN_PROCESS,
    intent_recognition_node,
)
from agent.memory import get_memory
from agent.nodes import (
    complaint_node,
    memory_summary_node,
    order_query_node,
    out_of_scope_node,
    policy_qa_node,
    return_process_node,
    validate_node,
)
from agent.state import AgentState
from utils.logger import logger


def route_by_intent(state: AgentState) -> str:
    """
    根据意图识别结果路由到对应节点

    Returns:
        下一个节点名称
    """
    intent = state.get("intent", INTENT_OUT_OF_SCOPE)
    need_human = state.get("need_human", False)

    # 意图识别置信度极低时直接转人工
    if need_human and intent == INTENT_OUT_OF_SCOPE:
        return "out_of_scope"

    route_map = {
        INTENT_POLICY_QA: "policy_qa",
        INTENT_ORDER_QUERY: "order_query",
        INTENT_RETURN_PROCESS: "return_process",
        INTENT_COMPLAINT: "complaint",
        INTENT_OUT_OF_SCOPE: "out_of_scope",
    }

    next_node = route_map.get(intent, "out_of_scope")
    logger.info(f"意图路由: intent={intent} -> node={next_node}")
    return next_node


def should_validate(state: AgentState) -> str:
    """
    判断是否需要经过校验节点

    Returns:
        "validate" 或 END
    """
    intent = state.get("intent", "")
    # 政策答疑和退换货流程需要校验
    if intent in [INTENT_POLICY_QA, INTENT_RETURN_PROCESS]:
        return "validate"
    # 其他直接结束
    return END


def build_agent_graph():
    """
    构建 Agent 工作流图

    图结构：
    START → intent_recognition → route_by_intent
      ├→ policy_qa → validate → END
      ├→ order_query → END
      ├→ return_process → validate → END
      ├→ complaint → END
      └→ out_of_scope → END

    Returns:
        编译后的 LangGraph 可执行图
    """
    graph = StateGraph(AgentState)

    # 添加节点
    graph.add_node("intent_recognition", intent_recognition_node)
    graph.add_node("policy_qa", policy_qa_node)
    graph.add_node("order_query", order_query_node)
    graph.add_node("return_process", return_process_node)
    graph.add_node("complaint", complaint_node)
    graph.add_node("out_of_scope", out_of_scope_node)
    graph.add_node("validate", validate_node)
    graph.add_node("memory_summary", memory_summary_node)

    # 设置入口
    graph.set_entry_point("intent_recognition")

    # 意图识别后的条件路由
    graph.add_conditional_edges(
        "intent_recognition",
        route_by_intent,
        {
            "policy_qa": "policy_qa",
            "order_query": "order_query",
            "return_process": "return_process",
            "complaint": "complaint",
            "out_of_scope": "out_of_scope",
        },
    )

    # 政策答疑和退换货处理后经过校验
    graph.add_conditional_edges(
        "policy_qa",
        should_validate,
        {
            "validate": "validate",
            END: END,
        },
    )

    graph.add_conditional_edges(
        "return_process",
        should_validate,
        {
            "validate": "validate",
            END: END,
        },
    )

    # 校验后进入记忆摘要
    graph.add_edge("validate", "memory_summary")

    # 订单查询、情绪安抚、超范围处理后直接进入记忆摘要
    graph.add_edge("order_query", "memory_summary")
    graph.add_edge("complaint", "memory_summary")
    graph.add_edge("out_of_scope", "memory_summary")

    # 记忆摘要后结束
    graph.add_edge("memory_summary", END)

    # 编译
    compiled = graph.compile()
    logger.info("LangGraph Agent 工作流图构建完成")
    return compiled


# 全局图实例
_agent_graph = None


def get_agent_graph():
    """获取编译后的 Agent 图（单例）"""
    global _agent_graph
    if _agent_graph is None:
        _agent_graph = build_agent_graph()
    return _agent_graph


def run_agent(
    user_input: str,
    session_id: str,
    messages: list = None,
    context_summary: str = "",
    turn_count: int = 0,
) -> dict:
    """
    运行 Agent 处理用户输入

    Args:
        user_input: 用户输入文本
        session_id: 会话 ID
        messages: 历史消息列表
        context_summary: 上下文摘要
        turn_count: 当前轮次

    Returns:
        Agent 处理结果字典，包含 response, intent, tool_calls, workflow_trace 等
    """
    from langchain_core.messages import HumanMessage

    import time as time_module

    start_time = time_module.time()

    if messages is None:
        messages = []

    # 构建初始状态
    initial_state: AgentState = {
        "user_input": user_input,
        "session_id": session_id,
        "messages": messages,
        "context_summary": context_summary,
        "turn_count": turn_count,
        "tool_calls": [],
        "tool_results": [],
        "workflow_trace": [],
        "need_human": False,
        "confidence": 0.0,
    }

    try:
        graph = get_agent_graph()
        final_state = graph.invoke(initial_state)

        response_time = time_module.time() - start_time

        # 构建结果
        result = {
            "response": final_state.get("response", ""),
            "intent": final_state.get("intent", ""),
            "intent_confidence": final_state.get("intent_confidence", 0.0),
            "slots": final_state.get("slots", {}),
            "need_human": final_state.get("need_human", False),
            "confidence": final_state.get("confidence", 0.0),
            "tool_calls": final_state.get("tool_calls", []),
            "tool_results": final_state.get("tool_results", []),
            "retrieved_docs": final_state.get("retrieved_docs", []),
            "is_retrieval_reliable": final_state.get("is_retrieval_reliable", True),
            "retrieval_max_score": final_state.get("retrieval_max_score", 0.0),
            "context_summary": final_state.get("context_summary", context_summary),
            "turn_count": final_state.get("turn_count", turn_count + 1),
            "response_time": response_time,
            "workflow_trace": final_state.get("workflow_trace", []),
            "error": final_state.get("error"),
        }

        logger.info(
            f"Agent 处理完成: session={session_id}, intent={result['intent']}, "
            f"time={response_time:.2f}s, human={result['need_human']}"
        )

        return result

    except Exception as e:
        logger.error(f"Agent 运行异常: {e}")
        response_time = time_module.time() - start_time
        return {
            "response": "抱歉，系统暂时遇到问题，正在为您转接人工客服。",
            "intent": "error",
            "intent_confidence": 0.0,
            "slots": {},
            "need_human": True,
            "confidence": 0.0,
            "tool_calls": [],
            "tool_results": [],
            "retrieved_docs": [],
            "is_retrieval_reliable": False,
            "retrieval_max_score": 0.0,
            "context_summary": context_summary,
            "turn_count": turn_count + 1,
            "response_time": response_time,
            "workflow_trace": [f"error: {e}"],
            "error": str(e),
        }


def run_agent_stream(
    user_input: str,
    session_id: str,
    messages: list = None,
    context_summary: str = "",
    turn_count: int = 0,
):
    """
    流式运行 Agent，yield 事件字典

    Yields:
        {"type": "progress", "step": "...", "message": "..."}  — 进度提示
        {"type": "intent", "intent": "...", "confidence": ...}  — 意图识别结果
        {"type": "chunk", "content": "..."}                     — 流式回答片段
        {"type": "done", "result": {...}}                        — 完整结果
    """
    import time as time_module
    from langchain_core.messages import HumanMessage
    from llm.qwen_client import get_qwen_client

    start_time = time_module.time()

    if messages is None:
        messages = []

    initial_state: AgentState = {
        "user_input": user_input,
        "session_id": session_id,
        "messages": messages,
        "context_summary": context_summary,
        "turn_count": turn_count,
        "tool_calls": [],
        "tool_results": [],
        "workflow_trace": [],
        "need_human": False,
        "confidence": 0.0,
        "_stream_mode": True,
        "_stream_prompt": None,
        "_stream_system_prompt": None,
    }

    # 节点名 -> (step, message)
    progress_map = {
        "intent_recognition": ("intent", "正在识别意图..."),
        "policy_qa": ("retrieval", "正在检索知识库..."),
        "order_query": ("querying", "正在查询订单..."),
        "return_process": ("processing", "正在处理退换货..."),
        "complaint": ("comforting", "正在生成回复..."),
        "out_of_scope": ("generating", "正在生成回复..."),
        "validate": ("validating", "正在校验结果..."),
    }

    try:
        graph = get_agent_graph()

        # 用 graph.stream() 获取每个节点执行后的进度
        accumulated_state = dict(initial_state)
        final_state = None

        for chunk in graph.stream(initial_state):
            for node_name, node_state in chunk.items():
                accumulated_state.update(node_state)
                final_state = dict(accumulated_state)

                # 发送进度事件
                if node_name in progress_map:
                    step, message = progress_map[node_name]
                    yield {"type": "progress", "step": step, "message": message}

                # 意图识别完成后发送意图
                if node_name == "intent_recognition":
                    yield {
                        "type": "intent",
                        "intent": node_state.get("intent", ""),
                        "confidence": node_state.get("intent_confidence", 0.0),
                    }

        if final_state is None:
            final_state = accumulated_state

        # 如果有待流式生成的 prompt，调用 LLM 流式输出
        stream_prompt = final_state.get("_stream_prompt")
        if stream_prompt:
            yield {"type": "progress", "step": "generating", "message": "正在生成回答..."}

            qwen_client = get_qwen_client()
            system_prompt = final_state.get("_stream_system_prompt")
            full_response = ""

            for text_chunk in qwen_client.chat_stream(
                messages=[HumanMessage(content=stream_prompt)],
                system_prompt=system_prompt,
            ):
                full_response += text_chunk
                yield {"type": "chunk", "content": text_chunk}

            # complaint 节点需要追加转人工话术
            if final_state.get("need_human") and final_state.get("intent") == "complaint":
                suffix = "\n\n已为您转接人工客服，客服人员将尽快为您处理。"
                full_response += suffix
                yield {"type": "chunk", "content": suffix}

            final_state["response"] = full_response
        else:
            # 非流式响应（工具查询等），直接发送完整 response
            response_text = final_state.get("response", "")
            if response_text:
                yield {"type": "chunk", "content": response_text}

        # 构建最终结果
        response_time = time_module.time() - start_time
        result = {
            "response": final_state.get("response", ""),
            "intent": final_state.get("intent", ""),
            "intent_confidence": final_state.get("intent_confidence", 0.0),
            "slots": final_state.get("slots", {}),
            "need_human": final_state.get("need_human", False),
            "confidence": final_state.get("confidence", 0.0),
            "tool_calls": final_state.get("tool_calls", []),
            "tool_results": final_state.get("tool_results", []),
            "retrieved_docs": final_state.get("retrieved_docs", []),
            "is_retrieval_reliable": final_state.get("is_retrieval_reliable", True),
            "retrieval_max_score": final_state.get("retrieval_max_score", 0.0),
            "context_summary": final_state.get("context_summary", context_summary),
            "turn_count": final_state.get("turn_count", turn_count + 1),
            "response_time": response_time,
            "workflow_trace": final_state.get("workflow_trace", []),
            "error": final_state.get("error"),
        }

        logger.info(
            f"Agent 流式处理完成: session={session_id}, intent={result['intent']}, "
            f"time={response_time:.2f}s, human={result['need_human']}"
        )

        yield {"type": "done", "result": result}

    except Exception as e:
        logger.error(f"Agent 流式运行异常: {e}")
        response_time = time_module.time() - start_time
        yield {
            "type": "done",
            "result": {
                "response": "抱歉，系统暂时遇到问题，正在为您转接人工客服。",
                "intent": "error",
                "intent_confidence": 0.0,
                "slots": {},
                "need_human": True,
                "confidence": 0.0,
                "tool_calls": [],
                "tool_results": [],
                "retrieved_docs": [],
                "is_retrieval_reliable": False,
                "retrieval_max_score": 0.0,
                "context_summary": context_summary,
                "turn_count": turn_count + 1,
                "response_time": response_time,
                "workflow_trace": [f"error: {e}"],
                "error": str(e),
            },
        }

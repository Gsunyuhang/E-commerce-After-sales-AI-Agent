"""
意图识别模块
通过 LLM + 结构化 Prompt 精准分类 5 种意图，同时提取槽位信息
"""

import json
import re
from typing import Optional

from langchain_core.messages import HumanMessage

from agent.state import AgentState, SlotInfo
from config.prompts import INTENT_RECOGNITION_PROMPT, format_prompt
from config.settings import get_settings
from llm.qwen_client import get_qwen_client
from utils.exception import IntentRecognitionError
from utils.logger import logger


# 意图类型常量
INTENT_POLICY_QA = "policy_qa"
INTENT_ORDER_QUERY = "order_query"
INTENT_RETURN_PROCESS = "return_process"
INTENT_COMPLAINT = "complaint"
INTENT_OUT_OF_SCOPE = "out_of_scope"

VALID_INTENTS = {
    INTENT_POLICY_QA,
    INTENT_ORDER_QUERY,
    INTENT_RETURN_PROCESS,
    INTENT_COMPLAINT,
    INTENT_OUT_OF_SCOPE,
}


class IntentRecognizer:
    """意图识别器"""

    def __init__(self):
        self.qwen_client = get_qwen_client()
        self.settings = get_settings()

    def _build_conversation_history(self, messages: list) -> str:
        """构建对话历史文本"""
        if not messages:
            return "无"

        history_parts = []
        # 只取最近几轮
        recent = messages[-6:] if len(messages) > 6 else messages
        for msg in recent:
            role = "用户" if isinstance(msg, HumanMessage) else "客服"
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            history_parts.append(f"{role}: {content}")

        return "\n".join(history_parts)

    def _parse_llm_response(self, response: str) -> dict:
        """解析 LLM 返回的 JSON 结果"""
        try:
            # 尝试直接解析 JSON
            # 先尝试提取 JSON 块
            json_match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # 尝试直接找 JSON 对象
                json_match = re.search(r"\{.*\}", response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    json_str = response

            result = json.loads(json_str)
            return result
        except json.JSONDecodeError as e:
            logger.error(f"意图识别 JSON 解析失败: {e}, response: {response}")
            return {}

    def recognize(self, user_input: str, messages: list) -> dict:
        """
        识别用户意图

        Args:
            user_input: 用户输入
            messages: 对话历史

        Returns:
            包含 intent, confidence, slots 的字典
        """
        try:
            conversation_history = self._build_conversation_history(messages)

            prompt = format_prompt(
                INTENT_RECOGNITION_PROMPT,
                user_input=user_input,
                conversation_history=conversation_history,
            )

            response = self.qwen_client.chat(
                messages=[HumanMessage(content=prompt)],
            )

            result = self._parse_llm_response(response)

            if not result or "intent" not in result:
                logger.warning(f"意图识别结果异常，默认转为人工: {response}")
                return {
                    "intent": INTENT_OUT_OF_SCOPE,
                    "confidence": 0.0,
                    "slots": {},
                    "need_human": True,
                    "reasoning": "意图识别失败，转人工处理",
                }

            intent = result.get("intent", "").strip()
            if intent not in VALID_INTENTS:
                logger.warning(f"未知意图类型: {intent}，默认转为人工")
                intent = INTENT_OUT_OF_SCOPE
                result["need_human"] = True

            confidence = float(result.get("confidence", 0.5))
            slots = result.get("slots", {})
            reasoning = result.get("reasoning", "")

            # 清理 slots 中的 null 值
            cleaned_slots = {}
            for key, value in slots.items():
                if value and value != "null" and value != "None":
                    cleaned_slots[key] = value

            logger.info(
                f"意图识别: intent={intent}, confidence={confidence:.2f}, "
                f"slots={cleaned_slots}, reasoning={reasoning}"
            )

            return {
                "intent": intent,
                "confidence": confidence,
                "slots": cleaned_slots,
                "need_human": confidence < self.settings.intent_confidence_threshold,
                "reasoning": reasoning,
            }

        except Exception as e:
            logger.error(f"意图识别异常: {e}")
            return {
                "intent": INTENT_OUT_OF_SCOPE,
                "confidence": 0.0,
                "slots": {},
                "need_human": True,
                "reasoning": f"意图识别异常: {e}",
            }


# 全局单例
_intent_recognizer: Optional[IntentRecognizer] = None


def get_intent_recognizer() -> IntentRecognizer:
    """获取 IntentRecognizer 单例"""
    global _intent_recognizer
    if _intent_recognizer is None:
        _intent_recognizer = IntentRecognizer()
    return _intent_recognizer


def intent_recognition_node(state: AgentState) -> AgentState:
    """
    LangGraph 意图识别节点

    从 state 中读取 user_input 和 messages，
    调用意图识别器，将结果写回 state
    """
    user_input = state.get("user_input", "")
    messages = state.get("messages", [])

    recognizer = get_intent_recognizer()
    result = recognizer.recognize(user_input, messages)

    # 更新状态
    trace = state.get("workflow_trace", [])
    trace.append(f"intent_recognition: intent={result['intent']}, confidence={result['confidence']:.2f}")

    new_state = {
        **state,
        "intent": result["intent"],
        "intent_confidence": result["confidence"],
        "slots": result["slots"],
        "need_human": result.get("need_human", False),
        "workflow_trace": trace,
    }

    return new_state

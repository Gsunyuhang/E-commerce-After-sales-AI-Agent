"""
Agent 测试
测试意图识别、分支流程、转人工逻辑
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.intent import (
    INTENT_COMPLAINT,
    INTENT_ORDER_QUERY,
    INTENT_OUT_OF_SCOPE,
    INTENT_POLICY_QA,
    INTENT_RETURN_PROCESS,
    VALID_INTENTS,
)
from agent.state import AgentState
from agent.graph import route_by_intent, should_validate
from config.prompts import format_prompt, INTENT_RECOGNITION_PROMPT, QUERY_REWRITE_PROMPT
from langgraph.graph import END


class TestIntentConstants:
    """意图常量测试"""

    def test_valid_intents(self):
        """测试意图常量"""
        assert INTENT_POLICY_QA == "policy_qa"
        assert INTENT_ORDER_QUERY == "order_query"
        assert INTENT_RETURN_PROCESS == "return_process"
        assert INTENT_COMPLAINT == "complaint"
        assert INTENT_OUT_OF_SCOPE == "out_of_scope"

    def test_all_intents_in_valid_set(self):
        """测试所有意图都在有效集合中"""
        assert VALID_INTENTS == {
            INTENT_POLICY_QA,
            INTENT_ORDER_QUERY,
            INTENT_RETURN_PROCESS,
            INTENT_COMPLAINT,
            INTENT_OUT_OF_SCOPE,
        }


class TestRoutingLogic:
    """路由逻辑测试（不需要 LLM）"""

    def test_route_policy_qa(self):
        """测试政策答疑路由"""
        state: AgentState = {"intent": INTENT_POLICY_QA, "need_human": False}
        assert route_by_intent(state) == "policy_qa"

    def test_route_order_query(self):
        """测试订单查询路由"""
        state: AgentState = {"intent": INTENT_ORDER_QUERY, "need_human": False}
        assert route_by_intent(state) == "order_query"

    def test_route_return_process(self):
        """测试退换货处理路由"""
        state: AgentState = {"intent": INTENT_RETURN_PROCESS, "need_human": False}
        assert route_by_intent(state) == "return_process"

    def test_route_complaint(self):
        """测试投诉路由"""
        state: AgentState = {"intent": INTENT_COMPLAINT, "need_human": False}
        assert route_by_intent(state) == "complaint"

    def test_route_out_of_scope(self):
        """测试超范围路由"""
        state: AgentState = {"intent": INTENT_OUT_OF_SCOPE, "need_human": False}
        assert route_by_intent(state) == "out_of_scope"

    def test_route_unknown_intent(self):
        """测试未知意图默认路由"""
        state: AgentState = {"intent": "unknown", "need_human": False}
        assert route_by_intent(state) == "out_of_scope"


class TestValidationLogic:
    """校验逻辑测试"""

    def test_should_validate_policy_qa(self):
        """测试政策答疑需要校验"""
        state: AgentState = {"intent": INTENT_POLICY_QA}
        assert should_validate(state) == "validate"

    def test_should_validate_return_process(self):
        """测试退换货流程需要校验"""
        state: AgentState = {"intent": INTENT_RETURN_PROCESS}
        assert should_validate(state) == "validate"

    def test_should_not_validate_order_query(self):
        """测试订单查询不需要校验"""
        state: AgentState = {"intent": INTENT_ORDER_QUERY}
        assert should_validate(state) == END

    def test_should_not_validate_complaint(self):
        """测试投诉不需要校验"""
        state: AgentState = {"intent": INTENT_COMPLAINT}
        assert should_validate(state) == END


class TestPromptFormatting:
    """Prompt 格式化测试"""

    def test_format_intent_prompt(self):
        """测试意图识别 Prompt 格式化"""
        prompt = format_prompt(
            INTENT_RECOGNITION_PROMPT,
            user_input="我要退货",
            conversation_history="无",
        )
        assert "我要退货" in prompt
        assert "policy_qa" in prompt  # 意图类型定义存在

    def test_format_query_rewrite_prompt(self):
        """测试 Query Rewrite Prompt 格式化"""
        prompt = format_prompt(
            QUERY_REWRITE_PROMPT,
            original_query="东西坏了能换吗",
        )
        assert "东西坏了能换吗" in prompt

    def test_format_missing_param(self):
        """测试缺少参数时的异常"""
        with pytest.raises(ValueError):
            format_prompt(INTENT_RECOGNITION_PROMPT, user_input="test")


class TestAgentState:
    """Agent 状态测试"""

    def test_state_has_required_fields(self):
        """测试状态包含必要字段"""
        state: AgentState = {
            "user_input": "测试",
            "session_id": "test_session",
            "messages": [],
            "intent": "",
            "need_human": False,
            "workflow_trace": [],
        }
        assert "user_input" in state
        assert "session_id" in state
        assert "messages" in state
        assert "intent" in state
        assert "need_human" in state

    def test_state_optional_fields(self):
        """测试状态可选字段"""
        state: AgentState = {
            "user_input": "测试",
            "session_id": "test_session",
            "tool_calls": [],
            "tool_results": [],
            "retrieved_docs": [],
            "confidence": 0.0,
            "context_summary": "",
            "turn_count": 0,
        }
        assert state["tool_calls"] == []
        assert state["confidence"] == 0.0

"""
Pydantic 请求/响应模型
"""

from typing import Any, Optional

from pydantic import BaseModel, Field


# ============================================
# 对话相关
# ============================================

class ChatMessage(BaseModel):
    """对话消息"""
    role: str = Field(..., description="角色: user/assistant")
    content: str = Field(..., description="消息内容")


class ChatRequest(BaseModel):
    """对话请求"""
    session_id: str = Field(..., description="会话 ID")
    message: str = Field(..., description="用户输入")
    history: list[ChatMessage] = Field(default=[], description="对话历史")
    context_summary: str = Field(default="", description="上下文摘要")
    turn_count: int = Field(default=0, description="当前轮次")


class ToolCallInfo(BaseModel):
    """工具调用信息"""
    tool_name: str
    tool_input: dict
    success: bool


class RetrievedDocInfo(BaseModel):
    """检索文档信息"""
    original_query: str = ""
    rewritten_query: str = ""
    doc_count: int = 0
    max_score: float = 0.0
    is_reliable: bool = True
    sources: list[dict] = []


class ChatResponse(BaseModel):
    """对话响应"""
    session_id: str
    response: str
    intent: str = ""
    intent_confidence: float = 0.0
    need_human: bool = False
    confidence: float = 0.0
    tool_calls: list[ToolCallInfo] = []
    retrieved_docs: list[Any] = []
    is_retrieval_reliable: bool = True
    retrieval_max_score: float = 0.0
    context_summary: str = ""
    turn_count: int = 0
    response_time: float = 0.0
    workflow_trace: list[str] = []
    error: Optional[str] = None


class ChatHistoryResponse(BaseModel):
    """对话历史响应"""
    session_id: str
    messages: list[ChatMessage] = []


# ============================================
# 订单相关
# ============================================

class OrderResponse(BaseModel):
    """订单查询响应"""
    found: bool
    order: Optional[dict] = None
    status_text: Optional[str] = None
    can_return: Optional[bool] = None
    can_cancel: Optional[bool] = None
    has_insurance: Optional[bool] = None
    message: Optional[str] = None


# ============================================
# 评估相关
# ============================================

class EvalRequest(BaseModel):
    """评估请求"""
    test_cases: Optional[list[dict]] = Field(default=None, description="自定义测试用例（为空则使用默认测试集）")


class EvalResult(BaseModel):
    """单条评估结果"""
    test_case_id: str
    input: str
    expected_intent: str
    actual_intent: str
    expected_outcome: str
    actual_response: str
    response_time: float
    need_human: bool
    tool_calls: list[dict] = []
    passed: bool
    notes: str = ""


class EvalReport(BaseModel):
    """评估报告"""
    total_cases: int
    resolution_rate: float
    refusal_accuracy: float
    avg_response_time: float
    tool_call_success_rate: float
    human_handoff_rate: float
    category_stats: dict = {}
    results: list[EvalResult] = []

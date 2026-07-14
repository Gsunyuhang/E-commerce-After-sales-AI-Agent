"""
统一异常处理与降级策略
自定义异常体系 + 降级话术，确保系统在异常情况下不崩溃
"""

from typing import Optional


class AgentBaseError(Exception):
    """Agent 基础异常类"""

    def __init__(self, message: str, fallback_response: Optional[str] = None):
        super().__init__(message)
        self.fallback_response = fallback_response or "抱歉，系统暂时遇到问题，请稍后重试或联系人工客服。"


class LLMError(AgentBaseError):
    """LLM 调用异常"""

    def __init__(self, message: str, fallback_response: Optional[str] = None):
        super().__init__(
            message,
            fallback_response or "AI 服务暂时不可用，正在为您转接人工客服，请稍候。",
        )


class RetrievalError(AgentBaseError):
    """知识库检索异常"""

    def __init__(self, message: str, fallback_response: Optional[str] = None):
        super().__init__(
            message,
            fallback_response or "知识检索服务暂时不可用，正在为您转接人工客服，请稍候。",
        )


class ToolError(AgentBaseError):
    """工具调用异常"""

    def __init__(self, tool_name: str, message: str, fallback_response: Optional[str] = None):
        self.tool_name = tool_name
        super().__init__(
            f"工具 [{tool_name}] 调用失败: {message}",
            fallback_response or f"查询服务暂时不可用，请您提供相关信息，人工客服将为您处理。",
        )


class DatabaseError(AgentBaseError):
    """数据库操作异常"""

    def __init__(self, message: str, fallback_response: Optional[str] = None):
        super().__init__(
            message,
            fallback_response or "数据服务暂时不可用，正在为您转接人工客服，请稍候。",
        )


class IntentRecognitionError(AgentBaseError):
    """意图识别异常"""

    def __init__(self, message: str, fallback_response: Optional[str] = None):
        super().__init__(
            message,
            fallback_response or "抱歉，我无法理解您的问题，正在为您转接人工客服。",
        )


# --- 降级话术模板 ---

FALLBACK_RESPONSES = {
    "llm_timeout": "AI 服务响应超时，正在为您转接人工客服，请稍候。",
    "llm_error": "AI 服务暂时不可用，正在为您转接人工客服，请稍候。",
    "retrieval_empty": "抱歉，我未能找到相关的售后政策信息，已为您转接人工客服进行核实。",
    "tool_order_failed": "订单查询服务暂时不可用，请您提供订单号，人工客服将为您查询。",
    "tool_logistics_failed": "物流查询服务暂时不可用，请您稍后重试或联系人工客服。",
    "tool_ticket_failed": "工单创建失败，正在为您转接人工客服处理。",
    "tool_refund_failed": "退款计算服务暂时不可用，人工客服将为您核算退款金额。",
    "db_error": "数据服务暂时不可用，正在为您转接人工客服，请稍候。",
    "general_error": "抱歉，系统暂时遇到问题，正在为您转接人工客服。",
    "out_of_scope": "抱歉，该问题超出了我的服务范围。我是售后智能助手，可以为您处理退换货、物流查询、售后政策咨询等问题。如需其他帮助，请联系对应客服。",
}


def get_fallback_response(error_type: str) -> str:
    """
    根据错误类型获取降级话术

    Args:
        error_type: 错误类型标识

    Returns:
        降级回复文本
    """
    return FALLBACK_RESPONSES.get(error_type, FALLBACK_RESPONSES["general_error"])


def handle_agent_error(error: Exception) -> str:
    """
    统一异常处理入口，返回用户可读的降级话术

    Args:
        error: 捕获的异常

    Returns:
        降级回复文本
    """
    if isinstance(error, AgentBaseError):
        return error.fallback_response
    return FALLBACK_RESPONSES["general_error"]

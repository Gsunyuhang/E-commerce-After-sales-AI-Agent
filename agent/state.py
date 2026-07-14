"""
Agent 状态定义
定义 LangGraph 工作流中各节点共享的状态结构
"""

from typing import Any, Optional, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage


class SlotInfo(TypedDict, total=False):
    """槽位信息"""
    order_id: Optional[str]
    return_type: Optional[str]  # return / exchange / refund
    product_issue: Optional[str]
    user_emotion: Optional[str]  # neutral / unhappy / angry


class RetrievedDocInfo(TypedDict, total=False):
    """检索文档信息"""
    filename: str
    score: float
    preview: str


class ToolCallRecord(TypedDict, total=False):
    """工具调用记录"""
    tool_name: str
    tool_input: dict
    tool_output: dict
    success: bool


class AgentState(TypedDict, total=False):
    """
    LangGraph Agent 状态

    所有节点共享此状态，通过读写实现信息传递
    """

    # --- 对话相关 ---
    messages: list[BaseMessage]
    """对话历史消息列表"""

    session_id: str
    """会话 ID"""

    user_input: str
    """当前用户输入"""

    # --- 意图识别 ---
    intent: str
    """识别出的意图: policy_qa / order_query / return_process / complaint / out_of_scope"""

    intent_confidence: float
    """意图识别置信度"""

    slots: SlotInfo
    """槽位信息"""

    # --- 检索相关 ---
    retrieved_docs: list[RetrievedDocInfo]
    """检索到的知识库片段"""

    retrieval_max_score: float
    """检索最高置信度"""

    is_retrieval_reliable: bool
    """检索结果是否可靠"""

    # --- 工具调用 ---
    tool_calls: list[ToolCallRecord]
    """工具调用记录"""

    tool_results: list[dict]
    """工具返回结果"""

    # --- 回复相关 ---
    response: str
    """Agent 最终回复"""

    need_human: bool
    """是否需要转人工"""

    confidence: float
    """整体置信度"""

    # --- 多轮记忆 ---
    context_summary: str
    """多轮对话上下文摘要"""

    turn_count: int
    """当前会话轮次计数"""

    # --- 元数据 ---
    response_time: float
    """响应耗时（秒）"""

    error: Optional[str]
    """错误信息"""

    workflow_trace: list[str]
    """工作流执行轨迹，用于调试和展示"""

    # --- 流式模式 ---
    _stream_mode: bool
    """是否启用流式输出模式"""

    _stream_prompt: Optional[str]
    """流式模式下待生成的 prompt（节点生成后由 run_agent_stream 发送给 LLM）"""

    _stream_system_prompt: Optional[str]
    """流式模式下的 system prompt"""

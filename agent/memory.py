"""
多轮对话记忆模块
滑动窗口 + 定期摘要压缩，平衡上下文完整性和 Token 消耗
"""

from typing import Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from config.prompts import CONTEXT_SUMMARY_PROMPT, format_prompt
from config.settings import get_settings
from llm.qwen_client import get_qwen_client
from utils.logger import logger


class ConversationMemory:
    """
    对话记忆管理器

    策略：
    1. 维护滑动窗口（最近 N 条消息），保证近期上下文
    2. 每 M 轮对话触发一次 LLM 摘要，提取核心信息
    3. 将摘要作为系统消息注入上下文
    """

    def __init__(self):
        self.settings = get_settings()
        self.qwen_client = get_qwen_client()
        self.window_size = self.settings.memory_window_size
        self.summary_interval = self.settings.memory_summary_interval

    def should_summarize(self, turn_count: int) -> bool:
        """判断是否需要触发摘要"""
        return turn_count > 0 and turn_count % self.summary_interval == 0

    def summarize(self, messages: list[BaseMessage]) -> str:
        """
        对对话历史进行摘要

        Args:
            messages: 需要摘要的消息列表

        Returns:
            摘要文本
        """
        if not messages or len(messages) < 2:
            return ""

        try:
            # 构建对话文本
            conversation_parts = []
            for msg in messages:
                if isinstance(msg, HumanMessage):
                    conversation_parts.append(f"用户: {msg.content}")
                elif isinstance(msg, AIMessage):
                    content = msg.content if isinstance(msg.content, str) else str(msg.content)
                    conversation_parts.append(f"客服: {content}")
                elif isinstance(msg, SystemMessage):
                    continue  # 跳过系统消息

            conversation = "\n".join(conversation_parts)

            prompt = format_prompt(CONTEXT_SUMMARY_PROMPT, conversation=conversation)
            summary = self.qwen_client.chat(
                messages=[HumanMessage(content=prompt)],
            ).strip()

            logger.info(f"对话摘要生成: {summary[:100]}...")
            return summary

        except Exception as e:
            logger.error(f"对话摘要生成失败: {e}")
            return ""

    def get_context_messages(
        self,
        messages: list[BaseMessage],
        context_summary: str = "",
        user_input: str = "",
    ) -> list[BaseMessage]:
        """
        获取用于 LLM 的上下文消息

        策略：
        1. 如果有摘要，添加为系统消息
        2. 取最近 window_size 条消息
        3. 如果有新的用户输入，添加到末尾

        Args:
            messages: 完整对话历史
            context_summary: 之前的上下文摘要
            user_input: 当前用户输入（可选）

        Returns:
            处理后的上下文消息列表
        """
        result = []

        # 添加上下文摘要
        if context_summary:
            result.append(SystemMessage(
                content=f"以下是之前对话的关键信息摘要，请参考：\n{context_summary}"
            ))

        # 滑动窗口
        if messages:
            recent = messages[-self.window_size:] if len(messages) > self.window_size else messages
            result.extend(recent)

        # 添加当前用户输入
        if user_input:
            result.append(HumanMessage(content=user_input))

        return result

    def update_summary(
        self,
        messages: list[BaseMessage],
        old_summary: str,
        turn_count: int,
    ) -> str:
        """
        根据轮次判断是否更新摘要

        Args:
            messages: 完整对话历史
            old_summary: 之前的摘要
            turn_count: 当前轮次

        Returns:
            更新后的摘要（如果没有更新则返回旧摘要）
        """
        if not self.should_summarize(turn_count):
            return old_summary

        # 将旧摘要 + 最近对话一起摘要
        messages_to_summarize = messages[:-2] if len(messages) > 2 else messages
        new_summary = self.summarize(messages_to_summarize)

        if new_summary:
            # 合并旧摘要和新摘要
            if old_summary:
                return f"{old_summary}\n{new_summary}"
            return new_summary

        return old_summary


# 全局单例
_memory: Optional[ConversationMemory] = None


def get_memory() -> ConversationMemory:
    """获取 ConversationMemory 单例"""
    global _memory
    if _memory is None:
        _memory = ConversationMemory()
    return _memory

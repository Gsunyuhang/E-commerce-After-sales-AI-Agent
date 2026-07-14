"""
通义千问 LLM 客户端封装
提供统一的对话、流式对话和 Embedding 接口
"""

from typing import Any, Optional

from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from config.settings import get_settings
from utils.exception import LLMError
from utils.logger import logger


class QwenClient:
    """通义千问 LLM 客户端"""

    _instance: Optional["QwenClient"] = None
    _chat_model: Optional[ChatTongyi] = None
    _embedding_model: Optional[DashScopeEmbeddings] = None

    def __new__(cls) -> "QwenClient":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._chat_model is None:
            self._init_models()

    def _init_models(self) -> None:
        """初始化模型"""
        settings = get_settings()
        try:
            import os
            os.environ["DASHSCOPE_API_KEY"] = settings.dashscope_api_key

            self._chat_model = ChatTongyi(
                model=settings.llm_model_name,
                temperature=settings.llm_temperature,
                max_tokens=settings.llm_max_tokens,
                dashscope_api_key=settings.dashscope_api_key,
            )

            self._embedding_model = DashScopeEmbeddings(
                model=settings.embedding_model_name,
                dashscope_api_key=settings.dashscope_api_key,
            )

            logger.info(f"通义千问客户端初始化完成，模型: {settings.llm_model_name}")
        except Exception as e:
            logger.error(f"通义千问客户端初始化失败: {e}")
            raise LLMError(f"LLM 初始化失败: {e}")

    @property
    def chat_model(self) -> ChatTongyi:
        """获取对话模型"""
        return self._chat_model

    @property
    def embedding_model(self) -> DashScopeEmbeddings:
        """获取 Embedding 模型"""
        return self._embedding_model

    def chat(
        self,
        messages: list[BaseMessage],
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        同步对话

        Args:
            messages: 消息列表
            system_prompt: 系统提示词（可选）

        Returns:
            AI 回复文本
        """
        try:
            full_messages = []
            if system_prompt:
                full_messages.append(SystemMessage(content=system_prompt))
            full_messages.extend(messages)

            response = self._chat_model.invoke(full_messages)
            content = response.content if isinstance(response.content, str) else str(response.content)
            return content
        except Exception as e:
            logger.error(f"LLM 对话失败: {e}")
            raise LLMError(f"LLM 对话失败: {e}")

    def chat_with_tool(
        self,
        messages: list[BaseMessage],
        tools: Optional[list] = None,
        system_prompt: Optional[str] = None,
    ) -> AIMessage:
        """
        支持工具调用的对话

        Args:
            messages: 消息列表
            tools: 可用工具列表（LangChain Tool 格式）
            system_prompt: 系统提示词

        Returns:
            AI 消息（可能包含 tool_calls）
        """
        try:
            full_messages = []
            if system_prompt:
                full_messages.append(SystemMessage(content=system_prompt))
            full_messages.extend(messages)

            if tools:
                response = self._chat_model.bind_tools(tools).invoke(full_messages)
            else:
                response = self._chat_model.invoke(full_messages)

            return response
        except Exception as e:
            logger.error(f"LLM 工具对话失败: {e}")
            raise LLMError(f"LLM 工具对话失败: {e}")

    def chat_stream(
        self,
        messages: list[BaseMessage],
        system_prompt: Optional[str] = None,
    ):
        """
        流式对话

        Args:
            messages: 消息列表
            system_prompt: 系统提示词

        Yields:
            流式输出的文本片段
        """
        try:
            full_messages = []
            if system_prompt:
                full_messages.append(SystemMessage(content=system_prompt))
            full_messages.extend(messages)

            for chunk in self._chat_model.stream(full_messages):
                content = chunk.content if isinstance(chunk.content, str) else str(chunk.content)
                if content:
                    yield content
        except Exception as e:
            logger.error(f"LLM 流式对话失败: {e}")
            raise LLMError(f"LLM 流式对话失败: {e}")

    def embed_text(self, text: str) -> list[float]:
        """
        生成文本的向量表示

        Args:
            text: 输入文本

        Returns:
            向量表示
        """
        try:
            return self._embedding_model.embed_query(text)
        except Exception as e:
            logger.error(f"Embedding 生成失败: {e}")
            raise LLMError(f"Embedding 生成失败: {e}")

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        批量生成文本向量

        Args:
            texts: 文本列表

        Returns:
            向量列表
        """
        try:
            return self._embedding_model.embed_documents(texts)
        except Exception as e:
            logger.error(f"批量 Embedding 生成失败: {e}")
            raise LLMError(f"批量 Embedding 生成失败: {e}")


# 全局单例
_qwen_client: Optional[QwenClient] = None


def get_qwen_client() -> QwenClient:
    """获取 QwenClient 单例"""
    global _qwen_client
    if _qwen_client is None:
        _qwen_client = QwenClient()
    return _qwen_client

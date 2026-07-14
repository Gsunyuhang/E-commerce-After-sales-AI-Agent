"""
Chroma 向量库管理模块
初始化、文档入库、检索
"""

from typing import Optional

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma

from config.settings import get_settings
from llm.qwen_client import get_qwen_client
from utils.exception import RetrievalError
from utils.logger import logger


class VectorStoreManager:
    """Chroma 向量库管理器"""

    _instance: Optional["VectorStoreManager"] = None
    _vector_store: Optional[Chroma] = None

    def __new__(cls) -> "VectorStoreManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._vector_store is None:
            self._settings = get_settings()

    def init_vector_store(self, documents: Optional[list[Document]] = None) -> Chroma:
        """
        初始化或加载向量库

        Args:
            documents: 如果提供，则构建新的向量库；否则尝试加载已有的

        Returns:
            Chroma 向量库实例
        """
        try:
            embedding_model = get_qwen_client().embedding_model

            if documents:
                # 构建新的向量库
                logger.info(f"正在构建向量库，文档数: {len(documents)}")
                self._vector_store = Chroma.from_documents(
                    documents=documents,
                    embedding=embedding_model,
                    collection_name=self._settings.chroma_collection_name,
                    persist_directory=str(self._settings.chroma_path),
                )
                logger.info(f"向量库构建完成，持久化到: {self._settings.chroma_path}")
            else:
                # 加载已有向量库
                self._vector_store = Chroma(
                    collection_name=self._settings.chroma_collection_name,
                    embedding_function=embedding_model,
                    persist_directory=str(self._settings.chroma_path),
                )
                logger.info("已加载现有向量库")

            return self._vector_store
        except Exception as e:
            logger.error(f"向量库初始化失败: {e}")
            raise RetrievalError(f"向量库初始化失败: {e}")

    @property
    def vector_store(self) -> Chroma:
        """获取向量库实例"""
        if self._vector_store is None:
            self.init_vector_store()
        return self._vector_store

    def similarity_search(
        self,
        query: str,
        k: Optional[int] = None,
    ) -> list[tuple[Document, float]]:
        """
        相似度搜索

        Args:
            query: 查询文本
            k: 返回结果数

        Returns:
            (文档, 相似度分数) 列表，按相似度降序排列
        """
        k = k or self._settings.retrieval_top_k
        try:
            results = self.vector_store.similarity_search_with_relevance_scores(
                query=query,
                k=k,
            )
            return results
        except Exception as e:
            logger.error(f"向量库检索失败: {e}")
            raise RetrievalError(f"向量库检索失败: {e}")

    def add_documents(self, documents: list[Document]) -> None:
        """
        向向量库添加文档

        Args:
            documents: 待添加的文档列表
        """
        try:
            self.vector_store.add_documents(documents)
            logger.info(f"向向量库添加 {len(documents)} 个文档")
        except Exception as e:
            logger.error(f"向向量库添加文档失败: {e}")
            raise RetrievalError(f"向向量库添加文档失败: {e}")

    def get_collection_count(self) -> int:
        """获取向量库中的文档数量"""
        try:
            return self.vector_store._collection.count()
        except Exception:
            return 0


# 全局单例
_vector_store_manager: Optional[VectorStoreManager] = None


def get_vector_store_manager() -> VectorStoreManager:
    """获取 VectorStoreManager 单例"""
    global _vector_store_manager
    if _vector_store_manager is None:
        _vector_store_manager = VectorStoreManager()
    return _vector_store_manager

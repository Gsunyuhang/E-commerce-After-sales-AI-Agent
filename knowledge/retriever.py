"""
知识库检索器
含 Query Rewrite 查询重写和置信度评分
"""

import json
import re
from typing import Optional

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

from config.prompts import QUERY_REWRITE_PROMPT, format_prompt
from config.settings import get_settings
from knowledge.vector_store import get_vector_store_manager
from llm.qwen_client import get_qwen_client
from utils.exception import RetrievalError
from utils.logger import logger


class RetrievalResult:
    """检索结果"""

    def __init__(
        self,
        query: str,
        rewritten_query: str,
        documents: list[Document],
        scores: list[float],
        max_score: float,
        is_reliable: bool,
    ):
        self.original_query = query
        self.rewritten_query = rewritten_query
        self.documents = documents
        self.scores = scores
        self.max_score = max_score
        self.is_reliable = is_reliable

    def to_context(self) -> str:
        """将检索结果格式化为上下文文本"""
        if not self.documents:
            return ""

        context_parts = []
        for i, (doc, score) in enumerate(zip(self.documents, self.scores)):
            source = doc.metadata.get("filename", "未知文档")
            context_parts.append(
                f"【来源{i + 1}: {source} | 相关度: {score:.2f}】\n{doc.page_content}"
            )
        return "\n\n---\n\n".join(context_parts)

    def to_dict(self) -> dict:
        """转为字典格式"""
        return {
            "original_query": self.original_query,
            "rewritten_query": self.rewritten_query,
            "doc_count": len(self.documents),
            "max_score": self.max_score,
            "is_reliable": self.is_reliable,
            "sources": [
                {
                    "filename": doc.metadata.get("filename", ""),
                    "score": score,
                    "preview": doc.page_content[:100] + "...",
                }
                for doc, score in zip(self.documents, self.scores)
            ],
        }


class PolicyRetriever:
    """售后政策检索器"""

    def __init__(self):
        self.settings = get_settings()
        self.qwen_client = get_qwen_client()
        self.vector_store_manager = get_vector_store_manager()

    def rewrite_query(self, original_query: str) -> str:
        """
        Query Rewrite: 将口语化提问改写为标准检索词

        Args:
            original_query: 用户原始查询

        Returns:
            改写后的查询
        """
        try:
            prompt = format_prompt(QUERY_REWRITE_PROMPT, original_query=original_query)
            response = self.qwen_client.chat(
                messages=[HumanMessage(content=prompt)],
            )
            rewritten = response.strip()

            # 如果改写结果为空或过长（可能是异常），使用原始查询
            if not rewritten or len(rewritten) > 200:
                logger.warning(f"Query Rewrite 结果异常，使用原始查询: {original_query}")
                return original_query

            logger.info(f"Query Rewrite: '{original_query}' -> '{rewritten}'")
            return rewritten
        except Exception as e:
            logger.error(f"Query Rewrite 失败，使用原始查询: {e}")
            return original_query

    def retrieve(
        self,
        query: str,
        use_rewrite: bool = True,
        k: Optional[int] = None,
    ) -> RetrievalResult:
        """
        检索知识库

        Args:
            query: 用户查询
            use_rewrite: 是否使用 Query Rewrite
            k: 返回结果数

        Returns:
            检索结果对象
        """
        try:
            # Query Rewrite
            if use_rewrite:
                search_query = self.rewrite_query(query)
            else:
                search_query = query

            # 向量检索
            k = k or self.settings.retrieval_top_k
            results = self.vector_store_manager.similarity_search(search_query, k=k)

            if not results:
                logger.warning(f"检索结果为空，查询: {search_query}")
                return RetrievalResult(
                    query=query,
                    rewritten_query=search_query,
                    documents=[],
                    scores=[],
                    max_score=0.0,
                    is_reliable=False,
                )

            documents = [doc for doc, _ in results]
            scores = [score for _, score in results]
            max_score = max(scores) if scores else 0.0

            # 置信度判断
            is_reliable = max_score >= self.settings.retrieval_confidence_threshold

            logger.info(
                f"检索完成: query='{search_query}', "
                f"results={len(documents)}, max_score={max_score:.3f}, "
                f"reliable={is_reliable}"
            )

            return RetrievalResult(
                query=query,
                rewritten_query=search_query,
                documents=documents,
                scores=scores,
                max_score=max_score,
                is_reliable=is_reliable,
            )
        except Exception as e:
            logger.error(f"检索失败: {e}")
            raise RetrievalError(f"知识库检索失败: {e}")


# 全局单例
_retriever: Optional[PolicyRetriever] = None


def get_retriever() -> PolicyRetriever:
    """获取 PolicyRetriever 单例"""
    global _retriever
    if _retriever is None:
        _retriever = PolicyRetriever()
    return _retriever

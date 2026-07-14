"""
文档加载与预处理模块
加载 Markdown 知识库文档，按语义分块
"""

from pathlib import Path
from typing import Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import get_settings
from utils.logger import logger


class DocumentLoader:
    """知识库文档加载器"""

    def __init__(self):
        settings = get_settings()
        self.knowledge_base_path = settings.knowledge_base_path
        self.chunk_size = settings.chunk_size
        self.chunk_overlap = settings.chunk_overlap
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n## ", "\n### ", "\n\n", "\n", "。", "；", "，", " "],
            keep_separator=True,
        )

    def load_documents(self) -> list[Document]:
        """
        加载知识库目录下所有 Markdown 文档

        Returns:
            文档列表
        """
        documents = []
        md_files = list(self.knowledge_base_path.glob("*.md"))

        if not md_files:
            logger.warning(f"知识库目录 {self.knowledge_base_path} 下未找到 Markdown 文档")
            return documents

        for md_file in md_files:
            try:
                content = md_file.read_text(encoding="utf-8")
                doc = Document(
                    page_content=content,
                    metadata={
                        "source": str(md_file),
                        "filename": md_file.name,
                        "title": md_file.stem,
                    },
                )
                documents.append(doc)
                logger.info(f"加载文档: {md_file.name}")
            except Exception as e:
                logger.error(f"加载文档 {md_file.name} 失败: {e}")

        logger.info(f"共加载 {len(documents)} 个文档")
        return documents

    def split_documents(self, documents: list[Document]) -> list[Document]:
        """
        将文档分块

        Args:
            documents: 原始文档列表

        Returns:
            分块后的文档列表
        """
        chunks = self._splitter.split_documents(documents)

        # 为每个分块添加序号元数据
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = i
            chunk.metadata["chunk_index"] = i

        logger.info(f"文档分块完成，共 {len(chunks)} 个文本块")
        return chunks

    def load_and_split(self) -> list[Document]:
        """
        加载并分块文档（便捷方法）

        Returns:
            分块后的文档列表
        """
        documents = self.load_documents()
        if not documents:
            return []
        return self.split_documents(documents)

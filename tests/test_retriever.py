"""
检索器测试
测试文档加载、分块、配置
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from knowledge.document_loader import DocumentLoader
from config.settings import get_settings


class TestDocumentLoader:
    """文档加载器测试"""

    def test_load_documents(self):
        """测试加载知识库文档"""
        loader = DocumentLoader()
        documents = loader.load_documents()

        assert len(documents) == 4  # 4 个 Markdown 文件
        # 验证文件名
        filenames = [doc.metadata["filename"] for doc in documents]
        assert "return_policy.md" in filenames
        assert "shipping_policy.md" in filenames
        assert "warranty_policy.md" in filenames
        assert "insurance_policy.md" in filenames

    def test_split_documents(self):
        """测试文档分块"""
        loader = DocumentLoader()
        documents = loader.load_documents()
        chunks = loader.split_documents(documents)

        assert len(chunks) > 4  # 分块后应该更多
        # 验证元数据
        for chunk in chunks:
            assert "chunk_id" in chunk.metadata
            assert "filename" in chunk.metadata
            assert "source" in chunk.metadata

    def test_load_and_split(self):
        """测试便捷方法"""
        loader = DocumentLoader()
        chunks = loader.load_and_split()

        assert len(chunks) > 0
        # 验证内容不为空
        for chunk in chunks:
            assert len(chunk.page_content) > 0

    def test_chunk_size_config(self):
        """测试分块大小配置"""
        settings = get_settings()
        loader = DocumentLoader()

        assert loader.chunk_size == settings.chunk_size
        assert loader.chunk_overlap == settings.chunk_overlap


class TestSettings:
    """配置测试"""

    def test_settings_loaded(self):
        """测试配置加载"""
        settings = get_settings()
        assert settings.llm_model_name == "qwen-plus"
        assert settings.retrieval_top_k == 3
        assert settings.retrieval_confidence_threshold == 0.65

    def test_paths_exist(self):
        """测试路径配置"""
        settings = get_settings()
        assert settings.knowledge_base_path.exists()
        assert settings.log_path.exists()

"""
统一配置管理模块
所有配置项通过 .env 文件加载，杜绝硬编码
"""

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent


class Settings(BaseSettings):
    """全局配置，从 .env 文件自动加载"""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- 通义千问 API 配置 ---
    dashscope_api_key: str = ""

    # --- 模型配置 ---
    llm_model_name: str = "qwen-plus"
    llm_temperature: float = 0.3
    llm_max_tokens: int = 2048
    embedding_model_name: str = "text-embedding-v2"

    # --- 数据库配置 ---
    sqlite_db_path: str = "data/ecommerce_agent.db"

    # --- 向量库配置 ---
    chroma_persist_dir: str = "data/chroma_db"
    chroma_collection_name: str = "after_sales_policy"

    # --- 知识库配置 ---
    knowledge_base_dir: str = "data/knowledge_base"
    chunk_size: int = 500
    chunk_overlap: int = 50
    retrieval_top_k: int = 3

    # --- 阈值参数 ---
    retrieval_confidence_threshold: float = 0.65
    intent_confidence_threshold: float = 0.7
    human_handoff_threshold: float = 0.5

    # --- 多轮记忆配置 ---
    memory_window_size: int = 6
    memory_summary_interval: int = 3

    # --- 日志配置 ---
    log_level: str = "INFO"
    log_dir: str = "logs"

    # --- 服务配置 ---
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    frontend_port: int = 8501

    # --- 计算属性 ---

    @property
    def db_url(self) -> str:
        """SQLite 数据库连接 URL"""
        db_path = PROJECT_ROOT / self.sqlite_db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{db_path}"

    @property
    def chroma_path(self) -> Path:
        """Chroma 持久化目录的绝对路径"""
        path = PROJECT_ROOT / self.chroma_persist_dir
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def knowledge_base_path(self) -> Path:
        """知识库文档目录的绝对路径"""
        return PROJECT_ROOT / self.knowledge_base_dir

    @property
    def log_path(self) -> Path:
        """日志目录的绝对路径"""
        path = PROJECT_ROOT / self.log_dir
        path.mkdir(parents=True, exist_ok=True)
        return path


@lru_cache()
def get_settings() -> Settings:
    """获取全局配置单例"""
    return Settings()

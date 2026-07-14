"""
结构化日志模块
全流程日志埋点，记录每轮对话的用户输入、意图识别结果、检索命中片段、
工具调用参数与返回、最终回复内容。支持文件 + 控制台双输出，按日期轮转。
"""

import json
import logging
import logging.handlers
from datetime import datetime
from typing import Any, Optional

from config.settings import get_settings


class JsonFormatter(logging.Formatter):
    """JSON 格式化器，输出结构化日志"""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "line": record.lineno,
        }

        # 合并 extra 字段
        if hasattr(record, "extra_data"):
            log_data["extra"] = record.extra_data

        # 异常信息
        if record.exc_info and record.exc_info[1] is not None:
            log_data["exception"] = {
                "type": type(record.exc_info[1]).__name__,
                "message": str(record.exc_info[1]),
            }

        return json.dumps(log_data, ensure_ascii=False, default=str)


def setup_logger(name: str = "ecommerce_agent") -> logging.Logger:
    """
    初始化并获取日志器

    Args:
        name: 日志器名称

    Returns:
        配置好的 Logger 实例
    """
    settings = get_settings()
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))

    # 控制台 handler（人类可读格式）
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # 文件 handler（JSON 格式，按日期轮转，保留 30 天）
    log_file = settings.log_path / "agent.log"
    file_handler = logging.handlers.TimedRotatingFileHandler(
        log_file,
        when="midnight",
        interval=1,
        backupCount=30,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(JsonFormatter())
    logger.addHandler(file_handler)

    logger.propagate = False
    return logger


# 全局日志器实例
logger = setup_logger()


def log_chat_turn(
    session_id: str,
    user_input: str,
    intent: Optional[str] = None,
    retrieved_docs: Optional[list] = None,
    tool_calls: Optional[list] = None,
    response: Optional[str] = None,
    response_time: Optional[float] = None,
    extra: Optional[dict] = None,
) -> None:
    """
    记录一轮对话的完整日志

    Args:
        session_id: 会话 ID
        user_input: 用户输入
        intent: 意图识别结果
        retrieved_docs: 检索命中的文档片段
        tool_calls: 工具调用记录
        response: Agent 最终回复
        response_time: 响应耗时（秒）
        extra: 额外信息
    """
    extra_data = {
        "session_id": session_id,
        "user_input": user_input,
        "intent": intent,
        "retrieved_docs": retrieved_docs or [],
        "tool_calls": tool_calls or [],
        "response": response,
        "response_time": response_time,
    }
    if extra:
        extra_data.update(extra)

    logger.info("对话轮次记录", extra={"extra_data": extra_data})


def log_tool_call(
    tool_name: str,
    tool_input: dict,
    tool_output: Any,
    success: bool = True,
    error: Optional[str] = None,
) -> None:
    """
    记录工具调用日志

    Args:
        tool_name: 工具名称
        tool_input: 工具输入参数
        tool_output: 工具输出结果
        success: 是否成功
        error: 错误信息
    """
    extra_data = {
        "tool_name": tool_name,
        "tool_input": tool_input,
        "tool_output": str(tool_output)[:500] if tool_output else None,
        "success": success,
        "error": error,
    }
    level = logging.INFO if success else logging.ERROR
    logger.log(level, f"工具调用: {tool_name}", extra={"extra_data": extra_data})

"""
工具基类
定义统一接口、参数校验、日志记录
"""

from abc import ABC, abstractmethod
from typing import Any, Optional

from utils.exception import ToolError
from utils.logger import log_tool_call, logger


class BaseTool(ABC):
    """工具基类"""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @abstractmethod
    def get_parameters_schema(self) -> dict:
        """获取工具参数的 JSON Schema"""
        pass

    @abstractmethod
    def _execute(self, **kwargs) -> dict:
        """实际执行逻辑（子类实现）"""
        pass

    @abstractmethod
    def validate_params(self, **kwargs) -> bool:
        """参数校验（子类实现）"""
        pass

    def execute(self, **kwargs) -> dict:
        """
        执行工具（含校验、日志、异常处理）

        Returns:
            执行结果字典，包含 success 和 data/error 字段
        """
        try:
            # 参数校验
            if not self.validate_params(**kwargs):
                error_msg = f"参数校验失败: {kwargs}"
                logger.warning(f"工具 [{self.name}] {error_msg}")
                log_tool_call(self.name, kwargs, None, success=False, error=error_msg)
                return {
                    "success": False,
                    "error": error_msg,
                    "tool_name": self.name,
                }

            # 执行
            logger.info(f"工具 [{self.name}] 开始执行，参数: {kwargs}")
            result = self._execute(**kwargs)

            log_tool_call(self.name, kwargs, result, success=True)
            logger.info(f"工具 [{self.name}] 执行成功")

            return {
                "success": True,
                "data": result,
                "tool_name": self.name,
            }

        except ToolError as e:
            logger.error(f"工具 [{self.name}] 执行失败: {e}")
            log_tool_call(self.name, kwargs, None, success=False, error=str(e))
            return {
                "success": False,
                "error": str(e),
                "fallback": e.fallback_response,
                "tool_name": self.name,
            }
        except Exception as e:
            logger.error(f"工具 [{self.name}] 未预期异常: {e}")
            log_tool_call(self.name, kwargs, None, success=False, error=str(e))
            return {
                "success": False,
                "error": f"工具执行异常: {e}",
                "fallback": f"查询服务暂时不可用，请稍后重试或联系人工客服。",
                "tool_name": self.name,
            }

    def to_langchain_tool(self):
        """转为 LangChain Tool 格式"""
        from langchain_core.tools import StructuredTool
        from pydantic import BaseModel, create_model

        # 动态创建 Pydantic 模型
        schema = self.get_parameters_schema()
        properties = schema.get("properties", {})
        required = schema.get("required", [])

        field_definitions = {}
        for field_name, field_info in properties.items():
            field_type = str
            default = ... if field_name in required else None
            field_definitions[field_name] = (field_type, default)

        model = create_model(f"{self.name}_Input", **field_definitions)

        def run(**kwargs):
            result = self.execute(**kwargs)
            return result

        return StructuredTool.from_function(
            func=run,
            name=self.name,
            description=self.description,
            args_schema=model,
        )

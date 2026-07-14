"""
物流查询工具
根据订单号查询物流状态
"""

import json
from pathlib import Path
from typing import Optional

from config.settings import PROJECT_ROOT
from database.connection import get_db_session
from database.models import Order
from tools.base import BaseTool
from utils.exception import DatabaseError, ToolError
from utils.logger import logger


class LogisticsQueryTool(BaseTool):
    """物流查询工具"""

    def __init__(self):
        super().__init__(
            name="logistics_query",
            description="查询物流信息。需要提供订单号，系统会根据订单关联的物流单号查询物流状态。",
        )
        self._logistics_data: Optional[list[dict]] = None

    def get_parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "订单编号",
                }
            },
            "required": ["order_id"],
        }

    def validate_params(self, **kwargs) -> bool:
        """参数校验"""
        order_id = kwargs.get("order_id")
        if not order_id or not isinstance(order_id, str):
            return False
        return True

    def _load_logistics_data(self) -> list[dict]:
        """加载物流数据（缓存）"""
        if self._logistics_data is None:
            logistics_path = PROJECT_ROOT / "data" / "seed" / "logistics.json"
            try:
                with open(logistics_path, "r", encoding="utf-8") as f:
                    self._logistics_data = json.load(f)
            except Exception as e:
                logger.error(f"加载物流数据失败: {e}")
                self._logistics_data = []

        return self._logistics_data

    def _execute(self, order_id: str, **kwargs) -> dict:
        """执行物流查询"""
        order_id = order_id.strip()

        try:
            # 先查订单获取物流单号
            with get_db_session() as session:
                order = session.query(Order).filter_by(order_id=order_id).first()

                if not order:
                    return {
                        "found": False,
                        "message": f"未找到订单号为 {order_id} 的订单。",
                    }

                if not order.tracking_number:
                    return {
                        "found": True,
                        "has_logistics": False,
                        "message": f"订单 {order_id} 尚未发货，暂无物流信息。",
                        "order_status": order.order_status,
                    }

                tracking_number = order.tracking_number

            # 查询物流数据
            logistics_data = self._load_logistics_data()
            logistics_info = None
            for item in logistics_data:
                if item.get("tracking_number") == tracking_number:
                    logistics_info = item
                    break

            if not logistics_info:
                return {
                    "found": True,
                    "has_logistics": True,
                    "tracking_number": tracking_number,
                    "message": f"物流单号 {tracking_number}，暂未查询到详细物流轨迹。",
                }

            # 格式化物流信息
            status_map = {
                "delivered": "已签收",
                "in_transit": "运输中",
                "pending": "待发货",
            }

            return {
                "found": True,
                "has_logistics": True,
                "tracking_number": tracking_number,
                "logistics_company": logistics_info.get("logistics_company", ""),
                "status": logistics_info.get("status", ""),
                "status_text": status_map.get(logistics_info.get("status", ""), "未知"),
                "estimated_arrival": logistics_info.get("estimated_arrival", ""),
                "tracking_info": logistics_info.get("tracking_info", []),
                "order_id": order_id,
            }

        except Exception as e:
            logger.error(f"物流查询异常: {e}")
            raise DatabaseError(f"物流查询失败: {e}")

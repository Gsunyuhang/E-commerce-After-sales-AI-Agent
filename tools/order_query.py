"""
订单查询工具
根据订单号查询 SQLite 中的订单信息
前置条件：必须检测到有效订单号才执行
"""

import re
from typing import Optional

from database.connection import get_db_session
from database.models import Order
from tools.base import BaseTool
from utils.exception import DatabaseError, ToolError
from utils.logger import logger


class OrderQueryTool(BaseTool):
    """订单查询工具"""

    def __init__(self):
        super().__init__(
            name="order_query",
            description="查询订单信息。仅在用户明确提供了订单号（格式为DD开头的编号）时使用。",
        )

    def get_parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "订单编号，格式为DD开头的数字串",
                }
            },
            "required": ["order_id"],
        }

    def validate_params(self, **kwargs) -> bool:
        """参数校验：订单号格式检查"""
        order_id = kwargs.get("order_id")
        if not order_id or not isinstance(order_id, str):
            return False
        # 订单号格式：DD 开头 + 数字
        if not re.match(r"^DD\d+$", order_id.strip()):
            logger.warning(f"订单号格式不正确: {order_id}")
            return False
        return True

    def _execute(self, order_id: str, **kwargs) -> dict:
        """执行订单查询"""
        order_id = order_id.strip()

        try:
            with get_db_session() as session:
                order = session.query(Order).filter_by(order_id=order_id).first()

                if not order:
                    return {
                        "found": False,
                        "message": f"未找到订单号为 {order_id} 的订单，请确认订单号是否正确。",
                    }

                order_data = order.to_dict()
                logger.info(f"查询到订单: {order_id}")

                # 判断订单状态是否支持售后
                post_sale_status = {
                    "paid": "已付款（未发货）",
                    "shipped": "已发货（运输中）",
                    "delivered": "已签收",
                    "completed": "已完成",
                    "cancelled": "已取消",
                }
                status_text = post_sale_status.get(
                    order.order_status, order.order_status
                )

                # 售后可用性判断
                can_return = order.order_status in ["delivered", "completed"]
                can_cancel = order.order_status == "paid"

                return {
                    "found": True,
                    "order": order_data,
                    "status_text": status_text,
                    "can_return": can_return,
                    "can_cancel": can_cancel,
                    "has_insurance": order.has_insurance,
                }

        except Exception as e:
            logger.error(f"订单查询数据库异常: {e}")
            raise DatabaseError(f"订单查询失败: {e}")

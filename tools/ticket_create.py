"""
工单生成工具
校验退换货资格后，向 SQLite 写入工单记录
"""

import re
import uuid
from datetime import datetime, timedelta
from typing import Optional

from database.connection import get_db_session
from database.models import Order, Ticket
from tools.base import BaseTool
from utils.exception import DatabaseError, ToolError
from utils.logger import logger


# 不可退货商品类目
NON_RETURNABLE_CATEGORIES = {"beauty", "food"}

# 不可换货商品类目
NON_EXCHANGEABLE_CATEGORIES = {"food"}


class TicketCreateTool(BaseTool):
    """工单生成工具"""

    def __init__(self):
        super().__init__(
            name="ticket_create",
            description="创建售后工单。在用户明确表达退换货/退款意愿且通过资格校验后使用。",
        )

    def get_parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "订单编号",
                },
                "ticket_type": {
                    "type": "string",
                    "enum": ["return", "exchange", "refund"],
                    "description": "工单类型：return=退货, exchange=换货, refund=退款",
                },
                "reason": {
                    "type": "string",
                    "description": "退换货原因",
                },
            },
            "required": ["order_id", "ticket_type", "reason"],
        }

    def validate_params(self, **kwargs) -> bool:
        """参数校验"""
        order_id = kwargs.get("order_id")
        ticket_type = kwargs.get("ticket_type")
        reason = kwargs.get("reason")

        if not order_id or not isinstance(order_id, str):
            return False
        if ticket_type not in ["return", "exchange", "refund"]:
            return False
        if not reason or not isinstance(reason, str) or len(reason.strip()) < 2:
            return False
        return True

    def _check_eligibility(self, order: Order, ticket_type: str) -> dict:
        """
        校验退换货资格

        Returns:
            {"eligible": bool, "reason": str}
        """
        # 1. 检查订单状态
        if order.order_status == "cancelled":
            return {"eligible": False, "reason": "该订单已取消，无法发起退换货。"}

        if order.order_status == "paid":
            if ticket_type == "refund":
                return {"eligible": True, "reason": "订单未发货，可直接申请退款。"}
            return {"eligible": False, "reason": "订单尚未发货，建议直接申请退款。"}

        if order.order_status not in ["delivered", "completed"]:
            return {"eligible": False, "reason": f"订单当前状态为{order.order_status}，暂不支持退换货。"}

        # 2. 检查退换货时效（签收7天内）
        # 使用 order_time 作为近似（实际应用中应使用签收时间）
        days_since_order = (datetime.now() - order.order_time).days
        if days_since_order > 15:
            return {
                "eligible": False,
                "reason": f"订单已超过15天（当前{days_since_order}天），超出退换货时效，建议联系品牌售后进行质保维修。",
            }

        # 3. 检查商品类目限制
        if ticket_type in ["return", "refund"]:
            if order.product_category in NON_RETURNABLE_CATEGORIES:
                return {
                    "eligible": False,
                    "reason": f"商品类目（{order.product_category}）不支持无理由退货，如有质量问题请联系人工客服。",
                }

        if ticket_type == "exchange":
            if order.product_category in NON_EXCHANGEABLE_CATEGORIES:
                return {
                    "eligible": False,
                    "reason": f"商品类目（{order.product_category}）不支持换货。",
                }

        # 4. 超过7天但15天内，只能因质量问题退换
        if days_since_order > 7:
            return {
                "eligible": True,
                "reason": f"订单已签收{days_since_order}天，超过7天无理由退换货期，但仍在15天内，可协商处理。",
                "warning": "超过7天退换货时效，需人工审核确认。",
            }

        return {"eligible": True, "reason": "符合退换货条件。"}

    def _execute(self, order_id: str, ticket_type: str, reason: str, **kwargs) -> dict:
        """执行工单创建"""
        order_id = order_id.strip()
        ticket_type = ticket_type.strip()
        reason = reason.strip()

        try:
            with get_db_session() as session:
                # 查询订单
                order = session.query(Order).filter_by(order_id=order_id).first()
                if not order:
                    return {
                        "success": False,
                        "message": f"未找到订单号为 {order_id} 的订单。",
                    }

                # 检查是否已有进行中的工单
                existing_ticket = (
                    session.query(Ticket)
                    .filter_by(order_id=order_id, status="pending")
                    .first()
                )
                if existing_ticket:
                    return {
                        "success": False,
                        "message": f"订单 {order_id} 已有正在处理中的工单（工单号：{existing_ticket.ticket_id}），请勿重复提交。",
                        "existing_ticket_id": existing_ticket.ticket_id,
                    }

                # 校验资格
                eligibility = self._check_eligibility(order, ticket_type)
                if not eligibility["eligible"]:
                    return {
                        "success": False,
                        "eligible": False,
                        "message": eligibility["reason"],
                    }

                # 计算退款金额（退货/退款时）
                refund_amount = None
                if ticket_type in ["return", "refund"]:
                    refund_amount = order.product_price
                    # 如果有运费且非质量问题，扣除运费
                    if order.shipping_fee > 0 and "质量" not in reason:
                        refund_amount = order.product_price  # 商品金额全额退
                        # 运费根据政策处理

                # 创建工单
                ticket_id = f"TK{datetime.now().strftime('%Y%m%d%H%M%S')}{uuid.uuid4().hex[:4].upper()}"

                ticket = Ticket(
                    ticket_id=ticket_id,
                    order_id=order_id,
                    user_id=order.user_id,
                    ticket_type=ticket_type,
                    reason=reason,
                    status="pending",
                    refund_amount=refund_amount,
                    created_at=datetime.now(),
                )
                session.add(ticket)

                logger.info(f"创建工单: {ticket_id}, 订单: {order_id}, 类型: {ticket_type}")

                ticket_type_text = {
                    "return": "退货",
                    "exchange": "换货",
                    "refund": "退款",
                }.get(ticket_type, ticket_type)

                return {
                    "success": True,
                    "eligible": True,
                    "ticket_id": ticket_id,
                    "ticket_type": ticket_type,
                    "ticket_type_text": ticket_type_text,
                    "order_id": order_id,
                    "product_name": order.product_name,
                    "refund_amount": refund_amount,
                    "message": eligibility.get("reason", ""),
                    "warning": eligibility.get("warning"),
                    "has_insurance": order.has_insurance,
                }

        except Exception as e:
            logger.error(f"工单创建异常: {e}")
            raise DatabaseError(f"工单创建失败: {e}")

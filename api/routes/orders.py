"""
订单查询接口
"""

from fastapi import APIRouter, HTTPException

from api.schemas import OrderResponse
from database.connection import get_db_session
from database.models import Order
from utils.logger import logger

router = APIRouter(prefix="/api/orders", tags=["订单"])


@router.get("/{order_id}", response_model=OrderResponse)
async def get_order(order_id: str) -> OrderResponse:
    """查询订单信息"""
    logger.info(f"查询订单: {order_id}")

    try:
        with get_db_session() as session:
            order = session.query(Order).filter_by(order_id=order_id).first()

            if not order:
                return OrderResponse(
                    found=False,
                    message=f"未找到订单号为 {order_id} 的订单。",
                )

            order_data = order.to_dict()
            status_map = {
                "paid": "已付款（未发货）",
                "shipped": "已发货（运输中）",
                "delivered": "已签收",
                "completed": "已完成",
                "cancelled": "已取消",
            }

            return OrderResponse(
                found=True,
                order=order_data,
                status_text=status_map.get(order.order_status, order.order_status),
                can_return=order.order_status in ["delivered", "completed"],
                can_cancel=order.order_status == "paid",
                has_insurance=order.has_insurance,
            )
    except Exception as e:
        logger.error(f"订单查询失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

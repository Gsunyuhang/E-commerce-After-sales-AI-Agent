"""
退款计算工具
根据商品价格、运费规则、运费险状态计算可退款金额
"""

from typing import Optional

from database.connection import get_db_session
from database.models import Order
from tools.base import BaseTool
from utils.exception import DatabaseError, ToolError
from utils.logger import logger


class RefundCalcTool(BaseTool):
    """退款计算工具"""

    def __init__(self):
        super().__init__(
            name="refund_calc",
            description="计算退款金额。根据商品价格、运费和运费险状态计算可退款金额。",
        )

    def get_parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "订单编号",
                },
                "return_reason": {
                    "type": "string",
                    "enum": ["no_reason", "quality_issue"],
                    "description": "退货原因：no_reason=无理由退货, quality_issue=质量问题",
                },
            },
            "required": ["order_id", "return_reason"],
        }

    def validate_params(self, **kwargs) -> bool:
        """参数校验"""
        order_id = kwargs.get("order_id")
        return_reason = kwargs.get("return_reason")

        if not order_id or not isinstance(order_id, str):
            return False
        if return_reason not in ["no_reason", "quality_issue"]:
            return False
        return True

    def _execute(self, order_id: str, return_reason: str, **kwargs) -> dict:
        """执行退款计算"""
        order_id = order_id.strip()

        try:
            with get_db_session() as session:
                order = session.query(Order).filter_by(order_id=order_id).first()

                if not order:
                    return {
                        "found": False,
                        "message": f"未找到订单号为 {order_id} 的订单。",
                    }

                product_price = order.product_price
                shipping_fee = order.shipping_fee
                has_insurance = order.has_insurance

                # 退款计算逻辑
                # 商品金额始终可退
                refund_product = product_price

                # 运费处理
                if return_reason == "quality_issue":
                    # 质量问题：运费全退（商家承担来回运费）
                    refund_shipping = shipping_fee
                    insurance_payout = 0  # 质量问题不走运费险
                    reason_text = "质量问题退货"
                    shipping_note = "质量问题退货，来回运费由商家承担"
                else:
                    # 无理由退货：运费不退（消费者承担寄回运费）
                    refund_shipping = 0
                    # 运费险理赔
                    if has_insurance:
                        # 运费险理赔金额（固定估算，实际由保险公司核定）
                        insurance_payout = min(shipping_fee, 25)  # 上限25元
                        reason_text = "7天无理由退货（含运费险）"
                        shipping_note = f"无理由退货，运费险可理赔约{insurance_payout}元"
                    else:
                        insurance_payout = 0
                        reason_text = "7天无理由退货（无运费险）"
                        shipping_note = "无理由退货，运费由消费者承担"

                total_refund = refund_product + refund_shipping
                total_with_insurance = total_refund + insurance_payout

                logger.info(
                    f"退款计算: order={order_id}, reason={return_reason}, "
                    f"product={refund_product}, shipping={refund_shipping}, "
                    f"insurance={insurance_payout}, total={total_with_insurance}"
                )

                return {
                    "found": True,
                    "order_id": order_id,
                    "product_name": order.product_name,
                    "product_price": product_price,
                    "shipping_fee": shipping_fee,
                    "has_insurance": has_insurance,
                    "return_reason": return_reason,
                    "reason_text": reason_text,
                    "refund_breakdown": {
                        "商品金额": refund_product,
                        "运费退款": refund_shipping,
                        "运费险理赔": insurance_payout,
                    },
                    "total_refund": total_refund,
                    "total_with_insurance": total_with_insurance,
                    "shipping_note": shipping_note,
                    "note": "实际退款金额以最终审核结果为准，运费险理赔金额由保险公司核定。",
                }

        except Exception as e:
            logger.error(f"退款计算异常: {e}")
            raise DatabaseError(f"退款计算失败: {e}")

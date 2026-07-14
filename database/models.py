"""
SQLAlchemy 数据模型
定义订单、工单、对话日志三张表
"""

from datetime import datetime
from typing import Optional

from sqlalchemy import Column, DateTime, Float, Integer, String, Text, Boolean
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """SQLAlchemy 声明式基类"""
    pass


class Order(Base):
    """订单表"""

    __tablename__ = "orders"

    order_id = Column(String(64), primary_key=True, comment="订单编号")
    user_id = Column(String(64), nullable=False, comment="用户ID")
    product_name = Column(String(256), nullable=False, comment="商品名称")
    product_category = Column(String(64), nullable=False, comment="商品类目")
    product_price = Column(Float, nullable=False, comment="商品价格")
    order_status = Column(String(32), nullable=False, comment="订单状态: paid/shipped/delivered/completed/cancelled")
    order_time = Column(DateTime, nullable=False, comment="下单时间")
    shipping_fee = Column(Float, nullable=False, default=0.0, comment="运费")
    has_insurance = Column(Boolean, nullable=False, default=False, comment="是否购买运费险")
    tracking_number = Column(String(64), nullable=True, comment="物流单号")

    def to_dict(self) -> dict:
        return {
            "order_id": self.order_id,
            "user_id": self.user_id,
            "product_name": self.product_name,
            "product_category": self.product_category,
            "product_price": self.product_price,
            "order_status": self.order_status,
            "order_time": self.order_time.isoformat() if self.order_time else None,
            "shipping_fee": self.shipping_fee,
            "has_insurance": self.has_insurance,
            "tracking_number": self.tracking_number,
        }


class Ticket(Base):
    """售后工单表"""

    __tablename__ = "tickets"

    ticket_id = Column(String(64), primary_key=True, comment="工单编号")
    order_id = Column(String(64), nullable=False, comment="关联订单编号")
    user_id = Column(String(64), nullable=False, comment="用户ID")
    ticket_type = Column(String(32), nullable=False, comment="工单类型: return/exchange/refund")
    reason = Column(Text, nullable=False, comment="退换货原因")
    status = Column(String(32), nullable=False, default="pending", comment="工单状态: pending/processing/completed/rejected")
    refund_amount = Column(Float, nullable=True, comment="退款金额")
    created_at = Column(DateTime, nullable=False, default=datetime.now, comment="创建时间")

    def to_dict(self) -> dict:
        return {
            "ticket_id": self.ticket_id,
            "order_id": self.order_id,
            "user_id": self.user_id,
            "ticket_type": self.ticket_type,
            "reason": self.reason,
            "status": self.status,
            "refund_amount": self.refund_amount,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class ChatLog(Base):
    """对话日志表"""

    __tablename__ = "chat_logs"

    log_id = Column(Integer, primary_key=True, autoincrement=True, comment="日志ID")
    session_id = Column(String(64), nullable=False, index=True, comment="会话ID")
    user_input = Column(Text, nullable=False, comment="用户输入")
    intent = Column(String(64), nullable=True, comment="意图识别结果")
    retrieved_docs = Column(Text, nullable=True, comment="检索命中文档(JSON)")
    tool_calls = Column(Text, nullable=True, comment="工具调用记录(JSON)")
    response = Column(Text, nullable=True, comment="Agent 回复")
    response_time = Column(Float, nullable=True, comment="响应耗时(秒)")
    need_human = Column(Boolean, nullable=False, default=False, comment="是否转人工")
    created_at = Column(DateTime, nullable=False, default=datetime.now, comment="创建时间")

    def to_dict(self) -> dict:
        import json
        return {
            "log_id": self.log_id,
            "session_id": self.session_id,
            "user_input": self.user_input,
            "intent": self.intent,
            "retrieved_docs": json.loads(self.retrieved_docs) if self.retrieved_docs else [],
            "tool_calls": json.loads(self.tool_calls) if self.tool_calls else [],
            "response": self.response,
            "response_time": self.response_time,
            "need_human": self.need_human,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

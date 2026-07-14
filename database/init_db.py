"""
数据库初始化脚本
建表 + 导入种子数据
"""

import json
from datetime import datetime
from pathlib import Path

from config.settings import PROJECT_ROOT, get_settings
from database.connection import create_tables, get_db_session, drop_tables
from database.models import Order, Ticket, ChatLog
from utils.logger import logger


def load_seed_data() -> tuple[list[dict], list[dict]]:
    """加载种子数据"""
    seed_dir = PROJECT_ROOT / "data" / "seed"
    orders_path = seed_dir / "orders.json"
    logistics_path = seed_dir / "logistics.json"

    with open(orders_path, "r", encoding="utf-8") as f:
        orders = json.load(f)

    with open(logistics_path, "r", encoding="utf-8") as f:
        logistics = json.load(f)

    return orders, logistics


def seed_orders(orders_data: list[dict]) -> int:
    """导入订单种子数据"""
    count = 0
    with get_db_session() as session:
        for order_data in orders_data:
            # 检查是否已存在
            existing = session.query(Order).filter_by(order_id=order_data["order_id"]).first()
            if existing:
                continue

            order = Order(
                order_id=order_data["order_id"],
                user_id=order_data["user_id"],
                product_name=order_data["product_name"],
                product_category=order_data["product_category"],
                product_price=order_data["product_price"],
                order_status=order_data["order_status"],
                order_time=datetime.fromisoformat(order_data["order_time"]),
                shipping_fee=order_data["shipping_fee"],
                has_insurance=order_data["has_insurance"],
                tracking_number=order_data.get("tracking_number"),
            )
            session.add(order)
            count += 1

    logger.info(f"导入 {count} 条订单数据")
    return count


def seed_tickets() -> int:
    """导入初始工单数据（几条历史工单用于演示）"""
    sample_tickets = [
        {
            "ticket_id": "TK20250705001",
            "order_id": "DD20250704004",
            "user_id": "U10001",
            "ticket_type": "exchange",
            "reason": "鞋子尺码偏大，需要换小一码",
            "status": "completed",
            "refund_amount": None,
            "created_at": datetime(2025, 6, 28, 15, 0, 0),
        },
        {
            "ticket_id": "TK20250706002",
            "order_id": "DD20250705005",
            "user_id": "U10004",
            "ticket_type": "return",
            "reason": "商品不喜欢，7天无理由退货",
            "status": "completed",
            "refund_amount": 30.00,
            "created_at": datetime(2025, 6, 19, 10, 0, 0),
        },
    ]

    count = 0
    with get_db_session() as session:
        for ticket_data in sample_tickets:
            existing = session.query(Ticket).filter_by(ticket_id=ticket_data["ticket_id"]).first()
            if existing:
                continue

            ticket = Ticket(**ticket_data)
            session.add(ticket)
            count += 1

    logger.info(f"导入 {count} 条历史工单数据")
    return count


def init_database(reset: bool = False) -> None:
    """
    初始化数据库

    Args:
        reset: 是否重置（先删表再建表）
    """
    logger.info("开始初始化数据库...")

    if reset:
        logger.warning("正在删除所有表...")
        drop_tables()

    # 创建表
    create_tables()
    logger.info("数据表创建完成")

    # 导入种子数据
    orders_data, logistics_data = load_seed_data()
    seed_orders(orders_data)
    seed_tickets()

    logger.info("数据库初始化完成")


if __name__ == "__main__":
    init_database(reset=True)

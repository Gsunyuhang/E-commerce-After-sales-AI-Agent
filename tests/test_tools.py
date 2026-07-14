"""
工具测试
测试各工具的正常调用、参数校验、异常处理
"""

import os
import sys

import pytest

# 确保项目根目录在路径中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.connection import create_tables, drop_tables, get_db_session
from database.init_db import init_database
from tools.order_query import OrderQueryTool
from tools.logistics_query import LogisticsQueryTool
from tools.ticket_create import TicketCreateTool
from tools.refund_calc import RefundCalcTool


@pytest.fixture(scope="module", autouse=True)
def setup_database():
    """测试前初始化数据库"""
    init_database(reset=True)
    yield
    # 测试后清理（可选）


class TestOrderQueryTool:
    """订单查询工具测试"""

    def test_valid_order_query(self):
        """测试查询有效订单"""
        tool = OrderQueryTool()
        result = tool.execute(order_id="DD20250701001")

        assert result["success"] is True
        assert result["data"]["found"] is True
        assert result["data"]["order"]["product_name"] == "纯棉短袖T恤 男款 白色 L码"

    def test_invalid_order_id_format(self):
        """测试无效订单号格式"""
        tool = OrderQueryTool()
        result = tool.execute(order_id="invalid_id")

        assert result["success"] is False
        assert "参数校验失败" in result["error"]

    def test_nonexistent_order(self):
        """测试查询不存在的订单"""
        tool = OrderQueryTool()
        result = tool.execute(order_id="DD999999999")

        assert result["success"] is True
        assert result["data"]["found"] is False

    def test_empty_order_id(self):
        """测试空订单号"""
        tool = OrderQueryTool()
        result = tool.execute(order_id="")

        assert result["success"] is False

    def test_delivered_order_can_return(self):
        """测试已签收订单可退货"""
        tool = OrderQueryTool()
        result = tool.execute(order_id="DD20250701001")

        assert result["success"] is True
        assert result["data"]["can_return"] is True

    def test_cancelled_order_cannot_return(self):
        """测试已取消订单不可退货"""
        tool = OrderQueryTool()
        result = tool.execute(order_id="DD20250715015")

        assert result["success"] is True
        assert result["data"]["can_return"] is False


class TestLogisticsQueryTool:
    """物流查询工具测试"""

    def test_valid_logistics_query(self):
        """测试查询有效物流"""
        tool = LogisticsQueryTool()
        result = tool.execute(order_id="DD20250701001")

        assert result["success"] is True
        assert result["data"]["found"] is True
        assert result["data"]["has_logistics"] is True
        assert len(result["data"]["tracking_info"]) > 0

    def test_not_shipped_order(self):
        """测试未发货订单"""
        tool = LogisticsQueryTool()
        result = tool.execute(order_id="DD20250707007")

        assert result["success"] is True
        assert result["data"]["has_logistics"] is False

    def test_nonexistent_order_logistics(self):
        """测试查询不存在订单的物流"""
        tool = LogisticsQueryTool()
        result = tool.execute(order_id="DD999999999")

        assert result["success"] is True
        assert result["data"]["found"] is False


class TestTicketCreateTool:
    """工单创建工具测试"""

    def test_create_return_ticket(self):
        """测试创建退货工单"""
        tool = TicketCreateTool()
        result = tool.execute(
            order_id="DD20250701001",
            ticket_type="return",
            reason="商品不想要了",
        )

        assert result["success"] is True
        assert result["data"]["eligible"] is True
        assert result["data"]["ticket_id"].startswith("TK")

    def test_create_exchange_ticket(self):
        """测试创建换货工单"""
        tool = TicketCreateTool()
        result = tool.execute(
            order_id="DD20250704004",
            ticket_type="exchange",
            reason="尺码偏大需要换小一码",
        )

        assert result["success"] is True
        assert result["data"]["eligible"] is True

    def test_non_returnable_category(self):
        """测试不可退货商品类目"""
        tool = TicketCreateTool()
        result = tool.execute(
            order_id="DD20250706006",
            ticket_type="return",
            reason="不想要了",
        )

        assert result["success"] is False
        assert result["data"]["eligible"] is False

    def test_cancelled_order_cannot_create_ticket(self):
        """测试已取消订单不能创建工单"""
        tool = TicketCreateTool()
        result = tool.execute(
            order_id="DD20250715015",
            ticket_type="return",
            reason="退货",
        )

        assert result["success"] is False

    def test_invalid_ticket_type(self):
        """测试无效工单类型"""
        tool = TicketCreateTool()
        result = tool.execute(
            order_id="DD20250701001",
            ticket_type="invalid",
            reason="测试",
        )

        assert result["success"] is False

    def test_empty_reason(self):
        """测试空原因"""
        tool = TicketCreateTool()
        result = tool.execute(
            order_id="DD20250701001",
            ticket_type="return",
            reason="",
        )

        assert result["success"] is False


class TestRefundCalcTool:
    """退款计算工具测试"""

    def test_no_reason_refund_with_insurance(self):
        """测试无理由退货退款计算（含运费险）"""
        tool = RefundCalcTool()
        result = tool.execute(
            order_id="DD20250701001",
            return_reason="no_reason",
        )

        assert result["success"] is True
        assert result["data"]["product_price"] == 89.00
        assert result["data"]["has_insurance"] is True
        assert result["data"]["refund_breakdown"]["运费险理赔"] > 0

    def test_quality_issue_refund(self):
        """测试质量问题退款计算"""
        tool = RefundCalcTool()
        result = tool.execute(
            order_id="DD20250701001",
            return_reason="quality_issue",
        )

        assert result["success"] is True
        assert result["data"]["refund_breakdown"]["运费退款"] > 0

    def test_no_insurance_refund(self):
        """测试无运费险退款"""
        tool = RefundCalcTool()
        result = tool.execute(
            order_id="DD20250703003",
            return_reason="no_reason",
        )

        assert result["success"] is True
        assert result["data"]["has_insurance"] is False
        assert result["data"]["refund_breakdown"]["运费险理赔"] == 0

    def test_invalid_return_reason(self):
        """测试无效退货原因"""
        tool = RefundCalcTool()
        result = tool.execute(
            order_id="DD20250701001",
            return_reason="invalid",
        )

        assert result["success"] is False

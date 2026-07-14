"""
评估指标计算模块
计算 5 项核心评估指标
"""

from typing import Optional

from pydantic import BaseModel


class EvalMetrics(BaseModel):
    """评估指标汇总"""
    total_cases: int = 0
    resolution_rate: float = 0.0
    """问题解决率：无需人工介入、独立完成处理的问题占比"""

    refusal_accuracy: float = 0.0
    """拒答准确率：对超范围、无答案问题正确拒答/转人工的比例"""

    avg_response_time: float = 0.0
    """平均响应时长（秒）"""

    tool_call_success_rate: float = 0.0
    """工具调用成功率"""

    human_handoff_rate: float = 0.0
    """人工转接率"""


def calculate_metrics(results: list[dict]) -> EvalMetrics:
    """
    计算评估指标

    Args:
        results: 评估结果列表，每项包含:
            - category: 测试用例类别
            - expected_outcome: 期望结果 (resolved/refused/handoff)
            - actual_intent: 实际意图
            - need_human: 是否转人工
            - response_time: 响应时间
            - tool_calls: 工具调用列表
            - passed: 是否通过

    Returns:
        EvalMetrics 指标汇总
    """
    total = len(results)
    if total == 0:
        return EvalMetrics()

    # 1. 问题解决率：通过且未转人工的比例
    resolved_count = sum(1 for r in results if r.get("passed") and not r.get("need_human", False))
    resolution_rate = resolved_count / total

    # 2. 拒答准确率：对应该拒答/转人工的用例，正确转人工的比例
    should_refuse = [r for r in results if r.get("expected_outcome") in ["refused", "handoff"]]
    if should_refuse:
        correct_refuse = sum(1 for r in should_refuse if r.get("need_human", False) or r.get("passed"))
        refusal_accuracy = correct_refuse / len(should_refuse)
    else:
        refusal_accuracy = 1.0

    # 3. 平均响应时长
    response_times = [r.get("response_time", 0.0) for r in results if r.get("response_time")]
    avg_response_time = sum(response_times) / len(response_times) if response_times else 0.0

    # 4. 工具调用成功率
    all_tool_calls = []
    for r in results:
        for tc in r.get("tool_calls", []):
            all_tool_calls.append(tc)
    if all_tool_calls:
        successful_calls = sum(1 for tc in all_tool_calls if tc.get("success", False))
        tool_call_success_rate = successful_calls / len(all_tool_calls)
    else:
        tool_call_success_rate = 1.0  # 没有工具调用视为100%

    # 5. 人工转接率
    handoff_count = sum(1 for r in results if r.get("need_human", False))
    human_handoff_rate = handoff_count / total

    return EvalMetrics(
        total_cases=total,
        resolution_rate=resolution_rate,
        refusal_accuracy=refusal_accuracy,
        avg_response_time=avg_response_time,
        tool_call_success_rate=tool_call_success_rate,
        human_handoff_rate=human_handoff_rate,
    )


def calculate_category_stats(results: list[dict]) -> dict:
    """
    计算各类别统计

    Returns:
        {category: {total, passed, resolution_rate}} 字典
    """
    categories = {}
    for r in results:
        cat = r.get("category", "unknown")
        if cat not in categories:
            categories[cat] = {"total": 0, "passed": 0, "resolution_rate": 0.0}
        categories[cat]["total"] += 1
        if r.get("passed"):
            categories[cat]["passed"] += 1

    for cat_stats in categories.values():
        if cat_stats["total"] > 0:
            cat_stats["resolution_rate"] = cat_stats["passed"] / cat_stats["total"]

    return categories

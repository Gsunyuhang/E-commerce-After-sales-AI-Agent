"""
评估运行器
批量执行测试集，记录结果，生成评估报告
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from agent.graph import run_agent
from api.schemas import EvalReport, EvalResult
from config.settings import PROJECT_ROOT
from evaluation.metrics import calculate_category_stats, calculate_metrics
from utils.logger import logger


class EvaluationRunner:
    """评估运行器"""

    def __init__(self):
        self.test_cases_path = Path(__file__).parent / "test_cases.json"
        self.report_dir = PROJECT_ROOT / "data" / "eval_reports"
        self.report_dir.mkdir(parents=True, exist_ok=True)

    def load_test_cases(self) -> list[dict]:
        """加载测试用例"""
        with open(self.test_cases_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _evaluate_result(
        self,
        test_case: dict,
        agent_result: dict,
    ) -> tuple[bool, str]:
        """
        评估单条结果是否通过

        Returns:
            (passed, notes)
        """
        expected_intent = test_case.get("expected_intent", "")
        expected_outcome = test_case.get("expected_outcome", "")
        actual_intent = agent_result.get("intent", "")
        need_human = agent_result.get("need_human", False)

        notes_parts = []

        # 意图匹配检查（宽松匹配，complaint 也可接受 out_of_scope）
        intent_match = actual_intent == expected_intent
        if not intent_match:
            # 投诉类和超范围类可以互相接受
            if {actual_intent, expected_intent} <= {"complaint", "out_of_scope"}:
                intent_match = True
                notes_parts.append(f"意图近似匹配: actual={actual_intent}, expected={expected_intent}")
            else:
                notes_parts.append(f"意图不匹配: actual={actual_intent}, expected={expected_intent}")

        # 期望结果检查
        outcome_match = True
        if expected_outcome == "resolved":
            # 应该独立解决，不需要转人工
            if need_human:
                outcome_match = False
                notes_parts.append("期望独立解决但转了人工")
        elif expected_outcome == "refused":
            # 应该拒答/引导
            if not need_human and actual_intent != "out_of_scope":
                # 对于知识缺失类，need_human 为 True 才算正确
                outcome_match = False
                notes_parts.append("期望拒答但未转人工")
        elif expected_outcome == "handoff":
            # 应该转人工
            if not need_human:
                outcome_match = False
                notes_parts.append("期望转人工但未转")

        # 对于 ambiguous 类，如果 Agent 追问了订单号也算通过
        if test_case.get("category") == "ambiguous":
            response = agent_result.get("response", "")
            if "订单号" in response or "订单编号" in response:
                outcome_match = True
                notes_parts.append("正确追问了订单号")

        passed = intent_match and outcome_match
        return passed, "; ".join(notes_parts) if notes_parts else "通过"

    def run_single(self, test_case: dict) -> dict:
        """
        运行单条测试用例

        Returns:
            包含测试结果和 Agent 输出的字典
        """
        test_id = test_case.get("id", str(uuid.uuid4()))
        user_input = test_case.get("input", "")

        logger.info(f"评估测试 {test_id}: {user_input[:50]}...")

        # 调用 Agent
        agent_result = run_agent(
            user_input=user_input,
            session_id=f"eval_{test_id}",
            messages=[],
            context_summary="",
            turn_count=0,
        )

        # 评估结果
        passed, notes = self._evaluate_result(test_case, agent_result)

        return {
            "test_case_id": test_id,
            "input": user_input,
            "category": test_case.get("category", "unknown"),
            "expected_intent": test_case.get("expected_intent", ""),
            "actual_intent": agent_result.get("intent", ""),
            "expected_outcome": test_case.get("expected_outcome", ""),
            "actual_response": agent_result.get("response", ""),
            "response_time": agent_result.get("response_time", 0.0),
            "need_human": agent_result.get("need_human", False),
            "tool_calls": agent_result.get("tool_calls", []),
            "passed": passed,
            "notes": notes,
        }

    def run_all(self, test_cases: Optional[list[dict]] = None) -> EvalReport:
        """
        运行全量测试集

        Args:
            test_cases: 自定义测试用例（为空则使用默认测试集）

        Returns:
            评估报告
        """
        if test_cases is None:
            test_cases = self.load_test_cases()

        logger.info(f"开始评估，共 {len(test_cases)} 条测试用例")

        results = []
        for i, tc in enumerate(test_cases):
            logger.info(f"进度: {i+1}/{len(test_cases)}")
            try:
                result = self.run_single(tc)
                results.append(result)
            except Exception as e:
                logger.error(f"测试 {tc.get('id', i)} 异常: {e}")
                results.append({
                    "test_case_id": tc.get("id", str(i)),
                    "input": tc.get("input", ""),
                    "category": tc.get("category", "unknown"),
                    "expected_intent": tc.get("expected_intent", ""),
                    "actual_intent": "error",
                    "expected_outcome": tc.get("expected_outcome", ""),
                    "actual_response": "",
                    "response_time": 0.0,
                    "need_human": True,
                    "tool_calls": [],
                    "passed": False,
                    "notes": f"测试异常: {e}",
                })

        # 计算指标
        metrics = calculate_metrics(results)
        category_stats = calculate_category_stats(results)

        # 生成报告
        report = EvalReport(
            total_cases=metrics.total_cases,
            resolution_rate=metrics.resolution_rate,
            refusal_accuracy=metrics.refusal_accuracy,
            avg_response_time=metrics.avg_response_time,
            tool_call_success_rate=metrics.tool_call_success_rate,
            human_handoff_rate=metrics.human_handoff_rate,
            category_stats=category_stats,
            results=[
                EvalResult(
                    test_case_id=r["test_case_id"],
                    input=r["input"],
                    expected_intent=r["expected_intent"],
                    actual_intent=r["actual_intent"],
                    expected_outcome=r["expected_outcome"],
                    actual_response=r["actual_response"][:500],
                    response_time=r["response_time"],
                    need_human=r["need_human"],
                    tool_calls=r["tool_calls"],
                    passed=r["passed"],
                    notes=r["notes"],
                )
                for r in results
            ],
        )

        # 保存报告
        self._save_report(report)

        logger.info(
            f"评估完成: total={report.total_cases}, "
            f"resolution_rate={report.resolution_rate:.2%}, "
            f"refusal_accuracy={report.refusal_accuracy:.2%}, "
            f"avg_time={report.avg_response_time:.2f}s, "
            f"tool_success={report.tool_call_success_rate:.2%}, "
            f"handoff_rate={report.human_handoff_rate:.2%}"
        )

        return report

    def _save_report(self, report: EvalReport) -> Path:
        """保存评估报告到文件"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.report_dir / f"eval_report_{timestamp}.json"

        report_data = {
            "timestamp": timestamp,
            "metrics": {
                "total_cases": report.total_cases,
                "resolution_rate": report.resolution_rate,
                "refusal_accuracy": report.refusal_accuracy,
                "avg_response_time": report.avg_response_time,
                "tool_call_success_rate": report.tool_call_success_rate,
                "human_handoff_rate": report.human_handoff_rate,
            },
            "category_stats": report.category_stats,
            "results": [r.model_dump() for r in report.results],
        }

        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)

        logger.info(f"评估报告已保存: {report_path}")

        # 同时保存为 latest
        latest_path = self.report_dir / "latest_report.json"
        with open(latest_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)

        return report_path

    def load_latest_report(self) -> Optional[EvalReport]:
        """加载最近的评估报告"""
        latest_path = self.report_dir / "latest_report.json"
        if not latest_path.exists():
            return None

        with open(latest_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return EvalReport(
            total_cases=data["metrics"]["total_cases"],
            resolution_rate=data["metrics"]["resolution_rate"],
            refusal_accuracy=data["metrics"]["refusal_accuracy"],
            avg_response_time=data["metrics"]["avg_response_time"],
            tool_call_success_rate=data["metrics"]["tool_call_success_rate"],
            human_handoff_rate=data["metrics"]["human_handoff_rate"],
            category_stats=data.get("category_stats", {}),
            results=[EvalResult(**r) for r in data.get("results", [])],
        )

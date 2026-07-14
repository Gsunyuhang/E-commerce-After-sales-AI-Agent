"""
评估接口
"""

from fastapi import APIRouter

from api.schemas import EvalReport, EvalRequest
from evaluation.runner import EvaluationRunner
from utils.logger import logger

router = APIRouter(prefix="/api/eval", tags=["评估"])


@router.post("/run", response_model=EvalReport)
async def run_evaluation(request: EvalRequest) -> EvalReport:
    """运行评估测试集"""
    logger.info("收到评估请求")

    runner = EvaluationRunner()
    report = runner.run_all(test_cases=request.test_cases)

    logger.info(
        f"评估完成: total={report.total_cases}, "
        f"resolution_rate={report.resolution_rate:.2%}, "
        f"avg_time={report.avg_response_time:.2f}s"
    )

    return report


@router.get("/report", response_model=EvalReport)
async def get_latest_report() -> EvalReport:
    """获取最近的评估报告"""
    runner = EvaluationRunner()
    report = runner.load_latest_report()

    if report is None:
        return EvalReport(
            total_cases=0,
            resolution_rate=0.0,
            refusal_accuracy=0.0,
            avg_response_time=0.0,
            tool_call_success_rate=0.0,
            human_handoff_rate=0.0,
        )

    return report

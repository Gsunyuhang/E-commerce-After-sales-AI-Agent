"""
FastAPI 应用入口
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import chat, eval, orders
from config.settings import get_settings
from database.connection import create_tables
from utils.logger import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    logger.info("=== 电商售后 Agent 服务启动 ===")

    # 初始化数据库
    try:
        create_tables()
        logger.info("数据库表初始化完成")
    except Exception as e:
        logger.error(f"数据库初始化失败: {e}")

    # 初始化向量库（延迟加载，首次使用时自动初始化）
    logger.info("向量库将在首次使用时自动初始化")

    logger.info("=== 服务启动完成 ===")
    yield

    logger.info("=== 电商售后 Agent 服务关闭 ===")


def create_app() -> FastAPI:
    """创建 FastAPI 应用"""
    settings = get_settings()

    app = FastAPI(
        title="电商售后智能处理 Agent",
        description="基于 LangGraph 的电商售后智能客服系统",
        version="1.0.0",
        lifespan=lifespan,
    )

    # CORS 配置
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 注册路由
    app.include_router(chat.router)
    app.include_router(orders.router)
    app.include_router(eval.router)

    # 健康检查
    @app.get("/health", tags=["健康检查"])
    async def health_check():
        return {"status": "ok", "service": "ecommerce-agent"}

    # 根路径
    @app.get("/", tags=["根"])
    async def root():
        return {
            "service": "电商售后智能处理 Agent",
            "version": "1.0.0",
            "docs": "/docs",
            "endpoints": {
                "chat": "POST /api/chat",
                "chat_history": "GET /api/chat/history/{session_id}",
                "order_query": "GET /api/orders/{order_id}",
                "eval_run": "POST /api/eval/run",
                "eval_report": "GET /api/eval/report",
            },
        }

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
    )

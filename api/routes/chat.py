"""
对话接口
"""

import asyncio
import json
from typing import Optional

from fastapi import APIRouter, HTTPException
from langchain_core.messages import AIMessage, HumanMessage
from sse_starlette.sse import EventSourceResponse

from agent.graph import run_agent, run_agent_stream
from api.schemas import ChatHistoryResponse, ChatMessage, ChatRequest, ChatResponse
from database.connection import get_db_session
from database.models import ChatLog
from utils.logger import log_chat_turn, logger

router = APIRouter(prefix="/api/chat", tags=["对话"])


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    对话接口
    接收用户输入，返回 Agent 处理结果
    """
    logger.info(f"收到对话请求: session={request.session_id}, message={request.message[:50]}...")

    # 构建消息列表
    messages = []
    for msg in request.history:
        if msg.role == "user":
            messages.append(HumanMessage(content=msg.content))
        else:
            messages.append(AIMessage(content=msg.content))

    # 调用 Agent
    result = run_agent(
        user_input=request.message,
        session_id=request.session_id,
        messages=messages,
        context_summary=request.context_summary,
        turn_count=request.turn_count,
    )

    # 记录对话日志到数据库
    try:
        with get_db_session() as session:
            log_entry = ChatLog(
                session_id=request.session_id,
                user_input=request.message,
                intent=result.get("intent", ""),
                retrieved_docs=json.dumps(result.get("retrieved_docs", []), ensure_ascii=False),
                tool_calls=json.dumps(result.get("tool_calls", []), ensure_ascii=False),
                response=result.get("response", ""),
                response_time=result.get("response_time", 0.0),
                need_human=result.get("need_human", False),
            )
            session.add(log_entry)
    except Exception as e:
        logger.error(f"保存对话日志失败: {e}")

    # 记录结构化日志
    log_chat_turn(
        session_id=request.session_id,
        user_input=request.message,
        intent=result.get("intent"),
        retrieved_docs=result.get("retrieved_docs"),
        tool_calls=result.get("tool_calls"),
        response=result.get("response"),
        response_time=result.get("response_time"),
    )

    return ChatResponse(
        session_id=request.session_id,
        response=result.get("response", ""),
        intent=result.get("intent", ""),
        intent_confidence=result.get("intent_confidence", 0.0),
        need_human=result.get("need_human", False),
        confidence=result.get("confidence", 0.0),
        tool_calls=result.get("tool_calls", []),
        retrieved_docs=result.get("retrieved_docs", []),
        is_retrieval_reliable=result.get("is_retrieval_reliable", True),
        retrieval_max_score=result.get("retrieval_max_score", 0.0),
        context_summary=result.get("context_summary", ""),
        turn_count=result.get("turn_count", 0),
        response_time=result.get("response_time", 0.0),
        workflow_trace=result.get("workflow_trace", []),
        error=result.get("error"),
    )


@router.get("/history/{session_id}", response_model=ChatHistoryResponse)
async def get_chat_history(session_id: str) -> ChatHistoryResponse:
    """获取对话历史"""
    try:
        with get_db_session() as session:
            logs = (
                session.query(ChatLog)
                .filter_by(session_id=session_id)
                .order_by(ChatLog.created_at)
                .all()
            )

            messages = []
            for log in logs:
                messages.append(ChatMessage(role="user", content=log.user_input))
                if log.response:
                    messages.append(ChatMessage(role="assistant", content=log.response))

            return ChatHistoryResponse(session_id=session_id, messages=messages)
    except Exception as e:
        logger.error(f"获取对话历史失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stream")
async def chat_stream(request: ChatRequest):
    """
    流式对话接口（SSE）
    返回 Server-Sent Events，逐步推送进度和回答片段
    """
    logger.info(f"收到流式对话请求: session={request.session_id}, message={request.message[:50]}...")

    # 构建消息列表
    messages = []
    for msg in request.history:
        if msg.role == "user":
            messages.append(HumanMessage(content=msg.content))
        else:
            messages.append(AIMessage(content=msg.content))

    async def event_generator():
        final_result = None
        try:
            for event in run_agent_stream(
                user_input=request.message,
                session_id=request.session_id,
                messages=messages,
                context_summary=request.context_summary,
                turn_count=request.turn_count,
            ):
                yield {"event": event["type"], "data": json.dumps(event, ensure_ascii=False)}
                if event["type"] == "done":
                    final_result = event.get("result", {})
                await asyncio.sleep(0)
        except Exception as e:
            logger.error(f"流式对话异常: {e}")
            yield {"event": "done", "data": json.dumps({
                "type": "done",
                "result": {
                    "response": f"抱歉，系统遇到错误: {e}",
                    "intent": "error",
                    "need_human": True,
                    "response_time": 0,
                    "error": str(e),
                }
            }, ensure_ascii=False)}

        # 保存对话日志
        if final_result:
            try:
                with get_db_session() as session:
                    log_entry = ChatLog(
                        session_id=request.session_id,
                        user_input=request.message,
                        intent=final_result.get("intent", ""),
                        retrieved_docs=json.dumps(final_result.get("retrieved_docs", []), ensure_ascii=False),
                        tool_calls=json.dumps(final_result.get("tool_calls", []), ensure_ascii=False),
                        response=final_result.get("response", ""),
                        response_time=final_result.get("response_time", 0.0),
                        need_human=final_result.get("need_human", False),
                    )
                    session.add(log_entry)
            except Exception as e:
                logger.error(f"保存流式对话日志失败: {e}")

            log_chat_turn(
                session_id=request.session_id,
                user_input=request.message,
                intent=final_result.get("intent"),
                retrieved_docs=final_result.get("retrieved_docs"),
                tool_calls=final_result.get("tool_calls"),
                response=final_result.get("response"),
                response_time=final_result.get("response_time"),
            )

    return EventSourceResponse(event_generator())

"""
Streamlit 前端界面 — DeepSeek 风格
左侧导航栏（Logo + 新对话 + 历史会话）+ 右侧聊天界面
"""

import json
import uuid

import httpx
import streamlit as st

# ============================================
# 页面配置
# ============================================

API_BASE_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="售后 Agent",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================
# DeepSeek 风格 CSS
# ============================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@300;400;500;600;700&display=swap');
* { font-family: 'Noto Sans SC', 'Microsoft YaHei', sans-serif; }

/* 隐藏 Streamlit 默认元素 */
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }

/* ===== 侧边栏 ===== */
section[data-testid="stSidebar"] {
    background: #F9FAFB;
    border-right: 1px solid #E5E7EB;
}
section[data-testid="stSidebar"] > div { padding-top: 12px; }

/* 侧边栏按钮 — 无边框风格 */
section[data-testid="stSidebar"] .stButton > button {
    background: transparent;
    border: none;
    border-radius: 8px;
    color: #4B5563;
    font-size: 0.85rem;
    padding: 8px 12px;
    text-align: left;
    transition: all 0.15s;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background: #F3F4F6;
    color: #1F2937;
    border: none;
}

/* 侧边栏主按钮（新建对话） */
section[data-testid="stSidebar"] button[kind="primary"],
section[data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-primary"] {
    background: #4D6BFE !important;
    border: none !important;
    color: white !important;
    border-radius: 10px !important;
    padding: 10px 16px !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
    text-align: center !important;
    white-space: nowrap !important;
}
section[data-testid="stSidebar"] button[kind="primary"]:hover,
section[data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-primary"]:hover {
    background: #3D5BF0 !important;
    color: white !important;
    box-shadow: 0 2px 8px rgba(77,107,254,0.3);
}

/* ===== 主区域 ===== */
.stAppViewContainer, .stMainBlock { background: #FFFFFF; }

/* 聊天内容居中 */
[data-testid="stChatMessage"], [data-testid="stChatInput"] {
    max-width: 800px;
    margin-left: auto;
    margin-right: auto;
}

/* 聊天气泡 — 简洁无框 */
[data-testid="stChatMessage"] {
    padding: 0.5rem 0;
    border-radius: 0;
    box-shadow: none;
    border: none;
    background: transparent;
}

/* 聊天输入框 */
[data-testid="stChatInput"] {
    border-radius: 12px !important;
    border: 1px solid #E5E7EB !important;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06) !important;
}
[data-testid="stChatInput"] textarea { border-radius: 10px !important; }

/* 主区域按钮 — 紧凑建议卡片 */
main .stButton > button {
    background: #F9FAFB;
    border: 1px solid #E5E7EB;
    border-radius: 8px;
    padding: 7px 12px;
    font-size: 0.78rem;
    color: #4B5563;
    text-align: left;
    width: 100%;
    transition: all 0.15s;
    line-height: 1.4;
}
main .stButton > button:hover {
    background: #F3F4F6;
    border-color: #4D6BFE;
    color: #1F2937;
}
main .stButton { margin-bottom: 0.15rem; }
main .stHorizontalBlock { gap: 0.2rem !important; }
main [data-testid="stColumn"] { padding: 0 3px !important; }
main button[kind="primary"],
main .stButton > button[data-testid="stBaseButton-primary"] {
    background: #4D6BFE !important;
    border: none !important;
    color: white !important;
    border-radius: 10px !important;
    font-weight: 500 !important;
    text-align: center !important;
}
main button[kind="primary"]:hover,
main .stButton > button[data-testid="stBaseButton-primary"]:hover {
    background: #3D5BF0 !important;
    color: white !important;
}

/* 徽章 */
.ds-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 500;
    margin-right: 4px;
}
.ds-badge-blue   { background: #EBF2FF; color: #4D6BFE; }
.ds-badge-green  { background: #ECFDF5; color: #059669; }
.ds-badge-orange { background: #FFF7ED; color: #EA580C; }
.ds-badge-red    { background: #FEF2F2; color: #DC2626; }
.ds-badge-gray   { background: #F3F4F6; color: #6B7280; }

/* 评估指标卡片 */
.ds-metric {
    background: #F9FAFB;
    border: 1px solid #E5E7EB;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    text-align: center;
    transition: all 0.2s;
}
.ds-metric:hover { border-color: #4D6BFE; box-shadow: 0 2px 12px rgba(77,107,254,0.1); }
.ds-metric-value { font-size: 1.6rem; font-weight: 700; margin: 0.3rem 0; }
.ds-metric-label { font-size: 0.78rem; color: #6B7280; }
.ds-metric.green   .ds-metric-value { color: #059669; }
.ds-metric.blue    .ds-metric-value { color: #4D6BFE; }
.ds-metric.purple  .ds-metric-value { color: #7C3AED; }
.ds-metric.orange  .ds-metric-value { color: #EA580C; }
.ds-metric.red     .ds-metric-value { color: #DC2626; }

/* 进度条 / Spinner */
.stProgress > div > div > div { background: #4D6BFE; border-radius: 20px; }
.stSpinner > div { border-top-color: #4D6BFE !important; }

/* 滚动条 */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #D1D5DB; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #9CA3AF; }

/* 分隔线 */
hr { border: none; height: 1px; background: #E5E7EB; margin: 0.5rem 0; }

/* 折叠面板 */
details { border: none !important; background: transparent !important; }
summary { font-size: 0.8rem !important; color: #6B7280 !important; }

/* 表格 */
.stDataFrame { border-radius: 12px; overflow: hidden; border: 1px solid #E5E7EB; }
</style>
""", unsafe_allow_html=True)


# ============================================
# API 调用
# ============================================

def stream_chat_api(session_id: str, message: str, history: list, context_summary: str, turn_count: int):
    """流式调用对话 API，yield (event_type, data_dict)"""
    try:
        with httpx.stream(
            "POST",
            f"{API_BASE_URL}/api/chat/stream",
            json={
                "session_id": session_id,
                "message": message,
                "history": [{"role": h["role"], "content": h["content"]} for h in history],
                "context_summary": context_summary,
                "turn_count": turn_count,
            },
            timeout=120.0,
        ) as response:
            event_type = None
            for line in response.iter_lines():
                if not line:
                    continue
                if line.startswith("event:"):
                    event_type = line[6:].strip()
                elif line.startswith("data:"):
                    raw = line[5:].strip()
                    try:
                        data = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    yield event_type or data.get("type", "unknown"), data
                    event_type = None
    except httpx.ConnectError:
        yield "error", {"error": "无法连接到后端服务，请确保 FastAPI 已启动（端口 8000）"}
    except Exception as e:
        yield "error", {"error": str(e)}


def call_eval_api() -> dict:
    try:
        r = httpx.post(f"{API_BASE_URL}/api/eval/run", json={}, timeout=300.0)
        r.raise_for_status()
        return r.json()
    except httpx.ConnectError:
        return {"error": "无法连接到后端服务"}
    except Exception as e:
        return {"error": str(e)}


# ============================================
# 辅助函数
# ============================================

def intent_badge(intent: str) -> str:
    mapping = {
        "policy_qa": ("政策问答", "ds-badge-blue"),
        "order_query": ("订单查询", "ds-badge-green"),
        "return_process": ("退换货", "ds-badge-orange"),
        "complaint": ("投诉安抚", "ds-badge-red"),
        "out_of_scope": ("超范围", "ds-badge-gray"),
        "error": ("异常", "ds-badge-red"),
    }
    label, cls = mapping.get(intent, (intent, "ds-badge-gray"))
    return f'<span class="ds-badge {cls}">{label}</span>'


def metric_card(label: str, value: str, color: str = "blue") -> str:
    return f"""
    <div class="ds-metric {color}">
        <div class="ds-metric-label">{label}</div>
        <div class="ds-metric-value">{value}</div>
    </div>"""


# ============================================
# 多会话管理
# ============================================

def create_conversation() -> dict:
    return {
        "id": str(uuid.uuid4()),
        "title": "新对话",
        "messages": [],
        "context_summary": "",
        "turn_count": 0,
        "last_agent_result": None,
    }


def get_current_conversation() -> dict | None:
    for c in st.session_state.conversations:
        if c["id"] == st.session_state.current_conversation_id:
            return c
    return None


# ============================================
# 初始化会话状态
# ============================================

if "conversations" not in st.session_state:
    c = create_conversation()
    st.session_state.conversations = [c]
    st.session_state.current_conversation_id = c["id"]
if "view" not in st.session_state:
    st.session_state.view = "chat"


# ============================================
# 侧边栏 — DeepSeek 风格导航
# ============================================

with st.sidebar:
    # --- Logo ---
    st.markdown(
        '<div style="display:flex;align-items:center;gap:8px;padding:8px 16px 16px;">'
        '<span style="font-size:1.5rem;">🛒</span>'
        '<span style="font-size:1.15rem;font-weight:700;color:#4D6BFE;">售后助手</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    # --- 新建对话按钮 ---
    if st.button("✨  新建对话", type="primary", use_container_width=True, key="new_chat"):
        nc = create_conversation()
        st.session_state.conversations.insert(0, nc)
        st.session_state.current_conversation_id = nc["id"]
        st.session_state.view = "chat"
        st.rerun()

    # --- 历史对话标签 ---
    st.markdown(
        '<div style="padding:16px 16px 4px;font-size:0.72rem;color:#9CA3AF;'
        'font-weight:500;letter-spacing:0.04em;">历史对话</div>',
        unsafe_allow_html=True,
    )

    # --- 历史会话列表 ---
    for conv in st.session_state.conversations:
        is_active = conv["id"] == st.session_state.current_conversation_id
        prefix = "▸ " if is_active else ""
        title = conv["title"]
        if len(title) > 26:
            title = title[:26] + "…"
        if st.button(f"{prefix}{title}", key=f"conv_{conv['id']}", use_container_width=True):
            st.session_state.current_conversation_id = conv["id"]
            st.session_state.view = "chat"
            st.rerun()

    # --- 底部区域 ---
    st.markdown("")
    st.markdown("---")
    if st.button("📊  评估看板", use_container_width=True, key="eval_btn"):
        st.session_state.view = "eval"
        st.rerun()
    st.caption("💡 LangGraph + 通义千问")


# ============================================
# 主区域 — 评估看板
# ============================================

if st.session_state.view == "eval":
    col_back, col_title = st.columns([1, 5])
    with col_back:
        if st.button("← 返回对话", key="back_chat"):
            st.session_state.view = "chat"
            st.rerun()
    with col_title:
        st.markdown("### 📊 评估看板")

    st.markdown("---")

    col_run, col_refresh = st.columns([2, 1])
    with col_run:
        if st.button("🚀 运行全量评估（80条）", type="primary", use_container_width=True, key="run_eval"):
            with st.spinner("正在运行评估测试集…"):
                st.session_state.eval_report = call_eval_api()
            st.rerun()
    with col_refresh:
        if st.button("🔄 刷新报告", use_container_width=True, key="refresh_eval"):
            st.rerun()

    report = st.session_state.get("eval_report")

    if not report:
        st.markdown(
            '<div style="text-align:center;padding:3rem 1rem;color:#9CA3AF;">'
            '点击上方按钮运行全量评估</div>',
            unsafe_allow_html=True,
        )
    elif report.get("error"):
        st.error(f"❌ {report['error']}")
    else:
        st.markdown("#### 核心指标")
        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.markdown(metric_card("问题解决率", f"{report.get('resolution_rate',0):.1%}", "green"), unsafe_allow_html=True)
        with m2:
            st.markdown(metric_card("拒答准确率", f"{report.get('refusal_accuracy',0):.1%}", "blue"), unsafe_allow_html=True)
        with m3:
            st.markdown(metric_card("平均响应", f"{report.get('avg_response_time',0):.2f}s", "purple"), unsafe_allow_html=True)
        with m4:
            st.markdown(metric_card("工具成功率", f"{report.get('tool_call_success_rate',0):.1%}", "orange"), unsafe_allow_html=True)
        with m5:
            st.markdown(metric_card("人工转接率", f"{report.get('human_handoff_rate',0):.1%}", "red"), unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### 分类统计")
        cat_names = {
            "simple_qa": "简单问答", "multi_turn": "多轮流程",
            "ambiguous": "歧义模糊", "knowledge_missing": "知识缺失", "out_of_scope": "超范围",
        }
        cat_icons = {
            "simple_qa": "💬", "multi_turn": "🔄",
            "ambiguous": "❓", "knowledge_missing": "📚", "out_of_scope": "🚫",
        }
        for cat, stats in report.get("category_stats", {}).items():
            label = cat_names.get(cat, cat)
            icon = cat_icons.get(cat, "📋")
            total = stats.get("total", 0)
            passed = stats.get("passed", 0)
            rate = stats.get("resolution_rate", 0)
            c1, c2, c3 = st.columns([3, 1, 2])
            with c1:
                st.write(f"{icon}  **{label}**")
            with c2:
                st.write(f"{passed}/{total}")
            with c3:
                st.progress(rate)

        st.markdown("---")
        st.markdown("#### 详细结果")
        results = report.get("results", [])
        if results:
            rows = [{
                "ID": r.get("test_case_id", ""),
                "输入": r.get("input", "")[:35] + "…",
                "期望意图": r.get("expected_intent", ""),
                "实际意图": r.get("actual_intent", ""),
                "通过": "✅" if r.get("passed") else "❌",
                "耗时": f"{r.get('response_time', 0):.2f}s",
                "转人工": "是" if r.get("need_human") else "否",
            } for r in results]
            st.dataframe(rows, use_container_width=True, hide_index=True)


# ============================================
# 主区域 — 聊天界面
# ============================================

else:
    conv = get_current_conversation()
    if conv is None:
        conv = create_conversation()
        st.session_state.conversations.insert(0, conv)
        st.session_state.current_conversation_id = conv["id"]

    messages = conv["messages"]

    # --- 欢迎界面（无消息时） ---
    if not messages:
        st.markdown(
            '<div style="display:flex;flex-direction:column;align-items:center;'
            'padding:4rem 1rem 2rem;text-align:center;">'
            '<div style="font-size:3rem;margin-bottom:1rem;">🛒</div>'
            '<div style="font-size:1.8rem;font-weight:700;color:#1F2937;margin-bottom:0.5rem;">售后助手</div>'
            '<div style="font-size:0.95rem;color:#6B7280;">退换货 · 物流查询 · 政策答疑 · 工单流转</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        suggestions = [
            ("📋", "7天无理由退货怎么操作？"),
            ("🚚", "退货运费由谁承担？"),
            ("↩️", "我想退货，订单号DD20250701001"),
            ("📦", "帮我查一下DD20250702002的物流"),
        ]
        _, s_center, _ = st.columns([1, 3, 1])
        with s_center:
            s_cols = st.columns(2)
            for i, (icon, text) in enumerate(suggestions):
                with s_cols[i % 2]:
                    if st.button(f"{icon}  {text}", key=f"sugg_{i}"):
                        st.session_state.pending_input = text
                        st.rerun()

    # --- 消息列表 ---
    for msg in messages:
        with st.chat_message(msg["role"], avatar="🛒" if msg["role"] == "assistant" else None):
            st.markdown(msg["content"])

            if msg["role"] == "assistant":
                meta = msg.get("meta", {})
                ar = msg.get("agent_result")

                # 底部徽章
                if meta:
                    badges = []
                    if meta.get("intent"):
                        badges.append(intent_badge(meta["intent"]))
                    if meta.get("need_human"):
                        badges.append('<span class="ds-badge ds-badge-orange">已转人工</span>')
                    tc = meta.get("tool_calls", 0)
                    if tc:
                        badges.append(f'<span class="ds-badge ds-badge-green">工具 {tc} 次</span>')
                    badges.append(f'<span class="ds-badge ds-badge-gray">{meta.get("response_time", 0):.1f}s</span>')
                    st.markdown(
                        f'<div style="margin-top:0.5rem;padding-top:0.4rem;'
                        f'border-top:1px solid #F3F4F6;">{"".join(badges)}</div>',
                        unsafe_allow_html=True,
                    )

                # Agent 工作细节折叠面板
                if ar:
                    with st.expander("🔍 Agent 工作细节", expanded=False):
                        c1, c2 = st.columns(2)
                        with c1:
                            st.write(f"**意图**: {ar.get('intent', '')}")
                            st.write(f"**意图置信度**: {ar.get('intent_confidence', 0):.1%}")
                            st.write(f"**整体置信度**: {ar.get('confidence', 0):.1%}")
                            st.write(f"**转人工**: {'是' if ar.get('need_human') else '否'}")
                            slots = ar.get("slots", {})
                            if slots:
                                st.write("**槽位信息**")
                                st.json(slots)

                        with c2:
                            st.write(f"**检索可靠**: {'✅ 是' if ar.get('is_retrieval_reliable') else '❌ 否'}")
                            st.write(f"**最高相关度**: {ar.get('retrieval_max_score', 0):.3f}")
                            docs = ar.get("retrieved_docs", [])
                            if docs:
                                for d in docs:
                                    if isinstance(d, dict):
                                        for s in d.get("sources", []):
                                            score = s.get("score", 0.0)
                                            icon = "🟢" if score > 0.6 else "🟡" if score > 0.4 else "🔴"
                                            st.write(f"{icon} `{s.get('filename', '')}` — {score:.3f}")

                        tcs = ar.get("tool_calls", [])
                        if tcs:
                            st.write("**工具调用**")
                            for tc in tcs:
                                icon = "✅" if tc.get("success") else "❌"
                                st.write(f"{icon} {tc.get('tool_name', '')}")

                        trace = ar.get("workflow_trace", [])
                        if trace:
                            st.write("**工作流轨迹**")
                            for j, step in enumerate(trace):
                                st.write(f"{j + 1}. {step}")

                        if ar.get("error"):
                            st.error(f"⚠️ {ar['error']}")

                if meta.get("need_human"):
                    st.warning("⚠️ 已为您转接人工客服，请稍候")

    # --- 聊天输入 ---
    user_input = st.chat_input("请输入您的问题…")

    if "pending_input" in st.session_state and st.session_state.pending_input:
        user_input = st.session_state.pending_input
        st.session_state.pending_input = None

    if user_input:
        # 更新标题（首条消息）
        if not messages:
            conv["title"] = user_input[:25] + ("…" if len(user_input) > 25 else "")

        conv["messages"].append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.markdown(user_input)

        # 流式输出
        with st.chat_message("assistant", avatar="🛒"):
            prog_ph = st.empty()
            resp_ph = st.empty()
            meta_ph = st.empty()

            full_response = ""
            final_result = None

            for etype, data in stream_chat_api(
                session_id=conv["id"],
                message=user_input,
                history=conv["messages"][:-1],
                context_summary=conv["context_summary"],
                turn_count=conv["turn_count"],
            ):
                if etype == "error":
                    prog_ph.empty()
                    st.error(f"❌ {data.get('error', '未知错误')}")
                    break
                elif etype == "progress":
                    prog_ph.info(f"⏳ {data.get('message', '')}")
                elif etype == "chunk":
                    full_response += data.get("content", "")
                    prog_ph.empty()
                    resp_ph.markdown(full_response)
                elif etype == "done":
                    final_result = data.get("result", {})

            if final_result:
                prog_ph.empty()

                if not full_response and final_result.get("response"):
                    full_response = final_result["response"]
                    resp_ph.markdown(full_response)

                conv["context_summary"] = final_result.get("context_summary", "")
                conv["turn_count"] = final_result.get("turn_count", conv["turn_count"] + 1)
                conv["last_agent_result"] = final_result
                need_human = final_result.get("need_human", False)

                conv["messages"].append({
                    "role": "assistant",
                    "content": full_response,
                    "agent_result": final_result,
                    "meta": {
                        "intent": final_result.get("intent", ""),
                        "response_time": final_result.get("response_time", 0.0),
                        "need_human": need_human,
                        "tool_calls": len(final_result.get("tool_calls", [])),
                    },
                })

                # 实时徽章
                badges = [intent_badge(final_result.get("intent", ""))]
                if need_human:
                    badges.append('<span class="ds-badge ds-badge-orange">已转人工</span>')
                tc = len(final_result.get("tool_calls", []))
                if tc:
                    badges.append(f'<span class="ds-badge ds-badge-green">工具 {tc} 次</span>')
                badges.append(f'<span class="ds-badge ds-badge-gray">{final_result.get("response_time", 0):.1f}s</span>')
                meta_ph.markdown(
                    f'<div style="margin-top:0.5rem;padding-top:0.4rem;'
                    f'border-top:1px solid #F3F4F6;">{"".join(badges)}</div>',
                    unsafe_allow_html=True,
                )

                if need_human:
                    st.warning("⚠️ 已为您转接人工客服，请稍候")

        st.rerun()

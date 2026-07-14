"""
电商售后智能处理 Agent - 统一入口
"""

import sys
import os

# 确保项目根目录在 Python 路径中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def cmd_init():
    """初始化数据库 + 构建向量库"""
    print("=" * 50)
    print("  初始化数据库")
    print("=" * 50)

    from database.init_db import init_database
    init_database(reset=True)

    print()
    print("=" * 50)
    print("  构建向量知识库")
    print("=" * 50)

    from knowledge.document_loader import DocumentLoader
    from knowledge.vector_store import get_vector_store_manager

    loader = DocumentLoader()
    documents = loader.load_and_split()

    if documents:
        manager = get_vector_store_manager()
        manager.init_vector_store(documents)
        print(f"向量库构建完成，共 {len(documents)} 个文本块")
    else:
        print("警告：未找到知识库文档，请检查 data/knowledge_base/ 目录")

    print()
    print("=" * 50)
    print("  初始化完成！")
    print("=" * 50)


def cmd_server():
    """启动 FastAPI 后端"""
    import uvicorn
    from config.settings import get_settings

    settings = get_settings()
    print(f"启动 FastAPI 后端服务: http://{settings.api_host}:{settings.api_port}")
    print(f"API 文档: http://{settings.api_host}:{settings.api_port}/docs")

    uvicorn.run(
        "api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
    )


def cmd_frontend():
    """启动 Streamlit 前端"""
    from config.settings import get_settings

    settings = get_settings()
    frontend_path = os.path.join(os.path.dirname(__file__), "frontend", "streamlit_app.py")

    print(f"启动 Streamlit 前端: http://localhost:{settings.frontend_port}")

    os.system(f"streamlit run {frontend_path} --server.port {settings.frontend_port} --server.headless true")


def cmd_eval():
    """运行全量评估"""
    print("=" * 50)
    print("  运行评估测试集")
    print("=" * 50)

    from evaluation.runner import EvaluationRunner

    runner = EvaluationRunner()
    report = runner.run_all()

    print()
    print("=" * 50)
    print("  评估结果")
    print("=" * 50)
    print(f"  测试用例总数: {report.total_cases}")
    print(f"  问题解决率:   {report.resolution_rate:.1%}")
    print(f"  拒答准确率:   {report.refusal_accuracy:.1%}")
    print(f"  平均响应时长: {report.avg_response_time:.2f}s")
    print(f"  工具调用成功率: {report.tool_call_success_rate:.1%}")
    print(f"  人工转接率:   {report.human_handoff_rate:.1%}")
    print()

    print("分类统计:")
    cat_names = {
        "simple_qa": "简单问答",
        "multi_turn": "多轮流程",
        "ambiguous": "歧义模糊",
        "knowledge_missing": "知识缺失",
        "out_of_scope": "超范围",
    }
    for cat, stats in report.category_stats.items():
        cat_label = cat_names.get(cat, cat)
        print(f"  {cat_label}: {stats['passed']}/{stats['total']} ({stats['resolution_rate']:.1%})")


def cmd_test():
    """运行单元测试"""
    import pytest

    print("运行单元测试...")
    pytest.main(["-v", "tests/"])


def show_help():
    """显示帮助信息"""
    print("""
电商售后智能处理 Agent - 使用说明
================================

用法: python main.py <command>

可用命令:
  init      初始化数据库 + 构建向量知识库
  server    启动 FastAPI 后端服务 (端口 8000)
  frontend  启动 Streamlit 前端界面 (端口 8501)
  eval      运行全量评估测试集 (80条)
  test      运行单元测试
  help      显示此帮助信息

快速开始:
  1. 复制 .env.example 为 .env，填入 DASHSCOPE_API_KEY
  2. pip install -r requirements.txt
  3. python main.py init      # 初始化
  4. python main.py server    # 启动后端
  5. python main.py frontend  # 启动前端（新终端）
""")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        show_help()
        sys.exit(0)

    command = sys.argv[1].lower()

    commands = {
        "init": cmd_init,
        "server": cmd_server,
        "frontend": cmd_frontend,
        "eval": cmd_eval,
        "test": cmd_test,
        "help": show_help,
    }

    if command in commands:
        commands[command]()
    else:
        print(f"未知命令: {command}")
        show_help()
        sys.exit(1)

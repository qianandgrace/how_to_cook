"""
完整流程验证脚本：初始化系统 → 构建知识库 → 问答
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")  # 显式加载根目录.env

from config import DEFAULT_CONFIG, RAGConfig
from main import RecipeRAGSystem


def main():
    rag = RecipeRAGSystem()
    rag.initialize_system()
    rag.build_knowledge_base()

    questions = [
        "宫保鸡丁怎么做？",
        "推荐几个简单的素菜",
        "红烧肉需要什么食材？",
    ]
    for q in questions:
        print("\n" + "=" * 70)
        answer = rag.ask_question(q, stream=False)
        print("\n【回答】")
        print(answer)


if __name__ == "__main__":
    main()

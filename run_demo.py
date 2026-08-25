"""
GraphRAG 演示脚本：初始化系统 → 加载知识库 → 多类型问答
包含传统混合检索、多跳图检索、复杂约束查询的路由效果对比
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent / ".env")

from main import AdvancedGraphRAGSystem


def main():
    rag = AdvancedGraphRAGSystem()
    rag.initialize_system()
    rag.build_knowledge_base()

    questions = [
        "红烧肉怎么做？",
        "推荐几个川菜",
        "鸡肉和什么蔬菜搭配比较好？",
        "减肥期间适合吃什么菜？",
        "适合糖尿病人吃的低糖川菜有哪些，并且制作时间不超过30分钟？",
    ]

    for q in questions:
        print("\n" + "=" * 70)
        result, analysis = rag.ask_question_with_routing(q, stream=False)
        strategy = analysis.recommended_strategy.value if analysis else "?"
        print(f"\n【路由策略】{strategy}  |  复杂度={analysis.query_complexity:.2f} 关系密集度={analysis.relationship_intensity:.2f}" if analysis else "")
        print("\n【回答】")
        print(result)

    rag._cleanup()


if __name__ == "__main__":
    main()

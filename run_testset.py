"""
使用GraphRAG在测试集（test/questions.json）上运行，保存结果到 test/graph_rag_results.json

用法:
  python run_testset.py                       # 跑全部 20 题
  python run_testset.py --only A1 A2 B1       # 只跑指定题目id
  python run_testset.py --append --only C1 C2 # 追加/合并到已有结果（断点续跑）

流程与 main.ask_question_with_routing 的检索核心一致：
  智能路由 route_query（内部按策略分发到 传统混合检索/图RAG/组合）→ 生成回答
每道题结束后立即写盘（checkpoint），中断可续跑。
"""
import json
import sys
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent / ".env")

from config import DEFAULT_CONFIG
from main import AdvancedGraphRAGSystem


def run_single(rag: AdvancedGraphRAGSystem, question: str) -> dict:
    """复刻 main.ask_question_with_routing 的检索核心并返回答案 + 路由/检索诊断"""
    result = {"question": question}

    # 1. 智能路由检索（内部按策略分发：传统/图RAG/组合）
    relevant_docs, analysis = rag.query_router.route_query(question, rag.config.top_k)

    if analysis is not None:
        result["strategy"] = analysis.recommended_strategy.value
        result["query_complexity"] = round(analysis.query_complexity, 3)
        result["relationship_intensity"] = round(analysis.relationship_intensity, 3)
        result["reasoning_required"] = analysis.reasoning_required
        result["entity_count"] = analysis.entity_count
        result["confidence"] = round(analysis.confidence, 3)
        result["reasoning"] = analysis.reasoning
    else:
        result["strategy"] = "unknown"

    # 2. 检索到的文档详情
    result["retrieved_items"] = []
    names = []
    for doc in relevant_docs:
        recipe_name = doc.metadata.get("recipe_name", "")
        names.append(recipe_name)
        result["retrieved_items"].append({
            "recipe_name": recipe_name,
            "search_type": doc.metadata.get("search_type")
                           or doc.metadata.get("search_method")
                           or doc.metadata.get("retrieval_level", ""),
            "score": round(float(doc.metadata.get("final_score",
                            doc.metadata.get("relevance_score",
                            doc.metadata.get("score", 0.0)))), 4),
            "route_strategy": doc.metadata.get("route_strategy", ""),
            "preview": doc.page_content[:60].replace("\n", " ")
        })
    # 去重保序的菜谱名
    seen = set()
    result["retrieved_documents"] = [n for n in names if not (n in seen or seen.add(n))]

    # 3. 生成回答
    if not relevant_docs:
        result["answer"] = "抱歉，没有找到相关的烹饪信息。请尝试其他问题。"
    else:
        result["answer"] = rag.generation_module.generate_adaptive_answer(question, relevant_docs)
    return result


def build_output(results_by_id, items):
    """按 questions.json 的顺序输出"""
    ordered = [results_by_id[it["id"]] for it in items if it["id"] in results_by_id]
    return {
        "model": "graph_rag",
        "note": "GraphRAG：智能路由(传统混合/图RAG/组合) + 双层图检索 + 多跳/子图 + Round-robin",
        "config": {
            "neo4j_uri": DEFAULT_CONFIG.neo4j_uri,
            "milvus_database": DEFAULT_CONFIG.milvus_database,
            "milvus_collection": DEFAULT_CONFIG.milvus_collection_name,
            "embedding_model": DEFAULT_CONFIG.embedding_model,
            "llm_model": DEFAULT_CONFIG.llm_model,
            "top_k": DEFAULT_CONFIG.top_k
        },
        "total": len(ordered),
        "results": ordered
    }


def main():
    parser = argparse.ArgumentParser(description="在GraphRAG上运行测试集")
    parser.add_argument("--only", nargs="*", default=None, help="只运行指定题目id，如 A1 B2")
    parser.add_argument("--append", action="store_true", help="追加/合并到已有结果文件（断点续跑）")
    parser.add_argument("--force", action="store_true", help="强制重跑（即使结果已存在）")
    args = parser.parse_args()

    test_dir = Path(__file__).resolve().parent / "test"
    out_path = test_dir / "graph_rag_results.json"
    with open(test_dir / "questions.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    items = data["questions"]
    if args.only:
        id_set = set(args.only)
        items = [it for it in items if it["id"] in id_set]
    if not items:
        print("没有需要运行的题目")
        return

    # 断点续跑：载入已有结果
    results_by_id = {}
    if args.append and out_path.exists():
        existing = json.load(open(out_path, encoding="utf-8"))
        results_by_id = {r["id"]: r for r in existing["results"]}
        if not args.force:
            skip = [it["id"] for it in items if it["id"] in results_by_id and "error" not in results_by_id[it["id"]]]
            if skip:
                print(f"跳过已完成的题目: {skip}")
                items = [it for it in items if it["id"] not in results_by_id or "error" in results_by_id[it["id"]]]

    print("初始化GraphRAG系统...")
    rag = AdvancedGraphRAGSystem()
    rag.initialize_system()
    rag.build_knowledge_base()

    for idx, item in enumerate(items, 1):
        qid = item["id"]
        q = item["question"]
        print(f"\n[{idx}/{len(items)}] {qid} [{item['group']}] {q}", flush=True)
        t0 = time.time()
        try:
            r = run_single(rag, q)
            r["id"] = qid
            r["group"] = item["group"]
            r["category"] = item["category"]
            r["elapsed_sec"] = round(time.time() - t0, 1)
            results_by_id[qid] = r
            print(f"  策略={r['strategy']} 复杂度={r.get('query_complexity')} 耗时={r['elapsed_sec']}s 检索到={r['retrieved_documents']}", flush=True)
        except Exception as e:
            print(f"  出错: {e}", flush=True)
            results_by_id[qid] = {"id": qid, "question": q, "group": item["group"], "error": str(e)}

        # 每道题结束立即写盘（checkpoint）
        test_dir.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(build_output(results_by_id, data["questions"]), f, ensure_ascii=False, indent=2)

    rag._cleanup()
    done = [it["id"] for it in data["questions"] if it["id"] in results_by_id]
    print(f"\n✅ 已完成 {len(done)}/20 道题，结果已保存到 {out_path}")


if __name__ == "__main__":
    main()

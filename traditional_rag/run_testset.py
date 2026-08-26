"""
使用传统RAG在测试集（test/questions.json）上运行，保存结果到 test/traditional_rag_results.json

用法:
  python run_testset.py                       # 跑全部 20 题
  python run_testset.py --only A1 A2 B1       # 只跑指定题目id
  python run_testset.py --append --only C1 C2 # 追加/合并到已有结果（断点续跑）

流程与 main.ask_question 一致（查询路由 → 重写 → 混合检索 → 父文档回传 → 生成），
额外捕获路由类型、重写查询、过滤条件、检索到的文档与耗时等诊断信息，便于与GraphRAG对比。
每道题结束后立即写盘（checkpoint），中断可续跑。
"""
import json
import sys
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from config import DEFAULT_CONFIG
from main import RecipeRAGSystem


def run_single(rag: RecipeRAGSystem, question: str) -> dict:
    """复刻 main.ask_question 流程并返回答案 + 诊断信息"""
    result = {"question": question}

    route_type = rag.generation_module.query_router(question)
    result["route_type"] = route_type

    if route_type == 'list':
        rewritten_query = question
    else:
        rewritten_query = rag.generation_module.query_rewrite(question)
    result["rewritten_query"] = rewritten_query

    filters = rag._extract_filters_from_query(question)
    result["filters"] = filters

    if filters:
        chunks = rag.retrieval_module.metadata_filtered_search(rewritten_query, filters, top_k=rag.config.top_k)
    else:
        chunks = rag.retrieval_module.hybrid_search(rewritten_query, top_k=rag.config.top_k)

    result["retrieved_chunks"] = []
    for c in chunks:
        result["retrieved_chunks"].append({
            "dish_name": c.metadata.get("dish_name", ""),
            "rrf_score": round(c.metadata.get("rrf_score", 0), 4),
            "preview": c.page_content[:50].replace("\n", " ")
        })

    if not chunks:
        result["answer"] = "抱歉，没有找到相关的食谱信息。请尝试其他菜品名称或关键词。"
        result["retrieved_documents"] = []
        return result

    parent_docs = rag.data_module.get_parent_documents(chunks)
    result["retrieved_documents"] = [d.metadata.get("dish_name", "") for d in parent_docs]

    if route_type == 'list':
        answer = rag.generation_module.generate_list_answer(question, parent_docs)
    elif route_type == 'detail':
        answer = rag.generation_module.generate_step_by_step_answer(question, parent_docs)
    else:
        answer = rag.generation_module.generate_basic_answer(question, parent_docs)
    result["answer"] = answer
    return result


def build_output(results_by_id, items):
    ordered = [results_by_id[it["id"]] for it in items if it["id"] in results_by_id]
    return {
        "model": "traditional_rag",
        "note": "传统RAG：Milvus向量 + BM25 + RRF混合检索",
        "config": {
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
    parser = argparse.ArgumentParser(description="在传统RAG上运行测试集")
    parser.add_argument("--only", nargs="*", default=None, help="只运行指定题目id，如 A1 B2")
    parser.add_argument("--append", action="store_true", help="追加/合并到已有结果文件（断点续跑）")
    args = parser.parse_args()

    test_dir = Path(__file__).resolve().parent.parent / "test"
    out_path = test_dir / "traditional_rag_results.json"
    with open(test_dir / "questions.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    items = data["questions"]
    if args.only:
        id_set = set(args.only)
        items = [it for it in items if it["id"] in id_set]
    if not items:
        print("没有需要运行的题目")
        return

    results_by_id = {}
    if args.append and out_path.exists():
        existing = json.load(open(out_path, encoding="utf-8"))
        results_by_id = {r["id"]: r for r in existing["results"]}
        skip = [it["id"] for it in items if it["id"] in results_by_id and "error" not in results_by_id[it["id"]]]
        if skip:
            print(f"跳过已完成的题目: {skip}")
            items = [it for it in items if it["id"] not in results_by_id or "error" in results_by_id[it["id"]]]

    print("初始化传统RAG系统...")
    rag = RecipeRAGSystem()
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
            print(f"  路由={r['route_type']} 耗时={r['elapsed_sec']}s 检索到={r['retrieved_documents']}", flush=True)
        except Exception as e:
            print(f"  出错: {e}", flush=True)
            results_by_id[qid] = {"id": qid, "question": q, "group": item["group"], "error": str(e)}

        # 每道题结束立即写盘（checkpoint）
        test_dir.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(build_output(results_by_id, data["questions"]), f, ensure_ascii=False, indent=2)

    done = [it["id"] for it in data["questions"] if it["id"] in results_by_id]
    print(f"\n✅ 已完成 {len(done)}/20 道题，结果已保存到 {out_path}")


if __name__ == "__main__":
    main()

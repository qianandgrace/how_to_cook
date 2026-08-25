"""
使用传统RAG在测试集（test/questions.json）上运行，保存结果到 test/traditional_rag_results.json
流程与 main.ask_question 一致（查询路由 → 重写 → 混合检索 → 父文档回传 → 生成），
额外捕获路由类型、重写查询、过滤条件、检索到的文档等诊断信息，便于与 GraphRAG 对比。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from config import DEFAULT_CONFIG
from main import RecipeRAGSystem


def run_single(rag: RecipeRAGSystem, question: str) -> dict:
    """复刻 main.ask_question 流程并返回答案 + 诊断信息"""
    result = {"question": question}

    # 1. 查询路由
    route_type = rag.generation_module.query_router(question)
    result["route_type"] = route_type

    # 2. 查询重写（list 类型保持原样）
    if route_type == 'list':
        rewritten_query = question
    else:
        rewritten_query = rag.generation_module.query_rewrite(question)
    result["rewritten_query"] = rewritten_query

    # 3. 元数据过滤提取
    filters = rag._extract_filters_from_query(question)
    result["filters"] = filters

    # 4. 混合检索
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

    # 5. 未检索到
    if not chunks:
        result["answer"] = "抱歉，没有找到相关的食谱信息。请尝试其他菜品名称或关键词。"
        result["retrieved_documents"] = []
        return result

    # 6. 父文档回传
    parent_docs = rag.data_module.get_parent_documents(chunks)
    result["retrieved_documents"] = [d.metadata.get("dish_name", "") for d in parent_docs]

    # 7. 按路由类型生成答案
    if route_type == 'list':
        answer = rag.generation_module.generate_list_answer(question, parent_docs)
    elif route_type == 'detail':
        answer = rag.generation_module.generate_step_by_step_answer(question, parent_docs)
    else:
        answer = rag.generation_module.generate_basic_answer(question, parent_docs)
    result["answer"] = answer
    return result


def main():
    test_dir = Path(__file__).resolve().parent.parent / "test"
    with open(test_dir / "questions.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    print("初始化传统RAG系统...")
    rag = RecipeRAGSystem()
    rag.initialize_system()
    rag.build_knowledge_base()

    items = data["questions"]
    results = []
    for idx, item in enumerate(items, 1):
        qid = item["id"]
        q = item["question"]
        print(f"\n[{idx}/{len(items)}] {qid} [{item['group']}] {q}")
        try:
            r = run_single(rag, q)
            r["id"] = qid
            r["group"] = item["group"]
            r["category"] = item["category"]
            results.append(r)
            print(f"  路由={r['route_type']} 检索到={r['retrieved_documents']}")
        except Exception as e:
            print(f"  出错: {e}")
            results.append({"id": qid, "question": q, "group": item["group"], "error": str(e)})

    output = {
        "model": "traditional_rag",
        "note": "传统RAG：Milvus向量 + BM25 + RRF混合检索",
        "config": {
            "milvus_database": DEFAULT_CONFIG.milvus_database,
            "milvus_collection": DEFAULT_CONFIG.milvus_collection_name,
            "embedding_model": DEFAULT_CONFIG.embedding_model,
            "llm_model": DEFAULT_CONFIG.llm_model,
            "top_k": DEFAULT_CONFIG.top_k
        },
        "total": len(results),
        "results": results
    }

    test_dir.mkdir(parents=True, exist_ok=True)
    out_path = test_dir / "traditional_rag_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 共 {len(results)} 道题，结果已保存到 {out_path}")


if __name__ == "__main__":
    main()

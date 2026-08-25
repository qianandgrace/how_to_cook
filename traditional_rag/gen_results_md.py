"""
把 test/{model}_results.json 渲染成可读的 markdown 摘要（兼容传统RAG与GraphRAG两种格式）
用法: python gen_results_md.py <results.json路径> <输出md路径>
"""
import json
import sys
from collections import Counter
from pathlib import Path

GROUPS = {"basic": "基础", "multi_hop": "多跳", "complex": "复杂"}


def _g(r, *keys, default=""):
    """按顺序取第一个存在的键"""
    for k in keys:
        if r.get(k) is not None:
            return r[k]
    return default


def doc_names(docs):
    """把检索文档列表转成名字列表（兼容字符串列表 / dict列表）"""
    names = []
    for d in docs:
        if isinstance(d, str):
            names.append(d)
        else:
            names.append(d.get("recipe_name") or d.get("dish_name") or d.get("name") or "")
    return [n for n in names if n]


def detail_lines(docs):
    """渲染检索详情（兼容 retrieved_chunks / retrieved_items）"""
    lines = []
    for c in docs:
        name = c.get("recipe_name") or c.get("dish_name") or c.get("name", "")
        score = c.get("final_score", c.get("rrf_score", c.get("relevance_score", c.get("score", ""))))
        source = c.get("search_type") or c.get("search_method") or c.get("retrieval_level", "")
        prev = c.get("preview", "")
        parts = []
        if name:
            parts.append(f"`{name}`")
        if score not in (None, ""):
            parts.append(f"score={score}")
        if source:
            parts.append(f"src={source}")
        lines.append(f"  - {' '.join(parts)} {prev}")
    return lines


def main():
    in_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2])
    d = json.load(open(in_path, encoding="utf-8"))

    lines = [f"# {d.get('model', 'RAG')} 测试集结果（{d.get('total', 0)}题）", ""]
    cfg = d.get("config", {})
    if cfg:
        coll = f"{cfg.get('milvus_database', '?')}.{cfg.get('milvus_collection', '?')}"
        lines.append(f"- 配置：Milvus {coll} | embedding {cfg.get('embedding_model', '?')} | llm {cfg.get('llm_model', '?')} | top_k {cfg.get('top_k', '?')}")
        if cfg.get("neo4j_uri"):
            lines.append(f"- Neo4j：{cfg['neo4j_uri']}")
        lines.append("")
    if d.get("note"):
        lines.append(f"- {d['note']}")
        lines.append("")

    cnt = Counter(r["group"] for r in d["results"])
    lines.append("题目分布：" + "，".join(f"{GROUPS.get(k, k)} {v}道" for k, v in cnt.items()))
    lines.append("")

    for r in d["results"]:
        g = GROUPS.get(r.get("group", ""), r.get("group", ""))
        lines.append(f"## {r['id']} [{g}] {r['question']}")
        lines.append("")
        if "error" in r:
            lines.append(f"**出错**：{r['error']}")
            lines.append("")
            continue

        strategy = _g(r, "strategy", "route_type")
        if strategy:
            lines.append(f"- 路由：`{strategy}`")
        if "query_complexity" in r:
            lines.append(f"- 复杂度：{r['query_complexity']} | 关系密集度：{r['relationship_intensity']} | 推理需求：{'是' if r.get('reasoning_required') else '否'} | 实体数：{r.get('entity_count')} | 置信度：{r.get('confidence')}")
        if r.get("reasoning"):
            lines.append(f"- 路由理由：{r['reasoning']}")
        if "rewritten_query" in r:
            lines.append(f"- 重写查询：`{r['rewritten_query']}`")
        if r.get("filters"):
            lines.append(f"- 过滤条件：{json.dumps(r['filters'], ensure_ascii=False)}")
        if r.get("retrieved_documents"):
            lines.append(f"- 检索到的文档：{', '.join(doc_names(r['retrieved_documents']))}")
        if r.get("retrieved_chunks"):
            lines.append("- 检索到的子块：")
            lines.extend(detail_lines(r["retrieved_chunks"]))
        if r.get("retrieved_items"):
            lines.append("- 检索到的详情：")
            lines.extend(detail_lines(r["retrieved_items"]))
        if r.get("answer"):
            lines.append("- 回答摘要：")
            ans = r["answer"].replace("\n", " ").strip()
            lines.append(f"  > {ans[:260]}{'...' if len(ans) > 260 else ''}")
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"written {out_path}")


if __name__ == "__main__":
    main()

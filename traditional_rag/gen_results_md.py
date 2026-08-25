"""
把 test/{model}_results.json 渲染成可读的 markdown 摘要
用法: python gen_results_md.py <results.json路径> <输出md路径>
"""
import json
import sys
from collections import Counter
from pathlib import Path

GROUPS = {"basic": "基础", "multi_hop": "多跳", "complex": "复杂"}


def main():
    in_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2])
    d = json.load(open(in_path, encoding="utf-8"))

    lines = [f"# {d.get('model', 'RAG')} 测试集结果（{d.get('total', 0)}题）", ""]
    cfg = d.get("config", {})
    if cfg:
        lines.append(f"- 配置：Milvus {cfg.get('milvus_database', '?')}.{cfg.get('milvus_collection', '?')} | embedding {cfg.get('embedding_model', '?')} | llm {cfg.get('llm_model', '?')} | top_k {cfg.get('top_k', '?')}")
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
        lines.append(f"- 路由类型：`{r['route_type']}`")
        lines.append(f"- 重写查询：`{r['rewritten_query']}`")
        lines.append(f"- 过滤条件：{json.dumps(r['filters'], ensure_ascii=False) if r.get('filters') else '无'}")
        lines.append(f"- 检索到的文档：{', '.join(r['retrieved_documents'])}")
        lines.append("- 检索到的子块：")
        for c in r["retrieved_chunks"]:
            lines.append(f"  - `{c['dish_name']}` (rrf={c['rrf_score']}) {c['preview']}")
        lines.append("- 回答摘要：")
        ans = r["answer"].replace("\n", " ").strip()
        lines.append(f"  > {ans[:260]}{'...' if len(ans) > 260 else ''}")
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"written {out_path}")


if __name__ == "__main__":
    main()

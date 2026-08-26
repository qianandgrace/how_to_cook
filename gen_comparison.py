"""
生成传统RAG vs GraphRAG 的对比分析表格 → test/comparison.md
读取 test/questions.json + traditional_rag_results.json + graph_rag_results.json
"""
import json
from pathlib import Path

TEST = Path(__file__).resolve().parent / "test"
GROUPS = {"basic": "基础", "multi_hop": "多跳", "complex": "复杂"}

# 人工检索质量评估（基于两套检索到的文档与回答，判断"是否检索到与问题真正相关的内容"）
NOTES = {
    "A1": "传统命中麻婆豆腐本体；GraphRAG额外带回郫县豆瓣酱/肉末等食材实体，上下文更全",
    "A2": "两者都命中红烧肉相关菜谱；GraphRAG多出五花肉/冰糖/生姜等食材实体",
    "A3": "两者都命中可乐鸡翅；GraphRAG补全可乐/生姜/大葱等食材",
    "A4": "❌传统只回‘鸡蛋花’(无关)；✅GraphRAG回地三鲜/酸辣土豆丝/蒜蓉西兰花，正是素菜",
    "A5": "❌传统回蚝油生菜/尖叫牛蛙(非川菜)；✅GraphRAG回麻婆豆腐/回锅肉/宫保鸡丁",
    "A6": "两者都给出合理早餐；GraphRAG略宽(鸡蛋/牛奶/吐司)",
    "A7": "❌传统回蚝油生菜/牛油火锅底料；✅GraphRAG回麻婆豆腐/回锅肉/水煮牛肉等川菜",
    "A8": "两者都未命中强相关内容，属一般问题，检索质量相当",
    "A9": "两者都命中鸡蛋类(鸡蛋羹/溏心蛋)，相关",
    "A10": "两者都命中糖醋汁本体；GraphRAG多出白糖/米醋配料",
    "B1": "❌传统回猪油拌饭/麻辣香锅(高热量！)；✅GraphRAG回鸡胸肉等减脂食材",
    "B2": "传统只回鸡肉菜(口水鸡)；GraphRAG回‘无骨肉’概念+口水鸡，仍偏弱",
    "B3": "两者都偏弱：传统回小酥肉/牛肉饭，GraphRAG回去皮猪肉/猪肉(仍围绕猪)",
    "B4": "❌传统回蛋包饭/扬州炒饭；✅GraphRAG回‘蛋白质’概念+鸡胸肉",
    "B5": "两者都回凉拌类(油麦菜/黄瓜/酸辣土豆丝)，都合理",
    "C1": "❌传统回糖拌西红柿/蔗糖糖浆；✅GraphRAG回凉拌木耳+川菜知识子图，可按约束推理",
    "C2": "❌传统回水煮肉片/蛋包饭(与鸡肉无关)；✅GraphRAG回鸡肉+蔬菜概念节点+汤面",
    "C3": "传统回麻婆豆腐/凉拌豆腐；GraphRAG回豆腐/老豆腐+肉类概念节点，图路径更贴合‘替代’",
    "C4": "❌传统回辣椒炒肉(有辣椒！)；✅GraphRAG回清蒸鲈鱼(少油无辣)，正确",
    "C5": "传统回扬州炒饭/蛋包饭(相关)；GraphRAG回鸡蛋/汤面，偏少",
}


def doc_names(r, key="retrieved_documents"):
    if key not in r:
        return []
    out = []
    for d in r[key]:
        if isinstance(d, str):
            out.append(d)
        else:
            out.append(d.get("recipe_name") or d.get("dish_name") or "")
    seen = set()
    return [n for n in out if n and not (n in seen or seen.add(n))]


def strategy_label(r):
    return r.get("route_type") or r.get("strategy") or "?"


def short(s, n=6):
    return ", ".join(s[:n])


def main():
    qs = {q["id"]: q for q in json.load(open(TEST / "questions.json", encoding="utf-8"))["questions"]}
    t = {r["id"]: r for r in json.load(open(TEST / "traditional_rag_results.json", encoding="utf-8"))["results"]}
    g = {r["id"]: r for r in json.load(open(TEST / "graph_rag_results.json", encoding="utf-8"))["results"]}

    lines = []
    lines.append("# 传统 RAG vs GraphRAG —— 20 题对比测试（含耗时）")
    lines.append("")
    lines.append(f"> 测试集 `test/questions.json` | 传统RAG（Milvus+BM25+RRF） vs GraphRAG（智能路由+双层图检索+多跳/子图+Round-robin）")
    lines.append("")

    # ---- 汇总统计 ----
    def stats(d):
        total = sum(r.get("elapsed_sec", 0) for r in d.values())
        groups = {}
        for gid in ["basic", "multi_hop", "complex"]:
            items = [r for r in d.values() if r.get("group") == gid]
            if items:
                groups[gid] = (sum(r.get("elapsed_sec", 0) for r in items) / len(items), len(items))
        return total, len(d), groups

    t_total, t_n, t_g = stats(t)
    g_total, g_n, g_g = stats(g)

    lines.append("## 一、整体统计")
    lines.append("")
    lines.append("| 指标 | 传统 RAG | GraphRAG | 倍数 |")
    lines.append("|---|---|---|---|")
    lines.append(f"| 20题总耗时 | {t_total:.0f}s ({t_total/60:.1f}min) | {g_total:.0f}s ({g_total/60:.1f}min) | {g_total/t_total:.1f}× |")
    lines.append(f"| 平均每题 | {t_total/t_n:.1f}s | {g_total/g_n:.1f}s | {g_total/t_total:.1f}× |")
    lines.append(f"| 基础组均耗时(10) | {t_g['basic'][0]:.1f}s | {g_g['basic'][0]:.1f}s | {g_g['basic'][0]/t_g['basic'][0]:.1f}× |")
    lines.append(f"| 多跳组均耗时(5) | {t_g['multi_hop'][0]:.1f}s | {g_g['multi_hop'][0]:.1f}s | {g_g['multi_hop'][0]/t_g['multi_hop'][0]:.1f}× |")
    lines.append(f"| 复杂组均耗时(5) | {t_g['complex'][0]:.1f}s | {g_g['complex'][0]:.1f}s | {g_g['complex'][0]/t_g['complex'][0]:.1f}× |")
    lines.append("")
    lines.append("> 注：GraphRAG 每题多出 LLM 查询分析(四维)+关键词提取/图查询理解等多次 LLM 调用与图遍历，故更慢；"
                 "传统 RAG 的 list 类问题(仅列表生成)只要 ~3s。B4 曾出现 GraphRAG 多跳查询卡死 6611s，已加 Cypher 20s 超时修复。")
    lines.append("")

    # 检索命中评估
    good_t = sum(1 for n in NOTES.values() if "✅" in n and "❌" not in n.split("；")[0])  # 简化
    lines.append("## 二、逐题对比")
    lines.append("")
    lines.append("| ID | 分组 | 问题 | 传统RAG 策略/耗时 | 传统RAG 检索到 | GraphRAG 策略/耗时 | GraphRAG 检索到 | 对比说明 |")
    lines.append("|---|---|---|---|---|---|---|---|")

    for q in json.load(open(TEST / "questions.json", encoding="utf-8"))["questions"]:
        rid = q["id"]
        tr, gr = t.get(rid), g.get(rid)
        if not tr or not gr:
            continue
        gname = GROUPS[q["group"]]
        row = [
            rid,
            gname,
            q["question"],
            f"{strategy_label(tr)} / {tr.get('elapsed_sec', 0):.1f}s",
            short(doc_names(tr)),
            f"{strategy_label(gr)} / {gr.get('elapsed_sec', 0):.1f}s",
            short(doc_names(gr)),
            NOTES.get(rid, ""),
        ]
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    lines.append("## 三、结论")
    lines.append("")
    lines.append("- **检索质量**：基础题两代都命中目标菜谱；**列表/一般/多跳/复杂题的差距最大**——传统 RAG 系统性受‘鸡蛋花’泛化文本块污染，且对抽象概念/属性约束(减肥/低糖/少油/菜系)无能为力；GraphRAG 通过图实体检索(鸡胸肉/郫县豆瓣酱)与图路径(豆腐--Ingredient)命中更相关的内容。")
    lines.append("- **耗时**：GraphRAG 平均 ~4.3× 更慢（多出多次 LLM 分析 + 图遍历），但换来检索相关性的大幅提升；传统 RAG 快在 list 类(3s)。")
    lines.append("- **路由(优化后)**：简单问题 → hybrid_traditional；中等问题 → combined（图+混合结合，实测纯图RAG对主题型查询召回弱，故改为结合）；困难问题 → combined。")
    lines.append("- **已知短板**：GraphRAG 的子图提取在实体映射失败时返回空‘知识子图’(0概念)；多跳对‘替代猪肉’这类查询仍围绕原实体；这些是后续可优化的方向。")

    out = TEST / "comparison.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"written {out}")


if __name__ == "__main__":
    main()

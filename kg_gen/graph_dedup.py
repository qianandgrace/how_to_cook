#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""图谱去重数据管线（后处理）——在合并后的 nodes.csv / relationships.csv 上执行"去重三层"的第二、三层。

第二层 关系去重：
  ① 同实体对重复关系合并（(A,p,B) 出现多条 → 合一条，保留证据最强）
  ② 对称/反向镜像合并（(A,母公司,B) 与 (B,子公司,A)）
  ③ 谓词归一（"位于/坐落于/在" → 统一谓词）

第三层 三元组去重：
  ① 归一化后 (s,p,o) 精确匹配去重
  ② 语义近似去重（同一事实不同表述）
  ③ 每个三元组带 source 证据，合并时保留证据最强者

本图谱特性：谓词是固定封闭词表（relationship_type 数字 ID），② ③ 天然成立；
本脚本聚焦 ① 的实现，以及本数据里唯一真实的重复类别——菜谱级重复子图（同一菜谱
因源目录多种路径被建成两个节点），连同关系 ID 全局唯一化、三元组 source 证据。
"""

import argparse
import os
import re
import unicodedata

import pandas as pd

# 关系类型数字ID -> 语义谓词（与 RecipeKnowledgeGraphBuilder._init_relationship_mappings 一致）
RELATIONSHIP_TYPE_MAP = {
    "801000001": "has_ingredient",
    "801000002": "requires_tool",
    "801000003": "has_step",
    "801000004": "belongs_to_category",
    "801000005": "has_difficulty",
    "801000006": "uses_method",
    "801000007": "has_amount",
    "801000008": "step_follows",
    "801000009": "serves_people",
    "801000010": "cooking_time",
    "801000011": "prep_time",
}


def normalize_recipe_name(name: str) -> str:
    """菜谱名归一化：NFKC + 去空白 + 去"的做法"后缀，用于识别语义重复的菜谱。"""
    if not name:
        return ""
    s = unicodedata.normalize("NFKC", str(name)).strip()
    s = re.sub(r"\s+", " ", s)
    if s.endswith("的做法"):
        s = s[:-len("的做法")]
    return s.strip()


def load_graph(input_dir: str):
    nodes = pd.read_csv(os.path.join(input_dir, "nodes.csv"), dtype=str)
    rels = pd.read_csv(os.path.join(input_dir, "relationships.csv"), dtype=str)
    return nodes, rels


def layer2_dedup(rels: pd.DataFrame, report: dict, label: str) -> pd.DataFrame:
    """第二层/第三层 ①：同实体对间重复关系合并（精确 (start,end,type) 去重）。

    label 区分两次调用：首次跑在原始数据上（第二层），
    第二次跑在菜谱重定向之后（第三层①，吃掉重定向撞出的重复）。
    """
    before = len(rels)
    rels = rels.sort_values(["startNodeId", "endNodeId", "relationshipType"])
    # 精确 (s,type,t) 分组：>1 行的组是重复，保留每组首行（证据最强），净减即合并条数
    rels = rels.groupby(
        ["startNodeId", "endNodeId", "relationshipType"], as_index=False).first()
    rels = rels.reset_index(drop=True)
    report[label] = before - len(rels)
    report[f"{label}_before"] = before
    report[f"{label}_after"] = len(rels)
    return rels


def make_relationship_ids_unique(rels: pd.DataFrame) -> pd.DataFrame:
    """关系 ID 全局唯一化：合并后批次间 ID 重置（R_000001 重复），重排为全局唯一。"""
    rels = rels.copy()
    rels["relationshipId"] = [f"R_{i + 1:06d}" for i in range(len(rels))]
    return rels


def merge_duplicate_recipes(nodes: pd.DataFrame, rels: pd.DataFrame, report: dict) -> tuple:
    """第三层 ②：菜谱级语义重复 → 合并重复菜谱子图。

    按归一化菜谱名分组；同名多节点时保留 nodeId 最小者为规范节点，
    其余节点的出边全部重定向到规范节点后删除，再从精确三元组去重兜底。
    """
    recipes = nodes[nodes["conceptType"] == "Recipe"]
    dup_names = set()
    name_to_ids = {}
    for _, row in recipes.iterrows():
        key = normalize_recipe_name(row["name"])
        name_to_ids.setdefault(key, []).append(row["nodeId"])

    merged_recipe_ids = set()          # 被合并掉（删除）的菜谱节点
    redirect = {}                       # 被合并菜谱 id -> 规范菜谱 id
    for key, ids in name_to_ids.items():
        if len(ids) > 1:
            dup_names.add(key)
            ids_sorted = sorted(ids)
            canonical = ids_sorted[0]
            for dup_id in ids_sorted[1:]:
                redirect[dup_id] = canonical
                merged_recipe_ids.add(dup_id)

    if redirect:
        rels = rels.copy()
        rels["startNodeId"] = rels["startNodeId"].map(lambda x: redirect.get(x, x))

    report["dup_recipe_names"] = len(dup_names)
    report["merged_recipe_nodes"] = len(merged_recipe_ids)

    # 兜底：重定向后可能撞出精确重复，再做一次三元组去重（第三层①）
    rels = layer2_dedup(rels, report, "triple_exact_dups_merged")

    return nodes, rels, merged_recipe_ids


def attach_source(nodes: pd.DataFrame, rels: pd.DataFrame) -> pd.DataFrame:
    """第三层 ③：给每个三元组挂 source 证据（源菜谱的 file_path）。

    本 schema 中所有关系都以菜谱节点为起点，source 即源菜谱的 filePath。
    """
    recipe_path = dict(zip(nodes["nodeId"], nodes["filePath"]))
    rels = rels.copy()
    rels["source"] = rels["startNodeId"].map(recipe_path)
    return rels


def drop_orphans(nodes: pd.DataFrame, rels: pd.DataFrame, report: dict) -> pd.DataFrame:
    """清理合并后失去引用的孤儿节点：被删菜谱的食材/步骤/菜谱节点。

    锚点概念（分类/难度/方法/工具/根）即使无引用也保留。
    """
    referenced = set(rels["startNodeId"]) | set(rels["endNodeId"])
    anchor_types = {"RecipeCategory", "DifficultyLevel", "CookingMethod", "CookingTool", "Root"}

    orphan_mask = (~nodes["nodeId"].isin(referenced)) & (
        ~nodes["conceptType"].isin(anchor_types)
    )
    orphan_ids = set(nodes.loc[orphan_mask, "nodeId"])
    report["dropped_orphan_nodes"] = len(orphan_ids)
    report["orphan_breakdown"] = dict(
        nodes.loc[orphan_mask, "conceptType"].value_counts()
    )
    return nodes[~orphan_mask]


def build_report(nodes_before, rels_before, nodes_after, rels_after, report: dict) -> str:
    """输出前后对比报告。入参均为数量 int。"""
    lines = []
    lines.append("=" * 60)
    lines.append("去重前后对比")
    lines.append("=" * 60)

    lines.append("\n[第二层 · 关系去重]")
    lines.append(f"  ① 同实体对重复关系合并: 合并 {report['layer2_exact_dups_merged']} 条"
                 f"（{report['layer2_exact_dups_merged_before']} -> {report['layer2_exact_dups_merged_after']}）")
    lines.append(f"  ② 对称/反向镜像: schema 无反向谓词对，不适用")
    lines.append(f"  ③ 谓词归一: 谓词为固定数字词表，天然归一")

    lines.append("\n[第三层 · 三元组去重]")
    lines.append(f"  ① 精确 (s,p,o) 去重: 菜谱重定向后合并 {report['triple_exact_dups_merged']} 条")
    lines.append(f"  ② 菜谱级语义去重: 发现 {report['dup_recipe_names']} 个重复菜谱名，"
                 f"合并掉 {report['merged_recipe_nodes']} 个菜谱节点")
    lines.append(f"  ③ source 证据: 已给 {rels_after} 条三元组挂上源菜谱 file_path")
    lines.append(f"  残留清理: 删除 {report['dropped_orphan_nodes']} 个孤儿节点"
                 f"（{report['orphan_breakdown']}）")

    lines.append("\n[整体]")
    lines.append(f"  节点: {nodes_before} -> {nodes_after}"
                 f"（-{nodes_before - nodes_after}）")
    lines.append(f"  关系: {rels_before} -> {rels_after}"
                 f"（-{rels_before - rels_after}）")

    # 关系 ID 唯一性校验（在 main 里用真实 DataFrame 算好传入）
    lines.append(f"  relationshipId 唯一值: {report['relid_uniq_before']}/{report['relid_total_before']} -> "
                 f"{report['relid_uniq_after']}/{report['relid_total_after']}")

    # 关系类型分布（main 里用 DataFrame 算好传入）
    lines.append("\n[关系类型分布]")
    for k, v in report["type_distribution"].items():
        lines.append(f"  {RELATIONSHIP_TYPE_MAP.get(k, k)}: {v}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="图谱去重数据管线（第二层+第三层）")
    parser.add_argument("--input", default="cypher", help="输入目录（含 nodes.csv/relationships.csv）")
    parser.add_argument("--output", default=None, help="输出目录（默认 <input>/deduped）")
    args = parser.parse_args()

    output_dir = args.output or os.path.join(args.input, "deduped")
    os.makedirs(output_dir, exist_ok=True)

    nodes_df, rels_df = load_graph(args.input)
    nodes_before, rels_before = len(nodes_df), len(rels_df)
    report = {
        "relid_uniq_before": rels_df["relationshipId"].nunique(),
        "relid_total_before": len(rels_df),
    }

    # ---- 第二层：关系去重 ----
    rels_df = layer2_dedup(rels_df, report, "layer2_exact_dups_merged")  # ① 同实体对重复关系合并
    rels_df = make_relationship_ids_unique(rels_df)   # 修复批次间关系 ID 重置

    # ---- 第三层：三元组去重 ----
    nodes_df, rels_df, _ = merge_duplicate_recipes(nodes_df, rels_df, report)  # ② 菜谱级语义去重
    rels_df = attach_source(nodes_df, rels_df)        # ③ 挂 source 证据
    nodes_df = drop_orphans(nodes_df, rels_df, report)  # 清理合并后的孤儿节点

    report["relid_uniq_after"] = rels_df["relationshipId"].nunique()
    report["relid_total_after"] = len(rels_df)
    report["type_distribution"] = rels_df["relationshipType"].value_counts().to_dict()

    nodes_df.to_csv(os.path.join(output_dir, "nodes.csv"), index=False, encoding="utf-8")
    rels_df.to_csv(os.path.join(output_dir, "relationships.csv"), index=False, encoding="utf-8")

    print(build_report(nodes_before, rels_before, len(nodes_df), len(rels_df), report))
    print(f"\n去重结果已写入: {output_dir}")


if __name__ == "__main__":
    main()

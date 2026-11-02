#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
规则优先的菜谱解析器（零 LLM 成本）。

HowToCook 菜谱是高度模板化文档（# X的做法 / 预估烹饪难度：★★★ / ## 必备原料和工具 /
## 计算 / ## 操作 / ## 附加内容），大量字段可确定性抽取。本解析器把这些字段用规则
抽出来，产出与 LLM 抽取完全一致的 RecipeInfo，供下游知识图谱构建使用。

仅在少量模糊字段（菜系、未命中的食材分类、时间）上保留可选 LLM 补漏。
"""

import re
from typing import Dict, List, Optional

from amount_normalizer import AmountNormalizer


class RuleRecipeParser:
    """基于规则的中文菜谱解析器"""

    # 菜系关键词（按优先级排序）
    CUISINE_KEYWORDS = [
        ("川菜", ["川味", "川菜", "麻辣", "郫县"]),
        ("粤菜", ["粤式", "粤菜", "广式", "广东", "广味", "潮汕"]),
        ("湘菜", ["湘菜", "湘味", "湖南"]),
        ("东北菜", ["东北", "哈尔滨", "铁锅"]),
        ("西北菜", ["西北", "陕西", "西安", "新疆", "清真", "兰州"]),
        ("京菜", ["北京", "京味", "京菜", "京式"]),
        ("沪菜", ["上海", "沪式", "沪菜", "本帮"]),
        ("浙菜", ["浙江", "浙式", "浙菜", "杭州", "宁波", "绍兴"]),
        ("苏菜", ["江苏", "苏式", "苏菜", "淮扬", "南京"]),
        ("闽菜", ["福建", "闽式", "闽菜", "闽南"]),
        ("徽菜", ["安徽", "徽式", "徽菜"]),
        ("鲁菜", ["鲁菜", "山东", "济南"]),
    ]

    # 调料关键词（is_main=False / 归调料）
    CONDIMENT_KEYWORDS = [
        "盐", "酱油", "生抽", "老抽", "醋", "糖", "白糖", "冰糖", "料酒", "蚝油",
        "味精", "鸡精", "花椒", "八角", "桂皮", "香叶", "油", "食用油", "香油",
        "芝麻油", "淀粉", "生粉", "胡椒粉", "辣椒粉", "蒜蓉", "姜", "葱", "蒜",
        "番茄酱", "豆瓣酱", "黄豆酱", "甜面酱", "沙茶酱", "咖喱", "孜然", "五香粉",
        "茄汁", "番茄汁", "甜辣酱", "辣椒酱", "黑胡椒", "白胡椒", "蜂蜜", "炼乳", "芝麻",
    ]

    # 蛋白质关键词
    PROTEIN_KEYWORDS = [
        "肉", "鱼", "虾", "鸡", "鸭", "猪", "牛", "羊", "蛋", "豆腐", "蟹",
        "贝", "鱿鱼", "蛤蜊", "牡蛎", "排骨", "火腿", "腊肉", "香肠",
    ]

    # 淀粉类关键词
    STARCH_KEYWORDS = ["面粉", "淀粉", "米", "米粉", "面条", "面包", "土豆", "红薯", "玉米淀粉", "木薯淀粉", "燕麦"]

    # 蔬菜关键词（兜底）
    VEGETABLE_KEYWORDS = [
        "菜", "椒", "葱", "蒜", "姜", "瓜", "豆", "菇", "笋", "芹", "萝卜",
        "茄", "番茄", "西红柿", "藕", "芦笋", "山药", "菠菜", "白菜", "洋葱",
        "辣椒", "韭菜", "豆芽", "西兰花", "花菜", "香菜", "茼蒿", "蕨菜",
    ]

    def __init__(self, directory_category_mapping: Optional[Dict[str, str]] = None,
                 ingredient_categories: Optional[Dict[str, List[str]]] = None,
                 cooking_methods: Optional[List[str]] = None,
                 cooking_tools: Optional[List[str]] = None):
        """构造规则解析器。

        Args:
            directory_category_mapping: 目录名 -> 菜谱分类，来自 RecipeAgent
            ingredient_categories: 分类 -> 食材关键词，来自 RecipeAgent
            cooking_methods: 烹饪方法关键词，来自 RecipeAgent
            cooking_tools: 工具关键词，来自 RecipeAgent
        """
        # 延迟导入 RecipeInfo，避免循环依赖（RecipeAgent 同时依赖本文件）
        from recipe_ai_agent import RecipeInfo, IngredientInfo, CookingStep

        self._RecipeInfo = RecipeInfo
        self._IngredientInfo = IngredientInfo
        self._CookingStep = CookingStep

        self.directory_category_mapping = directory_category_mapping or {}
        self.ingredient_categories = ingredient_categories or {}
        self.cooking_methods = cooking_methods or []
        self.cooking_tools = cooking_tools or []
        self.amount_normalizer = AmountNormalizer()

        # 把预定义分类映射展平为"关键词 -> 分类"，便于子串命中
        self._ingredient_cat_lookup = []
        for category, keywords in self.ingredient_categories.items():
            for kw in keywords:
                self._ingredient_cat_lookup.append((kw, category))

    # ------------------------------------------------------------------ 主体

    def parse(self, markdown: str, file_path: str = "") -> "RecipeInfo":
        """从菜谱 Markdown 抽取结构化信息（纯规则，零 LLM 调用）。"""
        sections = self._split_sections(markdown)
        description = markdown.split("##")[0] if "##" in markdown else markdown

        info = self._RecipeInfo(
            name=self._parse_name(markdown),
            difficulty=self._parse_difficulty(markdown),
            category=self._parse_category(file_path),
            cuisine_type=self._parse_cuisine_type(description),
            prep_time=self._parse_prep_time(description),
            cook_time=self._parse_cook_time(description),
            servings=self._parse_servings(sections.get("计算", "")),
        )

        info.ingredients = self._parse_ingredients(
            sections.get("必备原料和工具", ""),
            sections.get("计算", ""),
        )
        info.steps = self._parse_steps(sections.get("操作", ""))
        info.tags = self._parse_tags(sections.get("附加内容", ""))

        # name 为空时兜底
        if not info.name:
            info.name = "未知菜谱"
        return info

    # ------------------------------------------------------------- 基础工具

    @staticmethod
    def _split_sections(text: str) -> Dict[str, str]:
        """按 '## 标题' 把 Markdown 拆成 标题 -> 正文 的字典。"""
        sections: Dict[str, str] = {}
        current = None
        buf = []
        for line in text.split("\n"):
            m = re.match(r"^##\s+(.+)$", line.strip())
            if m:
                if current is not None:
                    sections[current] = "\n".join(buf).strip()
                current = m.group(1).strip()
                buf = []
            else:
                if current is not None:
                    buf.append(line)
        if current is not None:
            sections[current] = "\n".join(buf).strip()
        return sections

    @staticmethod
    def _bullet_items(body: str) -> List[str]:
        """提取 '## xxx' 正文中的 '- 项目' / '* 项目' 列表项（去重保序）。"""
        items = []
        seen = set()
        for line in body.split("\n"):
            m = re.match(r"^\s*[-*]\s+(.+)$", line.strip())
            if not m:
                continue
            item = m.group(1).strip()
            # 去掉 markdown 加粗标记
            item = item.replace("**", "").strip()
            if item and item not in seen:
                seen.add(item)
                items.append(item)
        return items

    # ------------------------------------------------------------ 逐字段规则

    @staticmethod
    def _parse_name(markdown: str) -> str:
        m = re.search(r"^#\s+(.+)$", markdown, re.MULTILINE)
        if not m:
            return ""
        name = m.group(1).strip()
        name = re.sub(r"的做法\s*$", "", name).strip()
        return name

    @staticmethod
    def _parse_difficulty(markdown: str) -> int:
        m = re.search(r"预估烹饪难度[：:]\s*(★+)", markdown)
        if m:
            return min(max(len(m.group(1)), 1), 5)
        # 兜底：任意 ★★ 行
        m = re.search(r"(★+)", markdown)
        if m:
            return min(max(len(m.group(1)), 1), 5)
        return 3

    def _parse_category(self, file_path: str) -> str:
        if not file_path:
            return ""
        path_parts = file_path.replace("\\", "/").split("/")
        for part in path_parts:
            if part in self.directory_category_mapping:
                return self.directory_category_mapping[part]
        return ""

    @staticmethod
    def _parse_cuisine_type(description: str) -> str:
        for cuisine, keywords in RuleRecipeParser.CUISINE_KEYWORDS:
            if any(k in description for k in keywords):
                return cuisine
        return ""

    @staticmethod
    def _parse_prep_time(description: str) -> str:
        return ""

    @staticmethod
    def _parse_cook_time(description: str) -> str:
        m = re.search(r"(?:全程大约需要|全程约|大约需要|需要|耗时|约)\s*(\d+(?:-\d+)?)\s*(分钟|小时)", description)
        if m:
            return f"{m.group(1)}{m.group(2)}"
        return ""

    def _parse_servings(self, calc_body: str) -> str:
        m = re.search(r"(\d+)\s*(?:个|位)?\s*人\s*食用", calc_body)
        if m:
            return f"{m.group(1)}人"
        m = re.search(r"按照?\s*(\d+)\s*(盘|人|份|个)", calc_body)
        if m:
            return f"{m.group(1)}{m.group(2)}"
        if "每份" in calc_body or "份量" in calc_body:
            return "每份"
        return ""

    # ------------------------------------------------------------ 食材解析

    def _parse_ingredients(self, tools_body: str, calc_body: str) -> List["IngredientInfo"]:
        names = self._bullet_items(tools_body)

        # 从"计算"节解析 名称 -> (amount, unit)
        calc_map = self._parse_amount_map(calc_body)

        # 计算节里出现但必备节没列的食材也补进去（如"水"、"油"）
        for name in calc_map:
            if name not in names:
                names.append(name)

        ingredients = []
        for name in names:
            category = self._classify_ingredient(name)
            amount, unit = self._lookup_amount(name, calc_map)
            ingredients.append(self._IngredientInfo(
                name=name,
                amount=amount,
                unit=unit,
                category=category,
                is_main=category != "调料",
            ))
        return ingredients

    def _parse_amount_map(self, calc_body: str) -> Dict[str, List[str]]:
        """解析"计算"节的 '食材 用量单位' 行为 {name: [amount, unit]}。"""
        result: Dict[str, List[str]] = {}
        for line in calc_body.split("\n"):
            m = re.match(r"^\s*[-*]\s+(.+)$", line.strip())
            if not m:
                continue
            line = m.group(1).strip().replace("**", "")
            if "=" in line:
                continue  # 换算行（如"1 汤匙 = 15ml"）
            parsed = self._split_amount(line)
            if parsed:
                name, amount, unit = parsed
                if name:
                    result[name] = [amount, unit]
        return result

    @staticmethod
    def _split_amount(line: str) -> Optional[tuple]:
        """把 '梅头猪肉 100g' / '盐 1/4 茶匙' / '生粉 2 1/2 茶匙' / '鸡翅 10 ～ 12 只' / '盐 适量' 拆成 (name, amount, unit)。"""
        # 数字（含混合数"2 1/2"、分数"1/4"、区间"10 ～ 12"）+ 可选单位
        m = re.match(
            r"^(?P<name>.+?)\s+"
            r"(?P<amt>\d+\s+\d+\s*/\s*\d+|\d+\s*/\s*\d+|\d+(?:\.\d+)?(?:\s*[~～\-–]\s*\d+(?:\.\d+)?)?)"
            r"\s*(?P<unit>[一-鿿A-Za-z]+)?$",
            line,
        )
        if m:
            amount = re.sub(r"\s+", " ", m.group("amt")).replace("～", "~").replace("–", "-")
            return m.group("name").strip(), amount, (m.group("unit") or "").strip()

        # 纯文字量（适量/少许 等）
        m = re.match(
            r"^(?P<name>.+?)\s+(?P<amt>适量|少许|酌量|少量|一小勺|一大勺|一茶匙|一汤匙|一把|一小把|几滴|几片|几个|几根|几颗)$",
            line,
        )
        if m:
            return m.group("name").strip(), m.group("amt"), ""
        return None

    def _classify_ingredient(self, name: str) -> str:
        # 预定义词典优先（子串命中）
        for kw, category in self._ingredient_cat_lookup:
            if kw in name:
                return category
        # 启发式兜底（按优先级）
        if any(k in name for k in self.CONDIMENT_KEYWORDS):
            return "调料"
        if any(k in name for k in self.PROTEIN_KEYWORDS):
            return "蛋白质"
        if any(k in name for k in self.STARCH_KEYWORDS):
            return "淀粉类"
        if any(k in name for k in self.VEGETABLE_KEYWORDS):
            return "蔬菜"
        return "其他"

    @staticmethod
    def _lookup_amount(name: str, calc_map: Dict[str, List[str]]) -> tuple:
        """精确名匹配优先，其次"计算节名称是食材名的子串"（如 鸡翅 vs 鸡翅中）。"""
        if name in calc_map:
            return tuple(calc_map[name])
        for calc_name, amount_unit in calc_map.items():
            if calc_name and calc_name in name:
                return tuple(amount_unit)
        return "", ""

    # ------------------------------------------------------------ 步骤解析

    def _parse_steps(self, ops_body: str) -> List["CookingStep"]:
        steps = []
        lines = ops_body.split("\n")
        i = 0
        while i < len(lines):
            m = re.match(r"^\s*(\d+)[.、]\s+(.+)$", lines[i])
            if not m:
                i += 1
                continue
            step_number = int(m.group(1))
            desc = m.group(2).strip()
            i += 1
            # 收集续行（非空、非序号、非子列表）
            while i < len(lines):
                line = lines[i].strip()
                if not line:
                    break
                if re.match(r"^\d+[.、]\s+", line):
                    break
                if re.match(r"^[-*]\s+", line):
                    break
                desc += "\n" + line
                i += 1

            methods = [mth for mth in self.cooking_methods if mth in desc]
            tools = [t for t in self.cooking_tools if t in desc]
            steps.append(self._CookingStep(
                step_number=step_number,
                description=desc,
                methods=methods,
                tools=tools,
                time_estimate=self._parse_step_time(desc),
            ))
        return steps

    @staticmethod
    def _parse_step_time(description: str) -> str:
        m = re.search(r"(\d+(?:-\d+)?)\s*(分钟|秒|小时)", description)
        if m:
            return f"{m.group(1)}{m.group(2)}"
        return ""

    # ------------------------------------------------------------ 标签解析

    @staticmethod
    def _parse_tags(extra_body: str) -> List[str]:
        tags = []
        for item in RuleRecipeParser._bullet_items(extra_body):
            if item.startswith("如果您遵循本指南") or "Pull request" in item or "Issue" in item:
                continue
            tags.append(item)
        return tags

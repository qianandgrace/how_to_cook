# 智厨 ChefBrain —— 基于图增强检索（GraphRAG）的智能烹饪问答系统

> 基于开源项目 HowToCook 的 300+ 份 Markdown 菜谱，构建端到端智能烹饪问答系统。
> 本仓库为**第二阶段：图增强检索（GraphRAG）**，在传统 RAG（`traditional_rag/`）之上引入 Neo4j 知识图谱与图检索，解决传统向量检索"无法跨文档做实体关系推理"的痛点。

## 1. 技术栈

Python · LangChain · Neo4j（图数据库）· Milvus（向量数据库）· BAAI/bge-small-zh-v1.5（512维中文嵌入）· qwen3.7-max（LLM，OpenAI兼容接口）· BM25 · rank-bm25 · Docker

| 基础设施 | 地址 / 配置 |
|---------|-----------|
| Neo4j | `bolt://localhost:7687`，账号 `neo4j/kevin666`（Docker，见 `docker-compose.yml`） |
| Milvus | `localhost:19530`，数据库 `chef`，集合 `cooking_knowledge`（1138 条向量） |
| 嵌入模型 | `BAAI/bge-small-zh-v1.5`，512 维，CPU 可跑 |
| LLM | `qwen3.7-max`，密钥在根目录 `.env`（`QWEN_API_KEY` / `QWEN_BASE_URL`） |

## 2. 目录结构

```
how_to_cook/
├── main.py                    # AdvancedGraphRAGSystem 主类：编排全部模块 + 交互问答
├── config.py                  # GraphRAGConfig：Neo4j + Milvus + 模型 + 检索/生成参数
├── run_demo.py                # 非交互演示脚本（5类代表性问题的路由与回答）
├── rag_modules/               # 核心模块（见 §3）
│   ├── graph_data_preparation.py    # 图数据准备：Neo4j → 菜谱文档/分块
│   ├── milvus_index_construction.py # Milvus 向量化与集合/索引管理
│   ├── graph_indexing.py            # 图索引：实体/关系 → 键值对(K,V)结构
│   ├── hybrid_retrieval.py          # 传统混合检索：实体级+主题级+向量，Round-robin融合
│   ├── graph_rag_retrieval.py       # 真图RAG：查询意图理解、多跳遍历、子图提取、结构推理
│   ├── intelligent_query_router.py  # 智能路由：四维分析 → 传统/图RAG/组合
│   └── generation_integration.py    # 生成：LightRAG 风格自适应回答（含流式+重试）
├── kg_gen/                    # 知识图谱生成器：LLM Agent 把菜谱 Markdown 抽成图数据
├── cypher/                    # 图数据导入脚本（nodes.csv / relationships.csv / neo4j_import.cypher）
├── docker-compose.yml         # Neo4j 一键编排 + 数据导入
├── data/HowToCook/dishes/     # 原始菜谱 Markdown（364 份）
├── traditional_rag/           # 第一阶段：传统 RAG（Milvus 向量 + BM25 + RRF）
└── test/                      # 20 题对比测试集 + 两代方案的结果
```

## 3. 系统架构与模块

一次问答的完整调用链：

```
用户提问
  │
  ▼
① IntelligentQueryRouter.analyze_query        ← LLM 四维分析（复杂度/关系密集度/推理需求/实体数）
  │                                              输出推荐策略：hybrid_traditional / graph_rag / combined
  ▼
② 按策略执行检索
  ├── hybrid_traditional → HybridRetrievalModule.hybrid_search
  │      └─ 双层检索(实体级+主题级，图索引/Neo4j)  +  增强向量检索(Milvus→图邻居)  →  Round-robin 融合
  ├── graph_rag         → GraphRAGRetrieval.graph_rag_search
  │      └─ LLM 理解图意图 → 多跳遍历 / 子图提取 / 路径查找 → 图结构推理 → 路径/子图转文本
  └── combined          → 两路各取一半，Round-robin 交替合并
  │
  ▼
③ GenerationIntegrationModule.generate_adaptive_answer    ← LightRAG 风格提示词，拼接检索上下文
  │
  ▼
回答（支持流式 + 失败重试降级）
```

### 各模块职责

| 模块 | 职责 |
|------|------|
| **graph_data_preparation** | 连接 Neo4j，加载 Recipe/Ingredient/CookingStep 三类节点；用 Cypher 按关系（REQUIRES / CONTAINS_STEP / BELONGS_TO_CATEGORY）把每个菜谱重组成结构化文档（含食材、步骤、菜系、难度、时间），再按章节/长度分块。**每个文档/块都携带 `node_id`** |
| **milvus_index_construction** | BGE 向量化，写入 Milvus `chef.cooking_knowledge`。Schema 含 `node_id`、`recipe_name`、`node_type`、`category`、`cuisine_type`、`difficulty`、`chunk_id`、`parent_id` 等标量字段，HNSW/COSINE 索引，支持按字段过滤检索 |
| **graph_indexing** | 把图节点/关系转成内存键值对（K,V）：实体以"名称"为索引键，关系生成多个索引键（含"食材搭配/制作步骤/菜品分类"等主题键），支撑实体级/主题级检索 |
| **hybrid_retrieval** | 传统混合检索：LLM 把查询拆成**实体级关键词**（具体食材/菜名）与**主题级关键词**（减肥/川菜等抽象概念），分别走图索引 KV + Neo4j 兜底；再叠加**增强向量检索**（Milvus 命中后回图取邻居）；最后 **Round-robin** 轮询去重合并（LightRAG 思路，免调权重） |
| **graph_rag_retrieval** | 真图检索：LLM 把自然语言映射成 `GraphQuery`（五类意图：entity_relation / multi_hop / subgraph / path_finding / clustering），执行 Cypher 多跳遍历 `MATCH path=(source)-[*1..depth]-(target)` 并按"短路径+高度数+关系命中"打分，或提取知识子图做图结构推理，把路径/子图翻译成自然语言 Document |
| **intelligent_query_router** | LLM 四维分析查询特征，路由到传统/图RAG/组合；LLM 失败降级为规则分析；检索异常降级回传统检索 |
| **generation_integration** | LightRAG 风格统一提示词自适应生成（列表/步骤/综合），流式输出带指数退避重试 + 非流式后备 |

## 4. ⭐ Milvus 与图数据（Neo4j）是如何连接的

这是本系统的核心设计。**Milvus（向量）与 Neo4j（图）通过"共享的 `node_id`"双向打通**，构成"图建库、向量检索、图增强、图遍历、统一融合"的闭环：

```
        ┌──────────────────────── 建库方向（离线一次） ────────────────────────┐
        │                                                                      │
   Neo4j 图数据 ──graph_data_preparation──► 菜谱文档/分块 ──milvus_index_construction──► Milvus 向量
   (Recipe/Ingredient/               每个 chunk.metadata                         每条向量带
    CookingStep 节点)                保存 node_id = 图节点 nodeId                node_id 字段
        │                                                                        │
        │  ① node_id 是两张表的"外键"：Milvus 每条向量都能指回 Neo4j 对应节点      │
        │                                                                        │
        └──────────────┬──────────────────────────────┬─────────────────────────┘
                       ▼                              ▼
        ┌──────────────────────────┐   ┌──────────────────────────────────────┐
        │  ② 检索方向 A：向量 → 图    │   │  ③ 检索方向 B：图直接检索               │
        │  vector_search_enhanced   │   │  graph_indexing KV + Neo4j Cypher     │
        │  Milvus similarity_search │   │  · 实体级：按名称查实体 KV，取邻居       │
        │  命中后拿到 node_id        │   │  · 主题级：按关系 KV 索引键查主题         │
        │  → 查 Neo4j 取邻居         │   │  · Neo4j 兜底：fulltext 索引 /         │
        │    (_get_node_neighbors)  │   │    category·cuisineType·tags 过滤      │
        │  → 邻居信息拼进内容         │   │  · 图RAG：多跳遍历/子图提取/路径查找      │
        └───────────┬──────────────┘   └──────────────────┬───────────────────┘
                    └──────────────────┬──────────────────┘
                                       ▼
                    ④ 融合：Round-robin 轮询交替取路结果，按 node_id/内容哈希去重
```

### 具体连接点（对应代码）

1. **建库时，向量"记住"自己的图身份**
   - `graph_data_preparation.py::load_graph_data`：从 Neo4j 读出菜谱/食材/步骤，`build_recipe_documents` 用 Cypher 把每个菜谱的食材（`REQUIRES`）、步骤（`CONTAINS_STEP`）、分类（`BELONGS_TO_CATEGORY`）重组为文档，`metadata["node_id"] = recipe.nodeId`。
   - `milvus_index_construction.py::build_vector_index`：分块后把 `node_id`、`recipe_name`、`category`、`cuisine_type`、`difficulty` 写入 Milvus Schema 字段。**每条向量都指向 Neo4j 的一个 Recipe 节点**。

2. **检索时，向量结果"回图取邻居"**
   - `hybrid_retrieval.py::vector_search_enhanced`：Milvus 搜索命中 → 从结果 metadata 拿 `node_id` → 调 `_get_node_neighbors(node_id)` 用 Cypher `MATCH (n {nodeId})-[r]-(neighbor)` 取该节点在图中的邻居 → 把邻居名追加进检索内容。这样纯向量召回的结果被图关系"扩写"。

3. **图本身也是检索引擎**
   - `graph_indexing.py`：把图节点/关系预编译为内存键值对（实体名→描述；关系→"食材搭配/制作步骤/菜品分类"等主题索引键）。
   - `hybrid_retrieval.py::dual_level_retrieval`：LLM 提取实体级+主题级关键词 → 分别查实体 KV、关系 KV → 不足时走 Neo4j 兜底（`recipe_fulltext_index` 全文检索、`category/cuisineType/tags` 属性过滤）。
   - `graph_rag_retrieval.py`：LLM 把问题映射为图查询（如 `multi_hop`、`subgraph`），执行 `MATCH path=(source)-[*1..max_depth]-(target)` 多跳遍历 / 子图提取，把路径翻译成"节点--关系-->节点"的推理链文本。

4. **两路结果统一融合**
   - `hybrid_retrieval.py::hybrid_search`：把"图信号结果"（dual_level）与"向量结果"（vector_enhanced）用 **Round-robin** 交替各取一条、按 `node_id`/内容去重合并，保证每路都贡献靠前的结果，且无需调权重。
   - `intelligent_query_router.py`：在更高一层决定走哪条检索（含 combined 两路各半再轮询），并在结果上打 `route_strategy` 元数据。

**一句话总结**：Milvus 负责"语义相似"召回，Neo4j 负责"结构关系"推理；两者靠每条向量的 `node_id` 字段互相引用——建库时从图生成带图身份的向量，检索时向量命中再回图取邻居做增强，图本身又能直接做实体/主题/多跳/子图检索，最后由 Round-robin 融合成统一上下文送进 LLM 生成。

## 5. 与传统 RAG（traditional_rag/）的关键差异

| 维度 | 传统 RAG | GraphRAG（本仓库） |
|------|---------|-------------------|
| 知识组织 | 纯文本块（2012 chunks） | 实体-关系知识图谱（323 菜谱 / 2906 食材 / 2514 步骤） |
| 向量库 | `chef.traditional_rag`（按菜谱 md 分块） | `chef.cooking_knowledge`（按图重建文档分块，带图字段） |
| 检索信号 | 向量 + BM25 | 向量 + BM25 + 实体级图检索 + 主题级图检索 + 图键值对 + 多跳/子图 |
| 融合策略 | RRF（倒数排名融合） | Round-robin 轮询（LightRAG 思路，免调权重） |
| 跨文档关系 | 无能为力（"鸡肉配什么蔬菜"答不上） | 图多跳/子图可答 |
| 属性约束 | 仅分类/难度过滤 | 属性级约束（低糖/30分钟内）+ 图结构过滤 |
| 查询路由 | list/detail/general 三类 | 四维分析 → 传统/图/组合，多级降级 |

## 6. 运行方式

```bash
# 1. conda 环境（已装好 langchain/pymilvus/neo4j/sentence-transformers 等）
conda activate chef

# 2. 启动依赖服务（Neo4j + Milvus）
docker compose up -d

# 3. 交互式问答（首次若集合不存在会自动重建 Milvus 索引）
python main.py
#    交互内可用命令：'stats' 看路由统计 / 'rebuild' 重建知识库 / 'quit' 退出

# 4. 或跑非交互演示（5 类代表性问题的路由与回答）
python run_demo.py
```

**数据准备前置**（本次已就绪，重建需重新执行）：
1. `kg_gen/`：用 LLM Agent 把 `data/HowToCook/dishes/*.md` 抽成 `cypher/nodes.csv`、`relationships.csv`（见 `kg_gen/README.md`）。
2. `docker compose up -d`：`neo4j-init` 服务自动执行 `cypher/neo4j_import.cypher` 导入图数据。
3. 首次运行 `python main.py`：`build_knowledge_base` 从 Neo4j 读图 → 建 323 篇菜谱文档 → 分 1138 块 → 写入 Milvus `cooking_knowledge`。

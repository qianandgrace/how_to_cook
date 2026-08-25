# 智厨 ChefBrain —— 传统 RAG（混合检索）流程与知识点总结

> 本项目为智厨 ChefBrain 的**第一阶段：传统 RAG** 实现，用于与第二阶段的 GraphRAG（Neo4j + Milvus 图增强检索）做对比。原始代码基于 FAISS，本版已升级为 **Milvus 向量数据库**（`db=chef`，`collection=traditional_rag`）。

---

## 1. 项目概览

- **目标**：基于 364 份 Markdown 菜谱，构建一个"检索增强生成（RAG）"的智能烹饪问答系统。
- **回答能力**：查询路由（列表 / 详细 / 一般三类）+ 混合检索 + 大模型生成，支持分步菜谱指导、菜品推荐、食材询问。
- **与 GraphRAG 的关系**：这是纯向量检索版（不建图），GraphRAG 在其上增加了知识图谱（Neo4j）与图检索，两者共享同一份菜谱数据与 Milvus 向量库，便于公平对比。

## 2. 技术栈与环境

| 类别 | 选型 |
|------|------|
| 语言 | Python 3.11（conda 环境 `chef`） |
| 嵌入模型 | `BAAI/bge-small-zh-v1.5`（512 维，CPU 可跑） |
| 向量数据库 | Milvus（`db_name=chef`，`collection_name=traditional_rag`，COSINE 相似度，HNSW 索引） |
| 大模型 | `qwen3.7-max`（阿里云 MaaS 兼容 OpenAI 接口） |
| 检索库 | LangChain / langchain-community / rank-bm25 |
| 关键依赖 | `langchain>=1.3.4`、`pymilvus>=3.0.0`、`sentence-transformers>=5.5.1`、`rank-bm25`、`python-dotenv` |

- Milvus 以 Docker 运行：`milvus-standalone`（端口 19530）、`milvus-minio`、`milvus-etcd`。
- API 密钥配置在仓库根目录 `.env`：`QWEN_API_KEY`、`QWEN_BASE_URL`。

## 3. 数据说明

- 原始数据：`data/HowToCook/dishes/`，共 **364** 份 `.md` 菜谱，按分类目录组织：
  - 目录（映射中文分类）：`meat_dish→荤菜`、`vegetable_dish→素菜`、`soup→汤品`、`dessert→甜品`、`breakfast→早餐`、`staple→主食`、`aquatic→水产`、`condiment→调料`、`drink→饮品`、`semi-finished`（归为"其他"）。
- 菜谱是统一的 Markdown 结构（约 700 字/篇）：
  - `# 菜品名称的做法`：主标题
  - `## 必备原料和工具` / `## 计算` / `## 操作` / `## 附加内容`：二级标题（信息分节）
  - `### 简易版本 / 复杂版本`：三级标题（版本细分）
  - 难度用星级 `★★★★` 标注，映射为"非常简单 / 简单 / 中等 / 困难 / 非常困难"。

## 4. 系统架构与模块

```
traditional_rag/
├── config.py                 # RAGConfig：数据路径 + Milvus + 模型 + 检索/生成参数
├── main.py                   # RecipeRAGSystem 主类：编排完整流程 + 交互式问答
├── run_demo.py               # 非交互演示脚本（初始化→建库→3个示例问题）
├── rag_modules/
│   ├── data_preparation.py       # 数据加载、元数据增强、Markdown 结构感知分块
│   ├── index_construction.py     # Milvus 向量化 + 集合/索引管理 + 相似度检索
│   ├── retrieval_optimization.py # 混合检索（向量+BM25）+ RRF 重排 + 元数据过滤
│   └── generation_integration.py # LLM 集成：查询路由/重写 + 三种回答模式 + 流式
└── pyproject.toml
```

由 `RecipeRAGSystem` 主类编排四个模块，形成完整闭环。

## 5. 完整流程（一次问答的调用链）

```
用户提问
  │
  ▼
① 查询路由 query_router()          ← LLM 判断类型：list / detail / general
  │
  ▼
② 查询重写 query_rewrite()         ← LLM 对模糊查询做智能改写（明确查询保持原样）
  │
  ▼
③ 元数据过滤提取                    ← 从问题中识别分类/难度关键词（如"素菜""简单"）
  │
  ▼
④ 混合检索 hybrid_search()
  │   ├── 向量检索 similarity_search()   ← Milvus（BGE 512维，COSINE）
  │   └── BM25 关键词检索 BM25Retriever  ← rank-bm25
  │   └── RRF 倒数排名融合重排           ← 1/(k+rank)，k=60
  │
  ▼
⑤ 父文档回传 get_parent_documents()  ← 依据 parent_id 取回完整菜谱（"小块检索、大块生成"）
  │
  ▼
⑥ 答案生成                           ← 按路由类型：
  │      list   → generate_list_answer      （简洁菜品列表）
  │      detail → generate_step_by_step_answer（分步骤菜谱指导，可流式）
  │      general→ generate_basic_answer       （基础回答，可流式）
  │
  ▼
回答输出
```

### 知识库构建流程（离线一次）

```
读取 data/HowToCook/dishes/**/*.md（364份）
  → 元数据增强（category / dish_name / difficulty）
  → Markdown 结构感知分块（按 # / ## / ### 切，生成 2012 个 chunk）
  → BGE 向量化（512维，normalize）
  → 批量插入 Milvus（batch=100，共2012条）
  → 创建 HNSW/COSINE 索引 → load_collection 载入内存
```

二次启动时 `load_index()` 检测到 Milvus 集合已存在则**直接加载跳过重建**（启动秒级），相当于把原来 FAISS 的"落盘缓存"搬到了 Milvus 服务端。

## 6. 核心知识点详解

### 6.1 父子文本块（Parent-Child Chunking）
- **问题**：块太大检索不精确，块太小生成上下文不完整。
- **方案**：检索用"小块"（按 Markdown 标题切出的子块，主题聚焦、检索精确）；生成时用 `parent_id` 回传"大块"（完整父文档，保证上下文连贯）。即**"小块检索、大块生成"**。
- 每个子块记录 `chunk_id`、`parent_id`、`doc_type=child`、`chunk_index`。

### 6.2 元数据增强
- 从文件路径推断**分类**（`meat_dish/` → 荤菜）。
- 从内容里的 `★` 数量解析**难度**（5★→非常困难 … 1★→非常简单）。
- 这些字段同时写入 Milvus 标量字段，支持检索时的标量过滤。

### 6.3 混合检索 + RRF 重排
- **向量检索**（Milvus）：语义相似度，能处理"同义/近义"表达。
- **BM25 检索**（rank-bm25）：关键词精确匹配，能命中菜名等专有名词。
- **RRF（Reciprocal Rank Fusion）融合**：`score = Σ 1/(k + rank)`，k=60，把两路结果的排名位置折算成分数相加排序，**无需调权重**，实现简单、鲁棒。
- 此外支持按分类/难度做**元数据过滤**（先混合检索取多路候选，再在内存中按 metadata 过滤）。

### 6.4 查询路由与重写
- **查询路由**：LLM 把问题分为三类——
  - `list`：要菜品列表/推荐（如"推荐几个素菜"）→ 只回菜名；
  - `detail`：要制作方法/步骤/食材 → 分步指导；
  - `general`：其他综合咨询 → 基础回答。
- **查询重写**：LLM 判断模糊查询（如"做菜"→"简单易做的家常菜谱"）并改写为更利于检索的表达；明确查询（含具体菜名）保持原样，避免画蛇添足。

### 6.5 Milvus 向量数据库（本次改造核心）
- 用 pymilvus 3.x 的 `MilvusClient` 直连，替代原 FAISS 落盘。
- **Schema**：主键 `id`(=chunk_id)、`vector`(512维)、`text`、`dish_name`、`category`、`difficulty`、`doc_type`、`chunk_id`、`parent_id`。保留元数据字段以便标量过滤与父子回传。
- **索引**：HNSW（M=16，efConstruction=200）+ COSINE 度量；插入后 `create_index` + `load_collection`。
- **持久化**：集合存在 Milvus 服务端，天然多客户端共享；`has_collection` + `load_collection` 实现秒级重启。
- 注意：pymilvus 3.x 的 `get_collection_stats` 在未加载时 `row_count` 可能显示 0，用 count 查询（`query(filter='id != ""', output_fields=['count(*)'])`）校准。

### 6.6 生成集成（LLM）
- 用 LangChain `ChatOpenAI` 接阿里云 MaaS 的 OpenAI 兼容接口（`QWEN_API_KEY`/`QWEN_BASE_URL`）。
- 按路由类型拼不同提示词（分步指导 / 基础回答 / 列表），上下文用 `_build_context` 拼接元数据 + 正文，限长 2000 字。
- 支持**流式输出**（`chain.stream`），交互体验更好。

## 7. 运行方式

```bash
# 1. 进入 conda 环境（依赖已装好）
conda activate chef

# 2. 启动 Milvus（如未启动）
docker compose up -d   # 或 docker start milvus-standalone

# 3a. 交互式问答（首次会自动构建 Milvus 索引，约 1~2 分钟）
cd traditional_rag
python main.py

# 3b. 或运行非交互演示（3 个示例问题）
cd traditional_rag
python run_demo.py
```

首次运行会创建 `chef.traditional_rag` 集合并写入 2012 条向量；之后启动自动从 Milvus 加载，秒级就绪。

## 8. 与 GraphRAG（第二阶段）的对比要点

| 维度 | 传统 RAG（本版） | GraphRAG（根目录 `rag_modules/` + `kg_gen/`） |
|------|------------------|-----------------------------------------------|
| 知识组织 | 纯文本块（chunk） | 实体-关系知识图谱（Neo4j） |
| 向量库 | Milvus `chef.traditional_rag` | Milvus `chef.cooking_knowledge` |
| 检索信号 | 向量 + BM25 | 向量 + BM25 + 实体级图检索 + 主题级图检索 + 图键值对 |
| 融合策略 | RRF（倒数排名融合） | Round-robin 轮询（LightRAG 思路，免调权重） |
| 复杂关系查询 | 无法跨文档关联（如"鸡肉配什么蔬菜好"） | 图多跳遍历、子图提取、路径查找可回答 |
| 查询路由 | 三类（list/detail/general） | 四维分析（复杂度/关系密集度/推理需求/实体数），路由到传统/图/组合 |
| 降级策略 | 无显式降级 | 多级降级（图失败→组合→传统保底） |

**核心差异**：传统 RAG 召回的是"语义相似的文本块"，对跨文档的隐含实体关系无能为力；GraphRAG 把实体关系显式建模成图，用结构化的多跳推理回答"鸡肉配什么蔬菜好"这类关联问题。

## 9. 遇到的问题与解决

1. **Windows 控制台 GBK 编码无法输出 emoji**（`UnicodeEncodeError`）
   - 在 `main.py` 顶部对 `sys.stdout/stderr` 执行 `reconfigure(encoding='utf-8')`。
2. **PyPI 网络中断导致安装失败**
   - 换清华镜像 `pip install -i https://pypi.tuna.tsinghua.edu.cn/simple ...` 重试成功。
3. **pymilvus 3.x `get_collection_stats` 行数不准**
   - 用 `count(*)` 查询校准（实测 2012 条）。
4. **chef 环境为空**：从零安装了 langchain 全家桶 + pymilvus + sentence-transformers + torch（CPU）。

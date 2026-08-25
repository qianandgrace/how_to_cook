"""
索引构建模块 - 基于 Milvus 向量数据库
"""

import logging
import time
from typing import List, Dict, Any, Optional

from pymilvus import MilvusClient, DataType, CollectionSchema, FieldSchema
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class IndexConstructionModule:
    """索引构建模块 - 负责向量化和 Milvus 索引构建"""

    def __init__(self,
                 model_name: str = "BAAI/bge-small-zh-v1.5",
                 host: str = "localhost",
                 port: int = 19530,
                 database: str = "chef",
                 collection_name: str = "traditional_rag",
                 dimension: int = 512,
                 index_save_path: str = "./vector_index"):
        """
        初始化索引构建模块

        Args:
            model_name: 嵌入模型名称
            host: Milvus服务器地址
            port: Milvus服务器端口
            database: Milvus数据库名称
            collection_name: 集合名称
            dimension: 向量维度
            index_save_path: 保留参数（Milvus无需落盘，仅兼容旧接口）
        """
        self.model_name = model_name
        self.host = host
        self.port = port
        self.database = database
        self.collection_name = collection_name
        self.dimension = dimension
        self.index_save_path = index_save_path

        self.client = None
        self.embeddings = None
        self.collection_created = False

        self._setup_client()
        self._setup_embeddings()

    def _safe_truncate(self, text: Optional[str], max_length: int) -> str:
        """安全截取字符串，处理None值"""
        if text is None:
            return ""
        return str(text)[:max_length]

    def _ensure_database(self):
        """确保Milvus数据库存在（不存在则创建）"""
        admin = MilvusClient(uri=f"http://{self.host}:{self.port}")
        try:
            if self.database not in admin.list_databases():
                admin.create_database(self.database)
                logger.info(f"已创建Milvus数据库: {self.database}")
            else:
                logger.info(f"Milvus数据库已存在: {self.database}")
        finally:
            admin.close()

    def _setup_client(self):
        """初始化Milvus客户端"""
        try:
            self._ensure_database()
            self.client = MilvusClient(
                uri=f"http://{self.host}:{self.port}",
                db_name=self.database
            )
            logger.info(f"已连接到Milvus: {self.host}:{self.port}, 数据库: {self.database}")
        except Exception as e:
            logger.error(f"连接Milvus失败: {e}")
            raise

    def _setup_embeddings(self):
        """初始化嵌入模型"""
        logger.info(f"正在初始化嵌入模型: {self.model_name}")

        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        logger.info("嵌入模型初始化完成")

    def _create_collection_schema(self) -> CollectionSchema:
        """
        创建集合模式（保留传统RAG的元数据字段，支持分类/难度过滤）

        Returns:
            集合模式对象
        """
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=150, is_primary=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.dimension),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=15000),
            FieldSchema(name="dish_name", dtype=DataType.VARCHAR, max_length=300),
            FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="difficulty", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="doc_type", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=150),
            FieldSchema(name="parent_id", dtype=DataType.VARCHAR, max_length=100)
        ]

        schema = CollectionSchema(
            fields=fields,
            description="传统RAG菜谱知识向量集合"
        )

        return schema

    def create_collection(self, force_recreate: bool = False) -> bool:
        """
        创建Milvus集合

        Args:
            force_recreate: 是否强制重新创建集合

        Returns:
            是否创建成功
        """
        try:
            if self.client.has_collection(self.collection_name):
                if force_recreate:
                    logger.info(f"删除已存在的集合: {self.collection_name}")
                    self.client.drop_collection(self.collection_name)
                else:
                    logger.info(f"集合 {self.collection_name} 已存在")
                    self.collection_created = True
                    return True

            schema = self._create_collection_schema()

            self.client.create_collection(
                collection_name=self.collection_name,
                schema=schema,
                metric_type="COSINE",
                consistency_level="Strong"
            )

            logger.info(f"成功创建集合: {self.collection_name}")
            self.collection_created = True
            return True

        except Exception as e:
            logger.error(f"创建集合失败: {e}")
            return False

    def create_index(self) -> bool:
        """创建向量索引"""
        try:
            if not self.collection_created:
                raise ValueError("请先创建集合")

            index_params = self.client.prepare_index_params()
            index_params.add_index(
                field_name="vector",
                index_type="HNSW",
                metric_type="COSINE",
                params={"M": 16, "efConstruction": 200}
            )

            self.client.create_index(
                collection_name=self.collection_name,
                index_params=index_params
            )

            logger.info("向量索引创建成功")
            return True

        except Exception as e:
            logger.error(f"创建索引失败: {e}")
            return False

    def build_vector_index(self, chunks: List[Document]):
        """
        构建向量索引

        Args:
            chunks: 文档块列表

        Returns:
            self（索引模块实例，兼容旧接口的调用方式）
        """
        logger.info(f"正在构建Milvus向量索引，文档数量: {len(chunks)}...")

        if not chunks:
            raise ValueError("文档块列表不能为空")

        # 1. 创建集合（schema不兼容则强制重建）
        if not self.create_collection(force_recreate=True):
            raise RuntimeError("创建集合失败")

        # 2. 生成向量
        logger.info("正在生成向量embeddings...")
        texts = [chunk.page_content for chunk in chunks]
        vectors = self.embeddings.embed_documents(texts)

        # 3. 准备插入数据
        entities = []
        for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
            chunk_id = self._safe_truncate(chunk.metadata.get("chunk_id", f"chunk_{i}"), 150)
            entity = {
                "id": chunk_id,
                "vector": vector,
                "text": self._safe_truncate(chunk.page_content, 15000),
                "dish_name": self._safe_truncate(chunk.metadata.get("dish_name", ""), 300),
                "category": self._safe_truncate(chunk.metadata.get("category", ""), 100),
                "difficulty": self._safe_truncate(chunk.metadata.get("difficulty", ""), 100),
                "doc_type": self._safe_truncate(chunk.metadata.get("doc_type", ""), 50),
                "chunk_id": chunk_id,
                "parent_id": self._safe_truncate(chunk.metadata.get("parent_id", ""), 100)
            }
            entities.append(entity)

        # 4. 批量插入数据
        logger.info("正在插入向量数据...")
        batch_size = 100
        for i in range(0, len(entities), batch_size):
            batch = entities[i:i + batch_size]
            self.client.insert(collection_name=self.collection_name, data=batch)
            logger.info(f"已插入 {min(i + batch_size, len(entities))}/{len(entities)} 条数据")

        # 5. 创建索引
        if not self.create_index():
            raise RuntimeError("创建索引失败")

        # 6. 加载集合到内存
        self.client.load_collection(self.collection_name)
        logger.info("集合已加载到内存")
        time.sleep(2)

        logger.info(f"向量索引构建完成，包含 {len(chunks)} 个向量")
        return self

    def add_documents(self, new_chunks: List[Document]) -> bool:
        """向现有集合添加新文档"""
        if not self.collection_created:
            raise ValueError("请先构建向量索引")

        logger.info(f"正在添加 {len(new_chunks)} 个新文档到索引...")

        texts = [chunk.page_content for chunk in new_chunks]
        vectors = self.embeddings.embed_documents(texts)

        entities = []
        for i, (chunk, vector) in enumerate(zip(new_chunks, vectors)):
            chunk_id = self._safe_truncate(chunk.metadata.get("chunk_id", f"new_chunk_{i}_{int(time.time())}"), 150)
            entity = {
                "id": chunk_id,
                "vector": vector,
                "text": self._safe_truncate(chunk.page_content, 15000),
                "dish_name": self._safe_truncate(chunk.metadata.get("dish_name", ""), 300),
                "category": self._safe_truncate(chunk.metadata.get("category", ""), 100),
                "difficulty": self._safe_truncate(chunk.metadata.get("difficulty", ""), 100),
                "doc_type": self._safe_truncate(chunk.metadata.get("doc_type", ""), 50),
                "chunk_id": chunk_id,
                "parent_id": self._safe_truncate(chunk.metadata.get("parent_id", ""), 100)
            }
            entities.append(entity)

        self.client.insert(collection_name=self.collection_name, data=entities)
        logger.info("新文档添加完成")
        return True

    def save_index(self):
        """Milvus集合本身已持久化，无需额外落盘（兼容旧接口）"""
        logger.info(f"Milvus集合 {self.collection_name} 已持久化在服务器上，无需本地保存")

    def load_index(self):
        """
        从Milvus加载向量索引

        Returns:
            索引模块实例，如果集合不存在返回None
        """
        if not self.embeddings:
            self._setup_embeddings()

        try:
            if self.client.has_collection(self.collection_name):
                self.client.load_collection(self.collection_name)
                self.collection_created = True
                logger.info(f"集合 {self.collection_name} 已从Milvus加载")
                return self
            else:
                logger.info(f"集合 {self.collection_name} 不存在，将构建新索引")
                return None
        except Exception as e:
            logger.warning(f"加载向量索引失败: {e}，将构建新索引")
            return None

    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """
        相似度搜索

        Args:
            query: 查询文本
            k: 返回结果数量

        Returns:
            Document文档列表
        """
        if not self.collection_created:
            raise ValueError("请先构建或加载向量索引")

        query_vector = self.embeddings.embed_query(query)

        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_vector],
            anns_field="vector",
            limit=k,
            output_fields=["text", "dish_name", "category", "difficulty",
                           "doc_type", "chunk_id", "parent_id"],
            search_params={"metric_type": "COSINE", "params": {"ef": 64}}
        )

        documents = []
        if results and len(results) > 0:
            for hit in results[0]:
                entity = hit["entity"]
                metadata = {
                    "dish_name": entity.get("dish_name", ""),
                    "category": entity.get("category", ""),
                    "difficulty": entity.get("difficulty", ""),
                    "doc_type": entity.get("doc_type", ""),
                    "chunk_id": entity.get("chunk_id", ""),
                    "parent_id": entity.get("parent_id", ""),
                    "score": hit["distance"]
                }
                documents.append(Document(
                    page_content=entity.get("text", ""),
                    metadata=metadata
                ))

        return documents

    def get_collection_stats(self) -> Dict[str, Any]:
        """获取集合统计信息"""
        try:
            if not self.collection_created:
                return {"error": "集合未创建"}

            stats = self.client.get_collection_stats(self.collection_name)
            return {
                "collection_name": self.collection_name,
                "row_count": stats.get("row_count", 0),
                "index_building_progress": stats.get("index_building_progress", 0)
            }
        except Exception as e:
            logger.error(f"获取集合统计信息失败: {e}")
            return {"error": str(e)}

    def has_collection(self) -> bool:
        """检查集合是否存在"""
        try:
            return self.client.has_collection(self.collection_name)
        except Exception as e:
            logger.error(f"检查集合存在性失败: {e}")
            return False

    def load_collection(self) -> bool:
        """加载集合到内存"""
        try:
            if not self.client.has_collection(self.collection_name):
                logger.error(f"集合 {self.collection_name} 不存在")
                return False

            self.client.load_collection(self.collection_name)
            self.collection_created = True
            logger.info(f"集合 {self.collection_name} 已加载到内存")
            return True
        except Exception as e:
            logger.error(f"加载集合失败: {e}")
            return False

    def delete_collection(self) -> bool:
        """删除集合"""
        try:
            if self.client.has_collection(self.collection_name):
                self.client.drop_collection(self.collection_name)
                logger.info(f"集合 {self.collection_name} 已删除")
                self.collection_created = False
                return True
            else:
                logger.info(f"集合 {self.collection_name} 不存在")
                return True
        except Exception as e:
            logger.error(f"删除集合失败: {e}")
            return False

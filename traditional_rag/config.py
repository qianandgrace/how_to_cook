"""
RAG系统配置文件
"""

from dataclasses import dataclass, field
from typing import Dict, Any
from pathlib import Path

# 数据路径：traditional_rag 的上级目录为仓库根目录
_DATA_ROOT = Path(__file__).resolve().parent.parent / "data" / "HowToCook" / "dishes"

@dataclass
class RAGConfig:
    """RAG系统配置类"""

    # 路径配置
    data_path: str = str(_DATA_ROOT)

    # Milvus 向量数据库配置
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_database: str = "chef"
    milvus_collection_name: str = "traditional_rag"
    milvus_dimension: int = 512  # BGE-small-zh-v1.5 的向量维度

    # 模型配置
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    llm_model: str = "qwen3.7-max"

    # 检索配置
    top_k: int = 3

    # 生成配置
    temperature: float = 0.1
    max_tokens: int = 2048

    def __post_init__(self):
        """初始化后的处理"""
        pass

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'RAGConfig':
        """从字典创建配置对象"""
        return cls(**config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'data_path': self.data_path,
            'milvus_host': self.milvus_host,
            'milvus_port': self.milvus_port,
            'milvus_database': self.milvus_database,
            'milvus_collection_name': self.milvus_collection_name,
            'milvus_dimension': self.milvus_dimension,
            'embedding_model': self.embedding_model,
            'llm_model': self.llm_model,
            'top_k': self.top_k,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens
        }

# 默认配置实例
DEFAULT_CONFIG = RAGConfig()

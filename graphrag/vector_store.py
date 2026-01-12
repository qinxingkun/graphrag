import json
from typing import List, Dict, Tuple
from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility,
)
from langchain_community.embeddings import ZhipuAIEmbeddings
import numpy as np
from graphrag.config import MilvusConfig, LLMConfig


class MilvusVectorStore:
    def __init__(self, config: MilvusConfig, llm_config: LLMConfig):
        self.config = config
        self.embeddings = ZhipuAIEmbeddings(
            api_key=llm_config.api_key, model=llm_config.embedding_model
        )
        self.collection = None
        self._initialize()

    def _initialize(self):
        try:
            # 连接到 Milvus
            connections.connect(
                alias="default", host=self.config.host, port=self.config.port
            )
            print("✅ Milvus 连接成功")

            # 创建或加载集合
            if utility.has_collection(self.config.collection_name):
                self.collection = Collection(self.config.collection_name)
                self.collection.load()
                print(f"✅ 加载已存在的集合: {self.config.collection_name}")
            else:
                self._create_collection()
                print(f"✅ 创建新集合: {self.config.collection_name}")

        except Exception as e:
            raise ConnectionError(f"❌ Milvus 初始化失败: {str(e)}")

    def _create_collection(self):
        """创建集合和索引"""
        # 定义字段
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(
                name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.config.dimension
            ),
            FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=65535),
        ]

        # 创建集合
        schema = CollectionSchema(
            fields=fields, description="Knowledge base embeddings"
        )
        self.collection = Collection(name=self.config.collection_name, schema=schema)

        # 创建索引
        index_params = {
            "index_type": self.config.index_type,
            "metric_type": self.config.metric_type,
            "params": {"nlist": 128},
        }
        self.collection.create_index(field_name="embedding", index_params=index_params)

        # 加载集合到内存
        self.collection.load()

    def add_texts(self, texts: List[str], metadatas: List[Dict] = None) -> List[int]:
        """批量添加文本"""
        if not texts:
            return []

        try:
            # 生成 embeddings
            embeddings = self.embeddings.embed_documents(texts)

            # 准备数据
            if metadatas is None:
                metadatas = [{}] * len(texts)

            metadata_strs = [json.dumps(m, ensure_ascii=False) for m in metadatas]

            # 插入数据
            entities = [texts, embeddings, metadata_strs]

            insert_result = self.collection.insert(entities)
            self.collection.flush()

            print(f"✅ 成功添加 {len(texts)} 条记录到 Milvus")
            return insert_result.primary_keys

        except Exception as e:
            print(f"❌ 添加文本失败: {str(e)}")
            return []

    def similarity_search(
        self, query: str, top_k: int = 5, score_threshold: float = 0.7
    ) -> List[Tuple[str, float, Dict]]:
        """相似度搜索"""
        try:
            # 生成查询向量
            query_embedding = self.embeddings.embed_query(query)

            # 搜索参数
            search_params = {
                "metric_type": self.config.metric_type,
                "params": {"nprobe": 10},
            }

            # 执行搜索
            results = self.collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                output_fields=["text", "metadata"],
            )

            # 格式化结果
            formatted_results = []
            for hits in results:
                for hit in hits:
                    # Milvus 的距离分数（距离越小越相似）
                    # 转换为相似度分数（0-1，越大越相似）
                    if self.config.metric_type == "COSINE":
                        similarity = (2 - hit.distance) / 2  # 余弦距离转相似度
                    else:
                        similarity = 1 / (1 + hit.distance)  # L2距离转相似度

                    # 应用阈值过滤
                    if similarity >= score_threshold:
                        metadata = json.loads(hit.entity.get("metadata", "{}"))
                        formatted_results.append(
                            (hit.entity.get("text"), similarity, metadata)
                        )

            return formatted_results

        except Exception as e:
            print(f"❌ 向量搜索失败: {str(e)}")
            return []

    def delete_by_ids(self, ids: List[int]):
        """删除指定ID的记录"""
        if ids:
            expr = f"id in {ids}"
            self.collection.delete(expr)
            self.collection.flush()

    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            "total_count": self.collection.num_entities,
            "collection_name": self.config.collection_name,
            "dimension": self.config.dimension,
        }

    def clear(self):
        """清空集合"""
        self.collection.drop()
        self._create_collection()

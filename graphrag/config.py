from dataclasses import dataclass
from typing import Optional
import os


@dataclass
class Neo4jConfig:
    url: str = "bolt://localhost:7687"
    username: str = "neo4j"
    password: str = "graphrag"
    database: Optional[str] = None


@dataclass
class PostgreSQLConfig:
    host: str = "localhost"
    port: int = 5432
    database: str = "neo4j_agent"
    username: str = "postgres"
    password: str = "postgres"

    @property
    def connection_string(self) -> str:
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"


@dataclass
class MilvusConfig:
    host: str = "localhost"
    port: int = 19530
    collection_name: str = "knowledge_base"
    dimension: int = 1536
    index_type: str = "IVF_FLAT"
    metric_type: str = "COSINE"


@dataclass
class LLMConfig:
    api_key: str = "bab8b02026ff3fc200d405feec6b8225.KmkJqlTfGEjFznV7"
    model: str = "glm-4-0520"
    temperature: float = 0
    embedding_model: str = "embedding-3"


@dataclass
class AppConfig:
    neo4j: Neo4jConfig
    postgresql: PostgreSQLConfig
    milvus: MilvusConfig
    llm: LLMConfig
    verbose: bool = True
    max_iterations: int = 10
    enable_memory: bool = True
    enable_embedding: bool = True


DEFAULT_CONFIG = AppConfig(
    neo4j=Neo4jConfig(),
    postgresql=PostgreSQLConfig(),
    milvus=MilvusConfig(),
    llm=LLMConfig(),
)

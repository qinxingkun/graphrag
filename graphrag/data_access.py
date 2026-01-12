from typing import List, Dict, Optional
from langchain_neo4j import Neo4jGraph
from abc import ABC, abstractmethod
import psycopg2
from psycopg2.extras import RealDictCursor
from graphrag.config import Neo4jConfig, PostgreSQLConfig
import json


class IGraphDatabase(ABC):
    @abstractmethod
    def query(self, cypher: str, params: Dict = None) -> List[Dict]:
        pass

    @abstractmethod
    def get_schema(self) -> str:
        pass


class Neo4jRepository(IGraphDatabase):
    def __init__(self, config: Neo4jConfig):
        self.config = config
        self._graph = None
        self._initialize()

    def _initialize(self):
        try:
            self._graph = Neo4jGraph(
                url=self.config.url,
                username=self.config.username,
                password=self.config.password,
                database=self.config.database,
            )
            self._graph.refresh_schema()
            print("✅ Neo4j 连接成功")
        except Exception as e:
            raise ConnectionError(f"❌ Neo4j 连接失败: {str(e)}")

    def query(self, cypher: str, params: Dict = None) -> List[Dict]:
        try:
            params = params or {}
            result = self._graph.query(cypher, params)
            return result if result else []
        except Exception as e:
            raise RuntimeError(f"查询执行失败: {str(e)}")

    def get_schema(self) -> str:
        return self._graph.schema

    @property
    def graph(self) -> Neo4jGraph:
        return self._graph


class PostgreSQLRepository:
    def __init__(self, config: PostgreSQLConfig):
        self.config = config
        self._conn = None
        self._initialize()

    def _initialize(self):
        try:
            self._conn = psycopg2.connect(
                host=self.config.host,
                port=self.config.port,
                database=self.config.database,
                user=self.config.username,
                password=self.config.password,
            )
            self._conn.autocommit = True
            self._create_tables()
            print("✅ PostgreSQL 连接成功")
        except psycopg2.OperationalError:
            # 数据库不存在，尝试创建
            print("⚠️  数据库不存在，尝试创建...")
            self._create_database()
            self._conn = psycopg2.connect(
                host=self.config.host,
                port=self.config.port,
                database=self.config.database,
                user=self.config.username,
                password=self.config.password,
            )
            self._conn.autocommit = True
            self._create_tables()
            print("✅ PostgreSQL 数据库创建成功")
        except Exception as e:
            raise ConnectionError(f"❌ PostgreSQL 连接失败: {str(e)}")

    def _create_database(self):
        """创建数据库"""
        try:
            conn = psycopg2.connect(
                host=self.config.host,
                port=self.config.port,
                database="postgres",
                user=self.config.username,
                password=self.config.password,
            )
            conn.autocommit = True
            cursor = conn.cursor()
            cursor.execute(f"CREATE DATABASE {self.config.database}")
            cursor.close()
            conn.close()
        except Exception as e:
            print(f"创建数据库失败: {str(e)}")

    def _create_tables(self):
        """创建表结构"""
        with self._conn.cursor() as cursor:
            # 会话表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id SERIAL PRIMARY KEY,
                    session_id VARCHAR(100) UNIQUE NOT NULL,
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # 消息表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id SERIAL PRIMARY KEY,
                    conversation_id INTEGER REFERENCES conversations(id) ON DELETE CASCADE,
                    role VARCHAR(50) NOT NULL,
                    content TEXT NOT NULL,
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # 创建索引
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_session_id 
                ON conversations(session_id)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_conversation_id 
                ON messages(conversation_id)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_created_at 
                ON messages(created_at)
            """)

    def create_conversation(self, session_id: str, metadata: Dict = None) -> int:
        """创建新会话"""
        with self._conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO conversations (session_id, metadata) 
                VALUES (%s, %s) 
                ON CONFLICT (session_id) DO UPDATE 
                SET updated_at = CURRENT_TIMESTAMP
                RETURNING id
                """,
                (session_id, json.dumps(metadata) if metadata else None),
            )
            return cursor.fetchone()[0]

    def save_message(
        self, conversation_id: int, role: str, content: str, metadata: Dict = None
    ):
        """保存消息"""
        with self._conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO messages (conversation_id, role, content, metadata)
                VALUES (%s, %s, %s, %s)
                """,
                (
                    conversation_id,
                    role,
                    content,
                    json.dumps(metadata) if metadata else None,
                ),
            )

    def get_conversation_id(self, session_id: str) -> Optional[int]:
        """获取会话ID"""
        with self._conn.cursor() as cursor:
            cursor.execute(
                "SELECT id FROM conversations WHERE session_id = %s", (session_id,)
            )
            result = cursor.fetchone()
            return result[0] if result else None

    def get_conversation_history(self, session_id: str, limit: int = 50) -> List[Dict]:
        """获取会话历史"""
        with self._conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(
                """
                SELECT m.role, m.content, m.metadata, m.created_at
                FROM messages m
                JOIN conversations c ON m.conversation_id = c.id
                WHERE c.session_id = %s
                ORDER BY m.created_at ASC
                LIMIT %s
                """,
                (session_id, limit),
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_all_sessions(self) -> List[Dict]:
        """获取所有会话"""
        with self._conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(
                """
                SELECT session_id, created_at, updated_at,
                       (SELECT COUNT(*) FROM messages WHERE conversation_id = c.id) as message_count
                FROM conversations c
                ORDER BY updated_at DESC
                """
            )
            return [dict(row) for row in cursor.fetchall()]

    def delete_conversation(self, session_id: str):
        """删除会话"""
        with self._conn.cursor() as cursor:
            cursor.execute(
                "DELETE FROM conversations WHERE session_id = %s", (session_id,)
            )

    def close(self):
        """关闭连接"""
        if self._conn:
            self._conn.close()

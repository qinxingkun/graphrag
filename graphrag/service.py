from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import uuid
from typing import List, Dict
from langchain_community.chat_models import ChatZhipuAI
from graphrag.config import AppConfig
from graphrag.data_access import Neo4jRepository, PostgreSQLRepository
from graphrag.tools import Neo4jToolkit
from graphrag.vector_store import MilvusVectorStore
from graphrag.agent import GraphAgentWithMemory


class Neo4jQueryService:
    def __init__(self, config: AppConfig):
        self.config = config

        self.neo4j_repo = Neo4jRepository(config.neo4j)
        self.pg_repo = PostgreSQLRepository(config.postgresql)

        if config.enable_embedding:
            self.vector_store = MilvusVectorStore(config.milvus, config.llm)
            self._initialize_vector_store()
        else:
            self.vector_store = None

        self.toolkit = Neo4jToolkit(self.neo4j_repo, config.llm, self.vector_store)

        self.llm = ChatZhipuAI(
            model=config.llm.model,
            api_key=config.llm.api_key,
            temperature=config.llm.temperature,
        )

        self.agent = GraphAgentWithMemory(
            llm=self.llm,
            tools=self.toolkit.get_tools(),
            pg_config=config.postgresql,
            max_iterations=config.max_iterations,
        )

    def _initialize_vector_store(self):
        try:
            stats = self.vector_store.get_stats()
            if stats["total_count"] > 0:
                print(f"✅ Milvus 已有 {stats['total_count']} 条记录")
                return

            result = self.neo4j_repo.query("""
                MATCH (n)
                WHERE n.name IS NOT NULL OR n.description IS NOT NULL
                RETURN 
                    labels(n)[0] as label,
                    coalesce(n.name, '') as name,
                    coalesce(n.description, '') as description,
                    id(n) as node_id
                LIMIT 1000
            """)

            if result:
                texts = []
                metadatas = []
                for item in result:
                    text = f"{item['label']}: {item['name']} - {item['description']}"
                    texts.append(text)
                    metadatas.append(
                        {
                            "label": item["label"],
                            "node_id": item["node_id"],
                            "name": item["name"],
                        }
                    )

                if texts:
                    self.vector_store.add_texts(texts, metadatas)
        except Exception as e:
            print(f"⚠️  向量库初始化失败: {str(e)}")

    def query(self, question: str, session_id: str = None) -> dict:
        if session_id is None:
            session_id = str(uuid.uuid4())[:8]

        result = self.agent.invoke(question, session_id)

        self._save_to_history(session_id, question, result)

        return self._format_result(result, session_id)

    def _save_to_history(self, session_id: str, question: str, result: dict):
        try:
            conv_id = self.pg_repo.get_conversation_id(session_id)
            if conv_id is None:
                conv_id = self.pg_repo.create_conversation(session_id)

            self.pg_repo.save_message(conv_id, "user", question)

            messages = result.get("messages", [])
            if messages:
                last_msg = messages[-1]
                if isinstance(last_msg, AIMessage):
                    self.pg_repo.save_message(conv_id, "assistant", last_msg.content)
        except Exception as e:
            print(f"⚠️  保存历史记录失败: {str(e)}")

    def _format_result(self, result: dict, session_id: str) -> dict:
        messages = result.get("messages", [])

        formatted = {
            "session_id": session_id,
            "question": messages[0].content if messages else "",
            "answer": messages[-1].content if messages else "",
            "conversation": [],
            "tool_calls": [],
        }

        for msg in messages:
            if isinstance(msg, HumanMessage):
                formatted["conversation"].append(
                    {"role": "user", "content": msg.content}
                )
            elif isinstance(msg, AIMessage):
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        formatted["tool_calls"].append(
                            {"tool": tool_call["name"], "args": tool_call["args"]}
                        )
                else:
                    formatted["conversation"].append(
                        {"role": "assistant", "content": msg.content}
                    )
            elif isinstance(msg, ToolMessage):
                formatted["conversation"].append(
                    {
                        "role": "tool",
                        "content": msg.content[:300] + "..."
                        if len(msg.content) > 300
                        else msg.content,
                    }
                )

        return formatted

    def get_conversation_history(self, session_id: str) -> List[Dict]:
        return self.pg_repo.get_conversation_history(session_id)

    def list_sessions(self) -> List[Dict]:
        return self.pg_repo.get_all_sessions()

    def delete_session(self, session_id: str):
        self.pg_repo.delete_conversation(session_id)

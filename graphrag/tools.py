from langchain_core.tools import tool
from langchain_neo4j import GraphCypherQAChain
from langchain_community.chat_models import ChatZhipuAI
from graphrag.data_access import Neo4jRepository
from graphrag.config import LLMConfig
from graphrag.vector_store import MilvusVectorStore


class Neo4jToolkit:
    def __init__(
        self,
        neo4j_repo: Neo4jRepository,
        llm_config: LLMConfig,
        vector_store: MilvusVectorStore = None,
    ):
        self.neo4j_repo = neo4j_repo
        self.vector_store = vector_store
        self.llm = ChatZhipuAI(
            model=llm_config.model,
            api_key=llm_config.api_key,
            temperature=llm_config.temperature,
        )

    def get_tools(self) -> list:
        tools = [
            self._create_cypher_tool(),
            self._create_nl_query_tool(),
            self._create_schema_tool(),
        ]

        if self.vector_store:
            tools.append(self._create_semantic_search_tool())
            tools.append(self._create_hybrid_search_tool())

        return tools

    def _create_cypher_tool(self):
        repo = self.neo4j_repo

        @tool
        def neo4j_cypher_query(query: str) -> str:
            """
            执行 Cypher 查询语句。
            适用于精确的结构化查询。

            示例：
            - "MATCH (n) RETURN labels(n), count(n)"
            - "MATCH (p:Person)-[r]->(m) RETURN p.name, type(r), m.name LIMIT 10"
            """
            try:
                result = repo.query(query)
                if not result:
                    return "查询未返回任何结果"
                return str(result[:50])
            except Exception as e:
                return f"查询失败: {str(e)}"

        return neo4j_cypher_query

    def _create_nl_query_tool(self):
        repo = self.neo4j_repo
        llm = self.llm

        @tool
        def neo4j_natural_language_query(question: str) -> str:
            """
            使用自然语言查询 Neo4j 知识图谱。
            适用于结构化知识的精确查询。

            示例：
            - "有多少个人节点？"
            - "张三认识哪些人？"
            - "找出度数最高的节点"
            """
            try:
                qa_chain = GraphCypherQAChain.from_llm(
                    llm=llm,
                    graph=repo.graph,
                    verbose=False,
                    allow_dangerous_requests=True,
                )
                response = qa_chain.invoke({"query": question})
                return response.get("result", "未找到相关信息")
            except Exception as e:
                return f"查询失败: {str(e)}"

        return neo4j_natural_language_query

    def _create_schema_tool(self):
        repo = self.neo4j_repo

        @tool
        def get_neo4j_schema() -> str:
            """
            获取 Neo4j 数据库的模式信息。
            包括节点类型、关系类型、属性等。
            """
            try:
                return repo.get_schema()
            except Exception as e:
                return f"获取模式失败: {str(e)}"

        return get_neo4j_schema

    def _create_semantic_search_tool(self):
        """语义搜索工具"""
        vector_store = self.vector_store

        @tool
        def semantic_search(query: str, top_k: int = 5) -> str:
            """
            使用语义相似度搜索知识库（基于 Milvus 向量数据库）。
            适用于模糊查询、概念搜索、相关内容推荐。

            示例：
            - "找到与人工智能相关的内容"
            - "搜索机器学习的概念"
            - "查找相似的实体"
            """
            try:
                results = vector_store.similarity_search(query, top_k=top_k)
                if not results:
                    return "未找到相关内容"

                output = f"🔍 找到 {len(results)} 个语义相关结果:\n\n"
                for i, (text, score, meta) in enumerate(results, 1):
                    output += f"{i}. [相似度: {score:.3f}]\n"
                    output += f"   {text[:200]}...\n"
                    if meta:
                        output += f"   📎 {meta}\n"
                    output += "\n"

                return output
            except Exception as e:
                return f"语义搜索失败: {str(e)}"

        return semantic_search

    def _create_hybrid_search_tool(self):
        """混合搜索工具 - 结合结构化和语义搜索"""
        neo4j_repo = self.neo4j_repo
        vector_store = self.vector_store

        @tool
        def hybrid_search(query: str, top_k: int = 5) -> str:
            """
            混合搜索：同时使用结构化查询和语义搜索。
            适用于复杂查询，需要同时考虑精确匹配和语义相关性。

            示例：
            - "找到与AI相关且最近更新的内容"
            - "搜索重要的机器学习概念"
            """
            try:
                # 1. 语义搜索
                semantic_results = vector_store.similarity_search(query, top_k=top_k)

                # 2. 提取相关实体ID进行图查询
                entity_ids = []
                for _, _, meta in semantic_results:
                    if "node_id" in meta:
                        entity_ids.append(meta["node_id"])

                # 3. 图查询获取关系信息
                graph_results = []
                if entity_ids:
                    cypher = """
                    MATCH (n)-[r]-(m)
                    WHERE id(n) IN $ids
                    RETURN n.name as entity, type(r) as relation, m.name as related
                    LIMIT 10
                    """
                    graph_results = neo4j_repo.query(cypher, {"ids": entity_ids})

                # 4. 合并结果
                output = "🔄 混合搜索结果:\n\n"
                output += "📊 语义相关内容:\n"
                for i, (text, score, meta) in enumerate(semantic_results[:3], 1):
                    output += f"  {i}. [{score:.3f}] {text[:150]}...\n"

                if graph_results:
                    output += "\n🕸️  关系图谱:\n"
                    for gr in graph_results[:5]:
                        output += f"  • {gr.get('entity')} --{gr.get('relation')}--> {gr.get('related')}\n"

                return output

            except Exception as e:
                return f"混合搜索失败: {str(e)}"

        return hybrid_search

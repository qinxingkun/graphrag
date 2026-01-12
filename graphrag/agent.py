from typing import TypedDict, Annotated, Sequence
from graphrag.config import PostgreSQLConfig
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg import Connection
from psycopg.rows import dict_row
import operator

from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    ChatMessagePromptTemplate,
    MessagesPlaceholder,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.retrievers import BaseRetriever
from langchain_core.document_loaders import (
    BaseBlobParser,
    BlobLoader,
    LangSmithLoader,
)  # 弃用，从Langchain_community
from langchain_core.stores import BaseStore, InMemoryBaseStore, InMemoryStore
from langchain_core.output_parsers import (
    JsonOutputParser,
    PydanticOutputParser,
    MarkdownListOutputParser,
    XMLOutputParser,
)
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.callbacks import BaseCallbackHandler, BaseCallbackManager, Callbacks
from langchain.chat_models import BaseChatModel

from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    AIMessage,
)  # 直接用langchain_core
from langchain.messages import HumanMessage, SystemMessage, AIMessage


# 看情况
from langchain_core.embeddings import Embeddings
from langchain.embeddings import Embeddings, init_embeddings

from langchain_core.agents import (
    AgentAction,
    AgentStep,
    AgentFinish,
)  # 已弃用，官方建议直接用langchain.agents
from langchain.agents import AgentState, create_agent

# 为了项目开发可维护性，从langchain_core直接进行导入,其所依赖的包是一致的，并且langchain_core.tools提供了比langchain.tools更多的包
from langchain.tools import BaseTool, ToolException, tool
from langchain_core.tools import (
    BaseTool,
    ToolException,
    tool,
    StructuredTool,
    BaseToolkit,
)

from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.memory import InMemorySaver

from langgraph.graph.state import StateGraph
from langgraph.graph.message import MessagesState, add_messages

from langgraph.store.memory import InMemoryStore
from langgraph.store.postgres import PostgresStore


class AgentState(TypedDict):
    """Agent 状态"""

    messages: Annotated[Sequence[BaseMessage], operator.add]
    iteration_count: int
    session_id: str


class GraphAgentWithMemory:
    def __init__(
        self, llm, tools: list, pg_config: PostgreSQLConfig, max_iterations: int = 10
    ):
        self.llm = llm
        self.tools = tools
        self.max_iterations = max_iterations
        self.llm_with_tools = llm.bind_tools(tools)

        conn = Connection.connect(
            pg_config.connection_string,
            autocommit=True,
            prepare_threshold=0,
            row_factory=dict_row,
        )
        self.memory = PostgresSaver(conn)
        self.memory.setup()

        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(AgentState)

        workflow.add_node("agent", self._agent_node)
        workflow.add_node("tools", ToolNode(self.tools))

        workflow.set_entry_point("agent")

        workflow.add_conditional_edges(
            "agent", self._should_continue, {"tools": "tools", "end": END}
        )

        workflow.add_edge("tools", "agent")

        return workflow.compile(checkpointer=self.memory)

    def _agent_node(self, state: AgentState) -> dict:
        messages = state["messages"]
        iteration = state.get("iteration_count", 0)
        response = self.llm_with_tools.invoke(messages)
        return {"messages": [response], "iteration_count": iteration + 1}

    def _should_continue(self, state: AgentState) -> str:
        messages = state["messages"]
        last_message = messages[-1]
        iteration = state.get("iteration_count", 0)

        if iteration >= self.max_iterations:
            return "end"

        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"

        return "end"

    def invoke(self, question: str, session_id: str = "default") -> dict:
        initial_state = {
            "messages": [HumanMessage(content=question)],
            "iteration_count": 0,
            "session_id": session_id,
        }

        config = {"configurable": {"thread_id": session_id}}
        return self.graph.invoke(initial_state, config)

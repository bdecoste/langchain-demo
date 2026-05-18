import os
import base64
import asyncio

from pprint import pprint
from dotenv import load_dotenv
from uuid import uuid7
from dataclasses import dataclass
from typing import Callable
from typing import Dict, Any
from typing import TypedDict, Literal

from tavily import TavilyClient

from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_anthropic import ChatAnthropic
from langchain.agents import create_agent
from langchain.agents.middleware import ModelFallbackMiddleware
from langchain.agents.middleware import SummarizationMiddleware
from langchain.chat_models import init_chat_model
from langsmith import traceable
from langchain.tools import tool, ToolRuntime
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse


# ---------------------------------------------------------------------------
# Runtime context
# Injected server-side per request so the model never sees or controls these values.
# ---------------------------------------------------------------------------

@dataclass
class AlbumContext(TypedDict, total=False):
    least_favorite_album: str = "U2"
    user_role: Literal["internal", "external"] = "internal"


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool
def get_least_favorite_album(runtime: ToolRuntime) -> str:
    """Get the least favourite album of the user"""
    return runtime.context.least_favorite_album


# ---------------------------------------------------------------------------
# Middleware
# Runs before each model call to swap the active tool set based on user role.
# Internal users get raw SQL access; external users are restricted to web search.
# ---------------------------------------------------------------------------

@wrap_model_call
async def dynamic_tool_call(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """Dynamically call tools based on the runtime context"""

    ctx = request.runtime.context
    user_role = ctx.user_role if ctx is not None else "internal"

    if user_role == "internal":
        tools = [sql_query, get_least_favorite_album]
        request = request.override(tools=tools)
    else:
        tools = [web_search]
        request = request.override(tools=tools)

    return await handler(request)


# ---------------------------------------------------------------------------
# External tools
# ---------------------------------------------------------------------------

tavily_client = TavilyClient()


@tool
def web_search(query: str) -> Dict[str, Any]:
    """Search the web for information"""
    return tavily_client.search(query)


@tool
def sql_query(query: str) -> str:
    """Obtain information from the database using SQL queries"""
    try:
        return db.run(query)
    except Exception as e:
        return f"Error: {e}"


# Connect to the Chinook SQLite database for direct SQL access
db = SQLDatabase.from_uri("sqlite:///chinook.db")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

model = init_chat_model(
    "openai:gpt-5.4",
    timeout=600,
    max_tokens=25000,
    streaming=True,
)
# model = ChatAnthropic(model="claude-opus-4-7")

# Same toolkit as before — these SQL tools haven't changed
# toolkit = SQLDatabaseToolkit(db=db, llm=model)
# tools = toolkit.get_tools()
# tools.append(get_least_favorite_album)

system_prompt = """

You are the manager of a record store

User: What is your favorite album?
Record Store Manager: 2112

Please use the below structure.

Band: The name of the band
Customer: The name of the customer
Album: The name of the album

"""

# checkpointer = InMemorySaver()

# ---------------------------------------------------------------------------
# Agent
# create_agent wires model + tools + middleware into a single streaming agent.
# ---------------------------------------------------------------------------

agent = create_agent(
    model=model,
    tools=[web_search, sql_query, get_least_favorite_album],
    system_prompt=system_prompt,
    # checkpointer=checkpointer,
    context_schema=AlbumContext,
    middleware=[
        dynamic_tool_call,
        ModelFallbackMiddleware("claude-opus-4-7"),
        # SummarizationMiddleware(
        #     model="claude-opus-4-7",
        #     trigger=("tokens", 5000),
        #     keep=("messages", 1),
        # ),
    ],
)


# ---------------------------------------------------------------------------
# Chat pipeline
# Streams agent steps and pretty-prints each message as it arrives.
# ---------------------------------------------------------------------------

@traceable(name="Chat Bot")
def chat_pipeline(messages: list, get_chat_history: bool = False):
    try:
        for step in agent.stream(
            {"messages": [("user", messages)]},
            stream_mode="values",
            context=AlbumContext(),
        ):
            step["messages"][-1].pretty_print()

        print("\n")
        pprint(step["messages"])
    except Exception as e:
        print(f"An error occurred: {e}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

# while True:
#     messages = input("\n\nPlease enter message: \n")
chat_pipeline("", get_chat_history=False)

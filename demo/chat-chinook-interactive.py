import os
import base64
import asyncio

from pprint import pprint
from dotenv import load_dotenv
from uuid import uuid7
from dataclasses import dataclass
from typing import Callable
from typing import Dict, Any

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

@dataclass
class AlbumContext:
    least_favorite_album: str = "U2"
    user_role: Literal["internal", "external"] = "internal"

@tool
def get_least_favorite_album(runtime: ToolRuntime) -> str:
    """Get the least favourite album of the user"""
    return runtime.context.least_favorite_album

@wrap_model_call
def dynamic_tool_call(request: ModelRequest, 
handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:

    """Dynamically call tools based on the runtime context"""

    user_role = request.runtime.context.user_role
    
    if user_role == "internal":
        tools = [sql_query, get_least_favorite_album] 
        request = request.override(tools=tools) 
    else:
        tools = [web_search] 
        request = request.override(tools=tools) 

    return handler(request)
    
THREAD_ID = str(uuid7())
THREADS_DIR = os.path.join(os.path.dirname(__file__), "threads")

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

# Connect to the database
db = SQLDatabase.from_uri("sqlite:///chinook.db")

# Initialize the model
model = init_chat_model(
    "openai:gpt-5.4",
    timeout=600,
    max_tokens=25000,
    streaming=True,
)
#model = ChatAnthropic(model="claude-opus-4-7")

# Same toolkit as before — these SQL tools haven't changed
#toolkit = SQLDatabaseToolkit(db=db, llm=model)
#tools = toolkit.get_tools()
#tools.append(get_least_favorite_album)

system_prompt = """

You are the manager of a record store

User: What is your favorite album?
Record Store Manager: 2112

Please use the below structure.

Band: The name of the band
Customer: The name of the customer
Album: The name of the album

"""

checkpointer = InMemorySaver()

# create_agent takes model, tools, and system_prompt directly
agent = create_agent(
    model=model,
    tools=[web_search, sql_query, get_least_favorite_album],
    system_prompt=system_prompt,
    checkpointer=checkpointer,
    context_schema=AlbumContext,
    middleware=[
        dynamic_tool_call,
        ModelFallbackMiddleware(
            "claude-opus-4-7",
        ),
#        SummarizationMiddleware(
#            model="claude-opus-4-7",
#            trigger=("tokens", 5000),
#            keep=("messages", 1)
#        )
    ],
)

@traceable(name="Chat Bot", metadata={"thread_id": THREAD_ID})
def chat_pipeline(messages: list, get_chat_history: bool = False):
    try:
        for step in agent.stream(
            {"messages": [("user", messages)]},
            stream_mode="values",
            config={"configurable": {"thread_id": THREAD_ID}},
            context=AlbumContext(),
        ):
            step["messages"][-1].pretty_print()

        print("\n")
        pprint(step["messages"])
    except Exception as e:
        print(f"An error occurred: {e}")

while True:
    messages = input("\n\nPlease enter message: \n")
    chat_pipeline(messages, get_chat_history=False)



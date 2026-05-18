import os
import base64
import asyncio

from pprint import pprint
from dotenv import load_dotenv
from uuid import uuid7
from dataclasses import dataclass
from typing import Callable
from typing import TypedDict, Dict, Any

from tavily import TavilyClient
import sqlite3

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
# Injected per-request so the model never sees or controls these values.
# ---------------------------------------------------------------------------

@dataclass
class AlbumContext:
    least_favorite_album: str = "Achtung Baby"
    user_role: Literal["internal", "external"] = "internal"
    customer_id: str = 23


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool
def get_least_favorite_album(runtime: ToolRuntime) -> str:
    """Get the least favourite album of the user"""
    return runtime.context.least_favorite_album


class PurchasedTrack(TypedDict):
    """A single purchased track with metadata and purchase info."""
    track_id: int
    track_name: str
    album: str | None
    artist: str | None
    genre: str | None
    unit_price: float
    quantity: int
    invoice_id: int
    invoice_date: str


# Connect to the Chinook SQLite database for direct SQL access
db = SQLDatabase.from_uri("sqlite:///chinook.db")


@tool
def get_purchased_tracks(runtime: ToolRuntime) -> list[PurchasedTrack]:
    """Get all tracks the current customer has purchased.

    Returns each track the authenticated customer has bought, including
    track name, album, artist, genre, price paid, quantity, and the
    invoice it appeared on. Results are ordered by invoice date (newest first).
    """
    # customer_id comes from the runtime context, NOT the model — prevents IDOR
    customer_id = runtime.context.customer_id

    query = """
        SELECT
            t.TrackId         AS track_id,
            t.Name            AS track_name,
            al.Title          AS album,
            ar.Name           AS artist,
            g.Name            AS genre,
            il.UnitPrice      AS unit_price,
            il.Quantity       AS quantity,
            i.InvoiceId       AS invoice_id,
            i.InvoiceDate     AS invoice_date
        FROM Invoice         AS i
        JOIN InvoiceLine     AS il ON il.InvoiceId = i.InvoiceId
        JOIN Track           AS t  ON t.TrackId    = il.TrackId
        LEFT JOIN Album      AS al ON al.AlbumId   = t.AlbumId
        LEFT JOIN Artist     AS ar ON ar.ArtistId  = al.ArtistId
        LEFT JOIN Genre      AS g  ON g.GenreId    = t.GenreId
        WHERE i.CustomerId = ?
        ORDER BY i.InvoiceDate DESC, il.InvoiceLineId ASC;
    """

    with sqlite3.connect("chinook.db") as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(query, (customer_id,)).fetchall()
    return [dict(row) for row in rows]


# ---------------------------------------------------------------------------
# Middleware
# Runs before each model call to adjust the tool set based on the caller's role.
# Internal users get raw SQL access; external users get web search + purchase history.
# ---------------------------------------------------------------------------

@wrap_model_call
def dynamic_tool_call(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """Dynamically call tools based on the runtime context"""

    ctx = request.runtime.context
    user_role = ctx.user_role if ctx is not None else "external"

    if user_role == "internal":
        tools = [sql_query, get_least_favorite_album]
        request = request.override(tools=tools)
    else:
        tools = [web_search, get_purchased_tracks]
        request = request.override(tools=tools)

    return handler(request)


# ---------------------------------------------------------------------------
# Conversation state
# Each session gets a unique thread ID so InMemorySaver can replay history.
# ---------------------------------------------------------------------------

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

checkpointer = InMemorySaver()

# ---------------------------------------------------------------------------
# Agent
# create_agent wires model + tools + middleware + persistent checkpointer.
# ---------------------------------------------------------------------------

agent = create_agent(
    model=model,
    tools=[web_search, sql_query, get_least_favorite_album, get_purchased_tracks],
    system_prompt=system_prompt,
    checkpointer=checkpointer,
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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

while True:
    messages = input("\n\nPlease enter message: \n")
    chat_pipeline(messages, get_chat_history=False)

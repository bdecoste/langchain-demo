import os
import base64
import asyncio

from pprint import pprint
from dotenv import load_dotenv
from uuid import uuid7

from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_anthropic import ChatAnthropic
from langchain.agents import create_agent 
from langchain.agents.middleware import ModelFallbackMiddleware
from langchain.chat_models import init_chat_model
from langsmith import traceable


uploaded_file = uploader.value[0]
content_mv = uploaded_file["content"]
img_bytes = bytes(content_mv)
img_b64 = base64.b64encode(img_bytes).decode("utf-8")

THREAD_ID = str(uuid7())
THREADS_DIR = os.path.join(os.path.dirname(__file__), "threads")

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
toolkit = SQLDatabaseToolkit(db=db, llm=model)
tools = toolkit.get_tools()

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
    tools=tools,
    system_prompt=system_prompt,
    checkpointer=checkpointer,
    middleware=[
        ModelFallbackMiddleware(
            "claude-opus-4-7",
        ),
    ],
)

@traceable(name="Chat Bot", metadata={"thread_id": THREAD_ID})
def chat_pipeline(messages: list, get_chat_history: bool = False):
    try:
        for step in agent.stream(
            {"messages": [("user", messages)]},
            stream_mode="values",
            config={"configurable": {"thread_id": THREAD_ID}}
        ):
            step["messages"][-1].pretty_print()

        print("\n")
        pprint(step["messages"])
    except Exception as e:
        print(f"An error occurred: {e}")

while True:
#    messages = input("\n\nPlease enter message: \n")
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=[
            {"type": "text", "text": "Use this image to answer."},
            {
                "type": "image",
                "source_type": "base64",
                "mime_type": "image/png",
                "data": img_b64,
            },
        ]),
    ]
    chat_pipeline(messages, get_chat_history=False)



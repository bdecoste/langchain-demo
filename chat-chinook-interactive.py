from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_anthropic import ChatAnthropic
from langchain.agents import create_agent 
from langchain.agents.middleware import ModelFallbackMiddleware
from langchain.chat_models import init_chat_model

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

system_prompt = """Please enter your request:"""

# create_agent takes model, tools, and system_prompt directly
agent = create_agent(
    model=model,
    tools=tools,
    system_prompt=system_prompt,
    middleware=[
        ModelFallbackMiddleware(
            "claude-opus-4-7",
        ),
    ],
)

while True:
    question = input("\n\nPlease enter your request: \n")

    for step in agent.stream(
        {"messages": [("user", question)]},
        stream_mode="values",
    ):
        step["messages"][-1].pretty_print()



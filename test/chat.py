from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_anthropic import ChatAnthropic
from langchain.agents import create_agent  # <-- new import path

# Connect to the database
db = SQLDatabase.from_uri("sqlite:///company.db")

# Initialize the model
llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0)

# Same toolkit as before — these SQL tools haven't changed
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
tools = toolkit.get_tools()

system_prompt = """You are an agent designed to interact with a SQL database.

Given an input question, create a syntactically correct SQLite query, run it,
look at the results, and return the answer.

Always start by listing tables, then inspect the schema of relevant tables
before writing a query. Limit results to 10 rows unless the user asks for more.
Never make DML statements (INSERT, UPDATE, DELETE, DROP) — read-only queries only.

If the question doesn't relate to the database, politely say you can't help."""

# create_agent takes model, tools, and system_prompt directly
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=system_prompt,
)

question = "Which department has the highest average salary, and who's the top earner there?"

for step in agent.stream(
    {"messages": [("user", question)]},
    stream_mode="values",
):
    step["messages"][-1].pretty_print()



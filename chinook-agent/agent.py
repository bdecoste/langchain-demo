"""LangChain SQL agent for the Chinook music store database."""

import os
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit, create_sql_agent
from langchain_anthropic import ChatAnthropic
from setup_db import setup_database


def create_agent():
    db_path = setup_database()
    db = SQLDatabase.from_uri(f"sqlite:///{db_path}")

    llm = ChatAnthropic(
        model="claude-sonnet-4-6",
        temperature=0,
    )

    toolkit = SQLDatabaseToolkit(db=db, llm=llm)

    return create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        agent_type="tool-calling",
        verbose=True,
        max_iterations=10,
    )


def main():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise EnvironmentError("ANTHROPIC_API_KEY environment variable is not set")

    print("Chinook Music Store SQL Agent")
    print("=" * 40)
    print("Tables: Artist, Album, Track, Genre, Customer, Invoice, Employee, Playlist")
    print("Type 'quit' or 'exit' to stop.\n")

    agent_executor = create_agent()

    while True:
        try:
            question = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in {"quit", "exit"}:
            print("Goodbye!")
            break

        try:
            result = agent_executor.invoke({"input": question})
            print(f"\nAnswer: {result['output']}\n")
        except Exception as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    main()

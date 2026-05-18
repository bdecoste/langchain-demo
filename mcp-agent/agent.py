"""LangChain agent powered by an MCP server via langchain-mcp-adapters."""

import asyncio
import os
from pathlib import Path

from langchain_anthropic import ChatAnthropic
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition

SERVER_PATH = str(Path(__file__).parent / "server.py")


def build_graph(llm, tools):
    """Compile a LangGraph react-style agent graph."""
    model_with_tools = llm.bind_tools(tools)

    def call_model(state: MessagesState):
        response = model_with_tools.invoke(state["messages"])
        return {"messages": response}

    builder = StateGraph(MessagesState)
    builder.add_node("call_model", call_model)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "call_model")
    builder.add_conditional_edges("call_model", tools_condition)
    builder.add_edge("tools", "call_model")
    return builder.compile()


async def main():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise EnvironmentError("ANTHROPIC_API_KEY environment variable is not set")

    llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0)

    client = MultiServerMCPClient(
        {
            "utility": {
                "command": "python3.11",
                "args": [SERVER_PATH],
                "transport": "stdio",
            }
        }
    )

    tools = await client.get_tools()
    print(f"Loaded {len(tools)} MCP tools: {[t.name for t in tools]}\n")

    graph = build_graph(llm, tools)

    print("LangChain + MCP Agent")
    print("=" * 40)
    print("Ask anything — the agent will use MCP tools when helpful.")
    print("Type 'quit' or 'exit' to stop.\n")

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
            result = await graph.ainvoke(
                {"messages": [{"role": "user", "content": question}]}
            )
            answer = result["messages"][-1].content
            print(f"\nAnswer: {answer}\n")
        except Exception as exc:
            print(f"Error: {exc}\n")


if __name__ == "__main__":
    asyncio.run(main())

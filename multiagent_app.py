"""
Interactive application to run the LangGraph multi-agent demo.
Run with: python multiagent_app.py
"""
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

from state import ProjectState
from multiagent_supervisor import supervisor_node
from multiagent_agents import research_agent, execute_agent


def build_graph():
    builder = StateGraph(ProjectState)

    builder.add_node(supervisor_node, name="supervisor_node")
    builder.add_node(research_agent, name="research_agent")
    builder.add_node(execute_agent, name="execute_agent")

    # edges
    builder.add_edge(START, "supervisor_node")
    builder.add_edge("supervisor_node", "research_agent")

    # research loop
    builder.add_edge("research_agent", "research_agent")
    builder.add_edge("research_agent", "execute_agent")

    builder.add_edge("execute_agent", END)

    return builder


def run_interactive():
    print("=== LangGraph: Research + Execute Demo ===")
    query = input("Enter your query: ")

    state = ProjectState(query=query)
    checkpointer = InMemorySaver()

    graph = build_graph().compile(checkpointer=checkpointer)
    result_state = graph.invoke(state, {"configurable": {"thread_id": "session-1"}})

    print("\n=== RUN SUMMARY ===")
    print(f"Research attempts: {result_state.get('research_attempts')}")
    print("Research notes:")
    for n in result_state.get("research_notes", []):
        print(f" - {n}")
    print("\nExecution log:")
    for l in result_state.get("execution_log", []):
        print(f" - {l}")
    print("\nResults saved to: results.txt")
    print("=== END ===")


if __name__ == '__main__':
    run_interactive()
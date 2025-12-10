"""
Main interactive CLI application for the LangGraph multi-agent demo.
Run: python multiagent_app.py
"""

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from dataclasses import dataclass, field
from typing import List, Any, Dict

from multiagent_supervisor import supervisor_node
from multiagent_agents import researcher_node, writer_node

@dataclass
class ProjectState(dict):
    query: str = ""
    findings: List[str] = field(default_factory=list)
    attempts: int = 0
    sufficient: bool = False
    final_output: str = ""


def build_graph():
    builder = StateGraph(ProjectState)

    builder.add_node(supervisor_node, name="supervisor_node")
    builder.add_node(researcher_node, name="researcher_node")
    builder.add_node(writer_node, name="writer_node")

    builder.add_edge(START, "supervisor_node")
    builder.add_edge("supervisor_node", "researcher_node")
    builder.add_edge("researcher_node", "researcher_node")   # loop
    builder.add_edge("researcher_node", "writer_node")
    builder.add_edge("writer_node", END)

    return builder


def run_interactive():
    print("=== Multi-Agent System (LangGraph) ===")
    query = input("Enter your query: ")

    state = ProjectState(query=query)
    checkpointer = InMemorySaver()

    graph = build_graph().compile(checkpointer=checkpointer)
    result = graph.invoke(state, {"configurable": {"thread_id": "session-1"}})

    print("\n=== RESULTS ===")
    print(result["final_output"])
    print("================")


if __name__ == "__main__":
    run_interactive()

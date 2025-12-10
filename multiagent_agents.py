"""Agents module: researcher and writer"""
from dataclasses import dataclass, field
from typing import List, Dict, Any


def researcher_node(state: Dict[str, Any]) -> Dict[str, Any]:
    state["attempts"] = state.get("attempts", 0) + 1
    attempt = state["attempts"]

    pool = {
        1: ["fact A: causes", "fact B: background"],
        2: ["fact C: stats", "fact D: example"],
        3: ["fact E: reference", "fact F: counterexample"],
    }
    found = pool.get(attempt, [f"extra fact {attempt}"])

    state.setdefault("findings", [])
    state["findings"].extend(found)

    if len(state["findings"]) >= 4 or attempt >= 3:
        state["sufficient"] = True
        return {"message": "research complete", "next": "writer_node"}

    state["sufficient"] = False
    return {"message": "need more research", "next": "researcher_node"}


def writer_node(state: Dict[str, Any]) -> Dict[str, Any]:
    findings = state.get("findings", [])
    intro = f"Summary for query: {state.get('query')}\n\n"
    body = "\n".join(f"- {f}" for f in findings)
    conclusion = f"\n\nGenerated in {state.get('attempts')} research iterations." 
    final = intro + body + conclusion
    state["final_output"] = final
    return {"message": "done", "final_output": final}

import os
import subprocess
from typing import Dict, Any

# Research agent: gathers incremental notes and sets ready_to_execute when done

def research_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    state["research_attempts"] = state.get("research_attempts", 0) + 1
    attempt = state["research_attempts"]

    # Replace this section with real LLM calls if desired
    research_data = {
        1: ["Identify problem", "Gather basic info"],
        2: ["Analyze constraints", "Generate possible actions"],
        3: ["Validate best strategy", "Prepare execution plan"]n    }

    new_notes = research_data.get(attempt, [f"Extra research round {attempt}"])
    state.setdefault("research_notes", [])
    state["research_notes"].extend(new_notes)

    # After 3 attempts we mark the plan ready
    if attempt >= 3:
        state["ready_to_execute"] = True
        return {"message": "Research complete", "next": "execute_agent"}

    state["ready_to_execute"] = False
    return {"message": "Continuing research", "next": "research_agent"}


# Execution agent: writes results to a notepad file and optionally opens Notepad (Windows)

def execute_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    notes = state.get("research_notes", [])
    attempts = state.get("research_attempts", 0)

    file_path = "results.txt"

    if attempts >= 3:
        status_message = "Research depth: HIGH – full execution applied."
    else:
        status_message = "Research depth: LOW – partial execution applied."

    content = [
        f"Query: {state.get('query', '')}",
        "",
        "=== Research Notes ===",
        *notes,
        "",
        "=== Execution Status ===",
        status_message,
        "",
        f"Total Research Attempts: {attempts}",
    ]

    # Write the file
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(content))

    abs_path = os.path.abspath(file_path)
    state.setdefault("execution_log", [])
    state["execution_log"].append(f"Results written to {abs_path}")
    state["result"] = "Execution completed and file saved."

    # OPTIONAL: auto-open Notepad on Windows
    try:
        if os.name == "nt":
            subprocess.Popen(["notepad.exe", abs_path])
    except Exception:
        # ignore errors opening Notepad
        pass

    return {"message": "Execution finished", "output_file": file_path}
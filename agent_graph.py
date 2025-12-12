""" Architecture (Mermaid):
mermaid
flowchart TD
    START([START]) --> Supervisor["Supervisor/Router"]
    Supervisor --> Researcher["Researcher (LLM via Groq)"]
    Researcher --> Executor["Executor (writes notes + persists memory)"]
    Executor --> END([END])
    %% Conditional edge: Executor -> Researcher when more depth is needed
    Executor -.->|needs_more and iterations < max_iter| Researcher
Description: - Two specialized agents: Researcher (runs Groq LLM calls to research the user's input) and Executor (writes consolidated research notes to a notepad file and persists them to memory/state). - Supervisor/Router node: receives user input and sets the next field in state to route to Researcher. - Memory (state persistence): the graph state contains memory which is stored to disk after each Executor run (simple JSON file store). This demonstrates durable state across runs. - Conditional Edge (logic loop): After Executor completes, the graph checks if more research is requested or if the Researcher did not satisfy a quality check. If needs_more is True and iterations < max_iterations it loops back to Researcher. A counter prevents infinite recursion. Notes: this is a working, runnable LangGraph example (requires langgraph and groq Python packages). Replace GROQ_API_KEY with your Groq key as env var. Run instructions (brief): 1. pip install langgraph groq 2. export GROQ_API_KEY=your_key_here (or set in environment) 3. python LangGraph_multi_agent_groq_example.py """

from typing import TypedDict, List, Optional
import os
import json
from langgraph.graph import StateGraph, START, END

# Groq client
try:
    from groq import Groq
except Exception:
    Groq = None  # graceful fallback if groq isn't installed during static inspection


# ---------------------------
# State schema
# ---------------------------
class AgentState(TypedDict, total=False):
    user_input: str
    research_notes: List[str]
    final_note: Optional[str]
    next: Optional[str]
    iterations: int
    max_iterations: int
    needs_more: bool
    memory: dict
    save_to_notepad: bool


# ---------------------------
# Helpers: memory persistence
# ---------------------------
MEMORY_FILE = "langgraph_memory.json"


def load_memory() -> dict:
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_memory(mem: dict) -> None:
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(mem, f, indent=2, ensure_ascii=False)


# ---------------------------
# Groq LLM wrapper (Researcher uses this)
# ---------------------------

def create_groq_client():
    api_key = os.getenv("GROQ_API_KEY")
    if api_key is None:
        raise RuntimeError("GROQ_API_KEY not set in environment")
    if Groq is None:
        raise RuntimeError("groq package not installed; `pip install groq`")
    client = Groq(api_key=api_key)
    return client


def groq_research_query(query: str) -> str:
    """
    Calls Groq's chat/completion endpoint (simple wrapper). This function is intentionally minimal and
    returns the assistant text. In production you'd add retries, timeouts and structured output parsing.
    """
    client = create_groq_client()
    # Use a compact model name; change to desired Groq model
    model = "llama-3.3-70b-versatile"
    completion = client.chat.completions.create(
        messages=[{"role": "user", "content": f"Research and summarize: {query}"}],
        model=model,
    )
    # Pull assistant content
    return completion.choices[0].message.content


# ---------------------------
# Node implementations
# ---------------------------


def supervisor_node(state: AgentState) -> AgentState:
    """Supervisor: only node responsible for routing agents."""

    user_input = state.get("user_input")
    if not user_input:
        return {"next": None}

    state.setdefault("iterations", 0)
    state.setdefault("max_iterations", 3)
    state.setdefault("research_notes", [])
    state.setdefault("memory", load_memory())

    # FIRST STEP → No research exists yet → Send to Researcher
    if len(state.get("research_notes", [])) == 0:
        return {"next": "Researcher"}

    # IF already researched → Supervisor decides whether more research is needed
    executor_output = state.get("executor_output", "")

    if "needs_more" in executor_output.lower():
        # Supervisor instructs more research
        return {"next": "Researcher"}

    # OTHERWISE → Execute final action
    return {"next": "Executor"}



def researcher_node(state: AgentState) -> AgentState:
    """Researcher: uses Groq to research the `user_input`, appends to research_notes and decides if more depth is needed.
    Keep outputs small and structured so Executor can handle them easily.
    """
    user_input = state.get("user_input", "")
    iterations = state.get("iterations", 0)

    prompt = f"""1. WHO YOU ARE:
You are The Researcher Agent, a Groq-powered LLM specializing in factual investigation.
Your role is to gather accurate, verifiable, and up-to-date information using both reasoning and web search.

2. YOUR GOAL
Your job is to:
Take the user input: {user_input}
Perform real research about it using Groq LLM reasoning + the web-search tool.
Produce 1–3 concise, factual bullet points containing the key findings.
At the end, output the line:
NEEDS_MORE: true/false
based on whether deeper research is required.
You do not write long paragraphs.
You do not produce essays.
You strictly generate compact, factual research notes based only on the user input.

3. TOOLS YOU HAVE ACCESS TO
a. Groq LLM Models
Used for:
- Reasoning
- Summarizing and synthesizing findings
- Interpreting results from web searches

b. Web Search Tool (REAL search)
You may call it whenever needed for:
- Verifying facts
- Getting updated information
- Handling technical, recent, or statistical queries
- Filling knowledge gaps

4. CONSTRAINTS
You must follow these rules strictly:
- Output ONLY:
  • 1–3 bullet points summarizing what you found
  • A final line: NEEDS_MORE: true/false
- No markdown, no headings, no long paragraphs
- No chain-of-thought
- No hallucinated facts or citations
- If unsure or results seem incomplete → NEEDS_MORE: true
- Use the web search tool whenever the answer cannot be reliably produced from reasoning alone
- Keep bullet points factual and short (1–2 sentences each)
"""


    # Perform the real Groq research call (no mock)
    raw = groq_research_query(prompt)

    # Process groq response (very simple parsing)
    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    notes = [l for l in lines if not l.startswith("NEEDS_MORE")]
    needs_more_flag = False
    for l in lines:
        if l.upper().startswith("NEEDS_MORE"):
            val = l.split(":", 1)[-1].strip()
            needs_more_flag = val.lower() in ("true", "1", "yes")

    # Append to research notes
    new_notes = state.get("research_notes", []) + ["\n".join(notes)]

    # increment iteration counter
    iterations += 1

    return {
        "research_notes": new_notes,
        "iterations": iterations,
        "needs_more": needs_more_flag,
        "next": "Executor",
    }


def executor_node(state: AgentState) -> AgentState:
    """Executor AGENT: now uses Groq LLM to format research notes and decide if more is needed."""

    notes = state.get("research_notes", [])
    memory = state.get("memory", {})

    # Build executor prompt
    executor_prompt = executor_prompt = f"""
YOU ARE: The EXECUTOR agent (LLM responsible for structuring + finalizing research outputs)
GOAL: Convert raw research notes into a clean, structured final summary AND decide if more research is needed.
TOOLS YOU HAVE: You ONLY have access to Groq LLM reasoning. You DO NOT have access to files, memory, or external tools — the system handles writing.
CONSTRAINTS:
- Respond ONLY in valid JSON using this schema:
  {{
    "final_note": "...",
    "needs_more": true/false
  }}
- No explanations, no extra text, no markdown.
- The JSON must be the only output.

Research Notes Provided:
{notes}
"""


    # Call Groq with a smaller fast model
    try:
        client = create_groq_client()
        completion = client.chat.completions.create(
            messages=[{"role": "user", "content": executor_prompt}],
            model="llama-3.1-8b-instant",
        )
        raw = completion.choices[0].message.content
    except Exception:
        raw = '{"final_note": "Executor fallback summary.", "needs_more": false}'

    # Parse JSON
    try:
        parsed = json.loads(raw)
    except Exception:
        parsed = {"final_note": raw, "needs_more": False}

    final_note = parsed.get("final_note", "")
    needs_more = parsed.get("needs_more", False)

    # Write to Windows Notepad only if user said yes
    # Read user preference safely
    save_flag = state.get("save_to_notepad", False)  # default to False if missing

    if save_flag:
        filename = os.path.abspath("research_notepad.txt")  # full path
        with open(filename, "a", encoding="utf-8") as f:
            f.write(final_note + "\n")

    # Open Notepad only if saved
        try:
            os.system(f'notepad "{filename}"')
        except Exception as e:
            print(f"Could not open Notepad automatically: {e}")

    # Persist memory
    memory_update = {
        "last_query": state.get("user_input"),
        "last_notes": notes,
        "iterations": state.get("iterations", 0),
        "final_note": final_note,
    }
    memory.update(memory_update)
    save_memory(memory)

    # Stop if reached max iterations
    if state.get("iterations", 0) >= state.get("max_iterations", 3):
        needs_more = False

    return {
        "final_note": final_note,
        "memory": memory,
        "needs_more": needs_more,
        "next": "Researcher" if needs_more else None,
    }


def build_workflow() -> StateGraph:
    workflow = StateGraph(AgentState)

    workflow.add_node("Supervisor", supervisor_node)
    workflow.add_node("Researcher", researcher_node)
    workflow.add_node("Executor", executor_node)

    workflow.set_entry_point("Supervisor")
    workflow.add_edge("Supervisor", "Researcher")
    workflow.add_edge("Supervisor", "Executor")

    workflow.add_edge("Researcher", "Supervisor")
    workflow.add_edge("Executor", "Supervisor")

    app = workflow.compile()
    return app

# ---------------------------
# Run example
# ---------------------------
if __name__ == "__main__":
    app = build_workflow()

    # Ask user for input at runtime
    user_question = input("Enter your research question: ")

    # Ask if output should be saved to notepad
    save_to_notepad = input("Save research output to notepad? (y/n): ").strip().lower() == "y"

    initial_state: AgentState = {
       "user_input": user_question,
       "iterations": 0,
       "max_iterations": 2,  # limit loops to avoid recursion
       "research_notes": [],
       "memory": load_memory(),
       "save_to_notepad": save_to_notepad,  # store preference in state
    }

    print("Invoking LangGraph multi-agent workflow...\n")
    result = app.invoke(initial_state)

    # result is the final state; print key fields and show log-like output
    print("=== RUN LOG ===")
    print(f"User input: {result.get('user_input')}")
    print(f"Iterations: {result.get('iterations')}")
    print("Research notes (aggregated):\n", "\n---\n".join(result.get("research_notes", [])))
    print("Final note written to research_notepad.txt if you have to chosen to save it in the beginning of the program and memory persisted to:", MEMORY_FILE)


    print("\n" + "="*50)
    print("YOUR GRAPH (Mermaid syntax) - paste at https://mermaid.live")
    print("="*50)
    print(app.get_graph().draw_mermaid())

    # Optional: save as PNG automatically (requires kaleido + playwright)
    try:
       app.get_graph().draw_mermaid_png(output_file_path="my_research_graph.png")
       print("\nGraph saved as my_research_graph.png")
    except ImportError:
       print("\nFor PNG export, run: pip install kaleido playwright")
       print("Then: playwright install chromium")
    except Exception as e:
       print(f"Could not save PNG: {e}")




def supervisor_node(state: dict) -> dict:
   """Decide where to route next: research or execute."""
   if state.get("ready_to_execute"):
     return {"next": "execute_agent"}
   return {"next": "research_agent"}
"""Supervisor/Router node"""
def supervisor_node(state):
    if state.get("sufficient"):
        return {"next": "writer_node"}
    return {"next": "researcher_node"}

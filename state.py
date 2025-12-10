from dataclasses import dataclass, field
from typing import List, Any, Dict

@dataclass
class ProjectState(dict):
    query: str = ""
    research_notes: List[str] = field(default_factory=list)
    research_attempts: int = 0
    ready_to_execute: bool = False
    execution_log: List[str] = field(default_factory=list)
    result: str = ""
    final_output: str = ""
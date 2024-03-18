from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List, Tuple, Annotated, TypedDict
import operator
from agents.planner import get_planner, get_replanner
from agents.executor import get_executor
# Define the State
class State(TypedDict):
    input: str
    plan: List[str]
    past_steps: Annotated[list[Tuple], operator.add]
    response: str
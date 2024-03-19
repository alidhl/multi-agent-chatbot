from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List, Tuple, Annotated, TypedDict
import operator
from agents.planner import get_planner, get_replanner
from agents.executor import get_executor
from agents.planner import Response
from langgraph.graph import StateGraph, END
import asyncio
# Define the State
class State(TypedDict):
    input: str
    plan: List[str]
    past_steps: Annotated[List[Tuple], operator.add]
    response: str
    
# Create agents
executor = get_executor()
planner = get_planner()
replanner = get_replanner()

# Create the graph

async def execute(state: State):
    task = state['plan'][0]
    output = await executor.ainvoke({'input': task, "chat_history" : []})
    return {"past_steps" : (task, output['agent_outcome'].return_values['output'])}

async def plan(state: State):
    plan = await planner.ainvoke({'objective': state['input']})
    return {"plan" : plan.steps}

async def replan(state: State):
    output = await replanner.ainvoke(state)
    # If the output is a reponse (the plan is complete), then return the response else update the plan
    if isinstance(output, Response):
        return {"response" : output.response}
    else:
        return {"plan" : output.steps}

def should_end(state: State):
    if (state['response']):
        return True
    else:
        return False
    
graph = StateGraph(State)
graph.add_node("planner", plan)
graph.add_node("executor", execute)
graph.add_node("replanner", replan)

graph.set_entry_point("planner")
graph.add_edge("planner", "executor")
graph.add_edge("executor", "replanner")
graph.add_conditional_edges(
    'replanner',
    should_end,
    {
        True: END,
        False: "executor"
    })

def get_graph():
    return graph.compile()

# Test the graph
async def main():
    g = get_graph()
    query = "When was Saudi Arabia The Line project announced?"
    async for event in g.astream({'input' : query}):
        for k, v in event.items():
            if k != "__end__":
                print(v)

if __name__ == "__main__":    
    asyncio.run(main())

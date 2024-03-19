from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.openai_functions import create_structured_output_runnable
from langchain.chains.openai_functions import create_openai_fn_runnable
from langchain_openai import ChatOpenAI
class Planner(BaseModel):
    """Plan the execution of the agent"""
    
    steps: list[str] = Field(
        description="The steps to execute the agent, should be in sorted order"
    )


planner_prompt = ChatPromptTemplate.from_template(
    """For the given objective, come up with a simple step by step plan. \
This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.

{objective}"""
)

def get_planner():
    return create_structured_output_runnable(
        output_schema= Planner,
        llm = ChatOpenAI(model = "gpt-4-0125-preview"),
        prompt = planner_prompt
    )
    
# Replanner for updating the plan or returning a response

class Response(BaseModel):
    """Response to the query"""
    
    response: str
    
    
# Inside the state will be four fields: input , plan , past_steps , response     
replanner_prompt = ChatPromptTemplate.from_template(
    """For the given objective, come up with a simple step by step plan. \
This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.

Your objective was this:
{input}

Your original plan was this:
{plan}

You have currently done the follow steps:
{past_steps}

Update your plan accordingly. If no more steps are needed and you can return to the user, then respond with that. Otherwise, fill out the plan. Only add steps to the plan that still NEED to be done. Do not return previously done steps as part of the plan."""
)

def get_replanner():
    return create_openai_fn_runnable(
        functions= [Planner, Response],
        llm = ChatOpenAI(model = "gpt-4-0125-preview"),
        prompt = replanner_prompt
    )


# Testing
if __name__ == "__main__":
    query = "When was Saudi Arabia The Line project announced?"
    planner = get_planner()
    replanner = get_replanner()
    
    print(query + "\n\n")
    print(planner.invoke({"objective" : query}))
    print("-"*50)
    print(planner.invoke({"objective" : query}).steps)
    
    
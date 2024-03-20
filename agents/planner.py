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

def get_planner():
    planner_prompt = ChatPromptTemplate.from_template(
        """For the given objective, come up with a simple and concise step by step plan. \
    This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
    The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.

    {objective}"""
    )
    return create_structured_output_runnable(
        output_schema= Planner,
        llm = ChatOpenAI(model = "gpt-4-0125-preview"),
        prompt = planner_prompt
    )
    
# Replanner for updating the plan or returning a response

class Response(BaseModel):
    """Response to user"""
    
    response: str
    
    
# Inside the state will be four fields: input , plan , past_steps , response     

def get_replanner():
    replanner_prompt = ChatPromptTemplate.from_template(
    """You are a replanner agent. Your role is to update the execution plan of another agent to achieve a given objective, step by step. If the answer to the objective is already found in the past steps, your task is to return that answer.\

    Here is your objective:
    {input}

    Here is your current plan:
    {plan}

    These are the steps that have been executed so far:
    {past_steps}

    Based on the execution of the plan so far, your task is to update the plan by removing the completed steps. Ensure that the remaining steps in the plan will lead to the answer for the objective. Remember, make sure to answer the objective fully not just part of it."""
    )
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
    
    
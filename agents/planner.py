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
"""Given the objective, devise a simple and concise step-by-step plan that involves using a search engine to find the answer. \
This plan should consist of individual search queries that, if executed correctly, will yield the correct answer. \
Avoid unnecessary steps and aim to make the plan as short as possible. The result of the final step should be the final answer. \

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
    """Your task is to revise the current plan based on the executed steps. Remove any steps that have been completed and ensure the remaining steps will lead to the complete answer for the objective. Remember, the objective should be fully answered, not just partially. If the answer is already found in the executed steps, return the answer to the objective.

Objective:
{input}

Current Plan:
{plan}

Executed Steps:
{past_steps}

"""
    )
    return create_openai_fn_runnable(
        functions= [Planner, Response],
        llm = ChatOpenAI(model = "gpt-4-0125-preview"),
        prompt = replanner_prompt
    )


# Testing
if __name__ == "__main__":
    query = "How do I get a passport in saudi arabia"
    planner = get_planner()
    replanner = get_replanner()
    
    print(query + "\n\n")
    print(planner.invoke({"objective" : query}).steps)
    
    
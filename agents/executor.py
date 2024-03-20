from langchain import hub
from langchain.agents import create_openai_functions_agent
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import create_agent_executor

def get_executor():
    tools = [TavilySearchResults(max_results=3)]
    # Basic prompt for the executer
    prompt = hub.pull("hwchase17/openai-functions-agent")
    print(prompt)
    llm = ChatOpenAI(model = "gpt-3.5-turbo")
    agent = create_openai_functions_agent(llm, tools, prompt)
    agent_executor = create_agent_executor(agent, tools=tools)
    return agent_executor
# Test the agent executor
if __name__ == "__main__":
    executor = get_executor()
    query = "When was Saudi Arabia The Line project announced?"
    print(query + "\n\n")
    print(executor.invoke({"input" : query , "chat_history" : []}))

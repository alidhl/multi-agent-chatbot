# Multi-Agent Chatbot

This project presents a multi-agent chatbot system integrated with a search engine, designed to handle complex user queries with a systematic approach. It leverages the capabilities of LangChain and LangGraph libraries, and Tavily for the search engine functionality. The chatbot uses the "plan-and-execute" coordination strategy among its agents to ensure effective and efficient responses.

## Demo

Check out the live demo of the multi-agent chatbot system: [Multi-Agent Chatbot Demo](https://multi-agent-chatbot.streamlit.app/)

## Features

- **Plan-and-Execute Agent Coordination:** The system incorporates three specialized agents:
  - **Planner:** Begins by formulating a structured plan based on the user's query.
  - **Executor:** Follows the defined plan, executing the necessary steps one by one.
  - **Replanner:** Monitors the execution process, evaluates its effectiveness, and decides whether the current plan suffices or needs adjustments to better address the user's query.

- **Integration with LangChain and LangGraph:** Utilizes the advanced capabilities of these libraries to understand and navigate complex language constructs.

- **Search Engine Integration:** Powered by Tavily, the system includes a robust search engine to fetch relevant information and support the chatbot's responses.

- **Streamlit Web Interface:** The project uses Streamlit for designing and deploying an intuitive web interface, enhancing user interaction and experience.

## How to Use Locally

1. **Clone Repo**

2. **Install requirements.txt**
```bash
pip install -r requirements.txt
```

3. **Run main.py via Streamlit**
```bash
streamlit run main.py
```

## Built With

- [LangChain](https://langchain.com)
- [LangGraph](https://langgraph.com)
- [Tavily](https://tavily.com)
- [OpenAI](https://openai.com)
- [Streamlit](https://streamlit.io)

## License

This project is licensed under the MIT License.

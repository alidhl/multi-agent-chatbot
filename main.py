from graph import get_graph
import streamlit as st
import os

with st.sidebar:
    st.sidebar.title("API Keys")
    openai_api_key = st.sidebar.text_input("OpenAI API Key", key= "chatbot_api_key" , type="password")
    #taveliy_api_key = st.sidebar.text_input("Taveliy API Key", type="password")



st.title("💬Multi-Agent Search Engine Chatbot")
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    if not openai_api_key:
        st.info("Please add your OpenAI API to continue.")
        st.stop()
    else:
        start_chat = st.button('Start Chat')
        if start_chat:
            os.environ['OPENAI_API_KEY'] = openai_api_key
            # Will be used in the future for now will use my own API key
            #os.environ['TAVELIY_API_KEY'] = taveliy_api_key
            graph = get_graph()
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)
            response = graph.invoke({"input": st.session_state.messages[-1]})
            msg = response["response"]
            st.session_state.messages.append({"role": "assistant", "content": msg})
            st.chat_message("assistant").write(msg)




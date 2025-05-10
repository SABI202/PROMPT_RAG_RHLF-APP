import streamlit as st
import pickle
import os
import csv
from typing import Optional

# Cache the model load
@st.cache_resource
def load_query_engine():
    with open('rag_model.pkl', 'rb') as f:
        return pickle.load(f)

# Cache feedback storage
@st.cache_data(max_entries=100)
def store_feedback(question: str, response: str, feedback: str):
    file_exists = os.path.exists("feedback_log.csv")
    with open("feedback_log.csv", "a", newline='', encoding='utf-8') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        if not file_exists:
            writer.writerow(["User Question", "Assistant Response", "Feedback"])
        writer.writerow([question, response, feedback])

# Initialize app state
def init_state():
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "How can I help you?"}]
    if "feedback_submitted" not in st.session_state:
        st.session_state.feedback_submitted = False

# Load components
init_state()
query_engine = load_query_engine()

# UI Elements (static parts)
with st.sidebar:
    st.write("Query Engine Chatbot")
    st.markdown("[View the source code](https://github.com/streamlit/llm-examples/blob/main/Chatbot.py)")

st.title("💬 Query Engine Chatbot")
st.caption("🚀 A Streamlit chatbot powered by a precomputed index")

# Display chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Chat input
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    with st.spinner("Thinking..."):
        response = query_engine.query(prompt)
        msg = str(response)
        
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)
    st.session_state.feedback_submitted = False

# Feedback handling
if (not st.session_state.feedback_submitted and 
    len(st.session_state.messages) > 1 and 
    st.session_state.messages[-1]["role"] == "assistant"):
    
    feedback = st.radio(
        "How was the answer?",
        ["👍 Good", "👎 Bad"],
        key="feedback_radio",
        index=None
    )
    
    if feedback:
        store_feedback(
            st.session_state.messages[-2]["content"],
            st.session_state.messages[-1]["content"],
            feedback
        )
        st.session_state.feedback_submitted = True
        st.rerun()

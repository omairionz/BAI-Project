import streamlit as st
import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import re
import time

load_dotenv()

CHROMA_PATH = "chroma-embeddings"

# -------- SAFETY CHECK --------
if not os.getenv("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY is missing from your .env file")
    st.stop()

# -------- PAGE CONFIG --------
st.set_page_config(
    page_title="UVA Computer Science Advisor",
    page_icon="🎓",
    layout="centered"
)

# -------- SESSION STATE --------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "major" not in st.session_state:
    st.session_state.major = None

if "completed_classes" not in st.session_state:
    st.session_state.completed_classes = []

# -------- SIDEBAR --------
with st.sidebar:
    st.header("Your Profile")

    major_options = ["", "BACS", "BSCS"]
    current_major = st.session_state.get("major", None)
    index = major_options.index(current_major) if current_major in major_options else 0

    selected_major = st.selectbox("Select your major:", major_options, index=index)

    if selected_major != st.session_state.major:
        st.session_state.major = selected_major if selected_major else None
        st.rerun()

    st.subheader("Completed Classes")
    new_class = st.text_input("Add a completed class (e.g., CS 1110):")

    if st.button("Add Class"):
        if new_class and new_class.upper() not in st.session_state.completed_classes:
            st.session_state.completed_classes.append(new_class.upper())
            st.rerun()

    for cls in st.session_state.completed_classes:
        col1, col2 = st.columns([4, 1])
        with col1:
            st.write(f"• {cls}")
        with col2:
            if st.button("❌", key=cls):
                st.session_state.completed_classes.remove(cls)
                st.rerun()

# -------- HEADER --------
st.title("🎓 UVA Computer Science Advisor")
st.divider()

# -------- CHAT HISTORY --------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -------- PROMPT --------
PROMPT_TEMPLATE = """
You are a UVA CS academic advisor.

Context:
{context}

History:
{history}

Question:
{question}
"""

prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

# -------- DB --------
@st.cache_resource
def load_db():
    return Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=OpenAIEmbeddings()
    )

db = load_db()

def extract_course_code(text):
    match = re.search(r"\bCS\s*([0-9]{4})\b", text.upper())
    return f"CS{match.group(1)}" if match else None

# -------- CORE LOGIC --------
def ask_question(query_text):

    with st.spinner("Searching knowledge base..."):
        results = db.similarity_search_with_relevance_scores(query_text, k=4)

    if not results or results[0][1] < 0.7:
        return "I couldn't find relevant information in the database."

    context = "\n\n---\n\n".join([d.page_content for d, _ in results])

    history = "\n".join(
        [f"{m['role']}: {m['content']}" for m in st.session_state.messages[-4:]]
    )

    prompt = prompt_template.format(
        context=context,
        history=history,
        question=query_text
    )

    model = ChatOpenAI(timeout=20)

    with st.spinner("Thinking..."):
        response = model.invoke(prompt)

    return response.content

# -------- INPUT --------
user_input = st.chat_input("Ask about UVA Computer Science...")

if user_input:

    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    answer = ask_question(user_input)

    # -------- SAFE TYPING ANIMATION --------
    with st.chat_message("assistant"):
        placeholder = st.empty()
        typed_text = ""

        for char in answer:
            typed_text += char
            placeholder.markdown(typed_text + "▌")
            time.sleep(0.01)

        placeholder.markdown(typed_text)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )

# -------- INTRO --------
if len(st.session_state.messages) == 0:
    st.markdown(
        "<h3 style='text-align:center;color:gray;'>Ask me anything about UVA Computer Science.</h3>",
        unsafe_allow_html=True
    )
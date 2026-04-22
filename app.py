import streamlit as st
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from datetime import datetime
from textwrap import fill
import time

load_dotenv()

CHROMA_PATH = "chroma-embeddings"

#--------------------------------- IGNORE ----------------------------------------------------------------------------
# -------- PAGE CONFIG --------
#st.set_page_config(page_title="UVA Engineering Advisor", page_icon="🎓")

# -------- SESSION STATE INIT --------
#if "messages" not in st.session_state:
   # st.session_state.messages = []

#if "chat_started" not in st.session_state:
    #st.session_state.chat_started = False

# -------- HEADER --------
#st.title("🎓 UVA Engineering Advisor", text_alignment="center")

# -------- CLEAR CHAT BUTTON --------
#if st.button("🗑️ Clear Chat"):
    #st.session_state.messages = []
    #st.session_state.chat_started = False
    #st.rerun()

# -------- SUBHEADER INTRO (DISAPPEARS AFTER FIRST MESSAGE) --------
#if not st.session_state.chat_started:
    #st.markdown(
        #"""
       # <h3 style='text-align: center; color: gray; opacity: 0.95;'>
         #   Ask me anything about UVA Engineering.
      #  </h3>
       # """,
      #  unsafe_allow_html=True
   # )

#-----------------------------------------------------------------------------------------------------------------------
# -------- PAGE CONFIG --------
st.set_page_config(
    page_title="UVA Computer Science Advisor",
    page_icon="🎓",
    layout="centered"
)

# -------- SIDEBAR FOR USER PROFILE --------
with st.sidebar:
    st.header("Your Profile")
    
    # Major selection
    major_options = ["", "BACS", "BSCS"]
    selected_major = st.selectbox(
        "Select your major:",
        major_options,
        index=major_options.index(st.session_state.major) if st.session_state.major in major_options else 0
    )
    if selected_major != st.session_state.major:
        st.session_state.major = selected_major if selected_major else None
        st.rerun()
    
    # Completed classes
    st.subheader("Completed Classes")
    new_class = st.text_input("Add a completed class (e.g., CS 1110):", key="new_class_input")
    if st.button("Add Class"):
        if new_class and new_class not in st.session_state.completed_classes:
            st.session_state.completed_classes.append(new_class.upper())
            st.rerun()
    
    # Display completed classes
    if st.session_state.completed_classes:
        st.write("Completed Classes:")
        for cls in st.session_state.completed_classes:
            col1, col2 = st.columns([4, 1])
            with col1:
                st.write(f"• {cls}")
            with col2:
                if st.button("❌", key=f"remove_{cls}"):
                    st.session_state.completed_classes.remove(cls)
                    st.rerun()
    else:
        st.write("No completed classes added yet.")

# -------- SESSION STATE INIT --------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "major" not in st.session_state:
    st.session_state.major = None

if "completed_classes" not in st.session_state:
    st.session_state.completed_classes = []

if "major" not in st.session_state:
    st.session_state.major = None

if "completed_classes" not in st.session_state:
    st.session_state.completed_classes = []

# -------- HEADER + CLEAR BUTTON ROW --------
header_col, button_col = st.columns([6, 1])

with header_col:
    st.markdown(
        """
        <h1 style='margin-bottom: 0; color: #FFFFFF;'>
            🎓 UVA Computer Science Advisor
        </h1>
        """,
        unsafe_allow_html=True
    )

with button_col:
    st.markdown("<div style='height: 27px;'></div>", unsafe_allow_html=True)
    if st.button("🗑️ Clear"):
        st.session_state.messages = []
        st.rerun()

# -------- DIVIDER --------
st.divider()



#-----------------------------------------------------------------------------------------------------------------------


# -------- DISPLAY CHAT HISTORY --------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# -------- PROMPT TEMPLATE --------
PROMPT_TEMPLATE = """
You are a helpful UVA Computer Science Academic Advisor.
Answer clearly and concisely in plain language.
If information is missing, say so honestly.
Do not fabricate requirements.
Have a warm, welcoming personality.
Help guide users to next question and if user says yes or wants that suggested question to be answered, answer it.
Give users proper contact information if they ask.

User Profile:
Major: {major}
Completed Classes: {completed_classes}

Chat History:
{history}

Context:
{context}

---

Question:
{question}
"""

prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

# -------- LOAD VECTOR DATABASE --------
@st.cache_resource
def load_db():
    embedding_function = OpenAIEmbeddings()
    return Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_function
    )

# -------- RAG FUNCTION --------
def ask_question(query_text):
    db = load_db()
    results = db.similarity_search_with_relevance_scores(query_text, k=4)

    if len(results) == 0 or results[0][1] < 0.7:
        return "I couldn't find relevant information in the database."

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
    history_text = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[-4:]])

    # Format user profile
    major_text = st.session_state.major if st.session_state.major else "Not specified"
    completed_classes_text = ", ".join(st.session_state.completed_classes) if st.session_state.completed_classes else "None specified"

    prompt = prompt_template.format(
        context=context_text,
        question=query_text,
        history=history_text,
        major=major_text,
        completed_classes=completed_classes_text
    )

    model = ChatOpenAI()
    response = model.invoke(prompt)

    return response.content

# -------- USER INPUT --------
user_input = st.chat_input("Ask about UVA Computer Science...")

if user_input:

    # Store user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    # Get assistant response
    answer = ask_question(user_input)

    # Typing animation
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        for char in answer:
            full_response += char
            message_placeholder.markdown(full_response + "▌")
            time.sleep(0.01)
        message_placeholder.markdown(full_response)

    # Store assistant response
    st.session_state.messages.append({"role": "assistant", "content": answer})

     # sets timestamp format
    timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")

    wrapped_user = fill(user_input, width=80)
    wrapped_response = fill(full_response, width=80)

    # append query_text and response into log.txt
    with open(file="log.txt", mode="a", encoding="utf-8") as f:
        f.write(f"{timestamp} User: {wrapped_user}\n\n")
        f.write(f"{timestamp} Assistent: {wrapped_response}\n\n")
        f.write("-" * 60 + "\n\n")

# -------- INTRO TEXT (DISAPPEARS AFTER FIRST MESSAGE) --------
if len(st.session_state.messages) == 0:
    st.markdown(
        """
        <h3 style='text-align: center; color: gray; opacity: 0.95;'>
            Ask me anything about UVA Computer Science.
        </h3>
        """,
        unsafe_allow_html=True
    )

st.markdown(
    """
    <script>
        setTimeout(function() {
            const chatContainer = window.parent.document.querySelector('.main');
            chatContainer.scrollTo(0, chatContainer.scrollHeight);
        }, 200);
    </script>
    """,
    unsafe_allow_html=True
)
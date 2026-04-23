import streamlit as st
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from datetime import datetime
from textwrap import fill
import time
from course_tools import check_prereqs, check_degree_requirements, get_course_difficulty
import re

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
<<<<<<< Updated upstream
You are the "Young Alum Mentor" for UVA Computer Science students. 
You graduated from the E-School recently and know the 'real' deal behind the requirements.

STANCE & TONE:
- Be a mix of calm expertise and radical transparency.
- Use UVA lingo naturally (e.g., Grounds, Thornton Hall, the E-School, BACS vs BSCS).
- Don't just list classes; explain the STRATEGY (e.g., "Don't take OS and Algo together unless you want no sleep").

CONSTRAINTS:
- Only answer based on the provided Context and Chat History.
- If the Context doesn't have the answer, say "Honestly, I don't have the data on that class—check the Undergraduate Record to be safe."
- Stay focused on CS requirements. If they ask about other majors, steer them back.

LOCAL_LINGO_GUIDE:
- Use "Grounds" instead of "Campus."
- Use "First-year/Second-year" instead of "Freshman/Sophomore."
- Refer to the Engineering school as "The E-School."
- When mentioning registration, refer to "SIS" or "Hoos' List."
- Mention "Thornton Hall" or "Rice Hall" as the heart of CS life. 

CORE REASONING PROTOCOL:
1. CHECK PREREQS: Before recommending any 3000-level or higher CS course, explicitly check if the user has mentioned completing the 2000-level core (CS 2100, 2120, 2130). If they haven't, warn them about the "gatekeepers."
2. WORKLOAD REALITY CHECK: 
   - If a course is labeled in the Context as "High Difficulty" (e.g., OS, CompArch, or Algo), tell the user: "Heads up, this is a heavy-hitter. Don't stack this with another heavy coding class."
   - If a course is a "GPA Booster" or "Manageable," suggest it as a pairing for a harder requirement.
3. BACS vs BSCS: If the user hasn't specified their track, ask them! The advice for a BA in the College is different than a BS in the E-School regarding Integration Electives and Math.

EXAMPLE_INTERACTIONS:

User: "I'm a first-year thinking about taking CS 2100 next semester. Is it hard?"
Mentor: "Welcome to the gauntlet! CS 2100 (DSA) is the legendary gatekeeper of the major. It's not just 'hard'—it's a massive time sink. But honestly? It's where you actually learn to be a programmer. If you take it, pair it with a light gen-ed so you have time to live in Rice Hall during office hours. You got this."

User: "Can I take OS and CompArch at the same time?"
Mentor: "Short answer: Please don't. Long answer: That's a 'no-sleep' semester. Both are heavy-hitters with massive coding loads. Unless you're trying to speedrun burnout, I’d suggest taking one now and saving the other for a semester where your other classes are chill. Radical transparency: your GPA (and sanity) will thank you."

STUDENT CAPACITY ASSESSMENT:
1. GAUGE EXPERIENCE: If a user asks about a class, check if they have a strong foundation. (e.g., "How comfortable are you with C++ or Java?").
2. LOAD TOLERANCE: Explicitly ask the user about their outside commitments if they are planning a heavy semester. 
   - Rule: If a user is planning more than two 'Heavy' courses, ask: "Are you doing research, a part-time job, or a heavy extracurricular load this semester?"
3. EMOTIONAL INTELLIGENCE: 
   - If a student sounds overwhelmed, prioritize "The Balanced Path" (recommend easy electives or "GPA boosters").
   - If a student sounds like a "High-Achiever/Fast-Tracker," provide the "Aggressive Path" but flag the peak stress points.

ADAPTIVE INTERACTION RULES:
- If the user uses words like "stressed," "overwhelmed," or "scared," lower the 'Workload Threshold' for recommendations and prioritize 'Chill' electives.
- Before confirming a 3+ CS course semester, MANDATORILY ask: "What's your 'bandwidth' like outside of class? Are you working, in a heavy club, or just want a social life?"
- If the user is a 'Fast-Tracker' (wants to graduate early), flag the 'Burnout Zones' (e.g., Spring of Second Year for BSCS students).

CONVERSATION STEERING & CTAs:
- Never end a response with "How can I help you?" 
- Instead, suggest a logical next step based on the context:
    * If you just discussed a hard class: "Want me to look for some lighter electives to pair that with so you're not overwhelmed?"
    * If you just discussed prerequisites: "Should we look at the 'Golden Path' for your next three semesters to make sure you're on track for graduation?"
    * If the user is a BACS student: "Since you're in the College, do you want to talk about which 'Integration Electives' actually overlap with CS interests?"
- PROACTIVE CHECK-IN: Every 3-4 messages, ask: "Just checking in—does this schedule feel manageable to you, or are we pushing the limit a bit?"

>>>>>>> Stashed changes

Context:
{context}

Chat History:
{history}

User Question: {question}
"""

prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

# -------- ROUTER HELPERS --------
def route_question(query_text):
    text = query_text.lower()

    requirements_keywords = [
        "requirement", "requirements", "prereq", "prerequisite",
        "need", "graduate", "graduation", "major", "degree",
        "bacs", "bscs", "eligible", "can i take", "what classes",
        "what class", "track"
    ]

    easiness_keywords = [
        "easy", "easier", "easiest", "hard", "harder", "hardest",
        "difficulty", "difficult", "workload", "manageable",
        "stressful", "time-consuming", "heavy"
    ]

    for word in easiness_keywords:
        if word in text:
            return "easiness"

    for word in requirements_keywords:
        if word in text:
            return "requirements"

    return "general"


def extract_course_code(query_text):
    match = re.search(r"\bCS\s*([0-9]{4})\b", query_text.upper())
    if match:
        return f"CS{match.group(1)}"
    return None

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

# -------- MAIN QUESTION LOGIC --------
def ask_question(query_text):
    route = route_question(query_text)
    course_code = extract_course_code(query_text)
    lower_query = query_text.lower()

    # -------- REQUIREMENTS ROUTE --------
    if route == "requirements":
        if course_code and (
            "can i take" in lower_query
            or "eligible" in lower_query
            or "prereq" in lower_query
            or "prerequisite" in lower_query
        ):
            prereq_result = check_prereqs(course_code, st.session_state.completed_classes)

            if prereq_result["eligible"]:
                return (
                    f"Based on the classes you entered, you appear eligible to take {course_code}. "
                    f"You have all listed prerequisites in the current tool."
                )
            else:
                missing = ", ".join(prereq_result["missing_prereqs"])
                return (
                    f"Based on the classes you entered, you are not yet ready for {course_code}. "
                    f"You are still missing: {missing}."
                )

        if st.session_state.major and (
            "requirement" in lower_query
            or "requirements" in lower_query
            or "graduate" in lower_query
            or "graduation" in lower_query
            or "degree" in lower_query
        ):
            degree_type = "BA" if st.session_state.major == "BACS" else "BS"
            degree_result = check_degree_requirements(
                st.session_state.completed_classes,
                degree_type
            )

            if "error" not in degree_result:
                if degree_result["eligible"]:
                    return (
                        f"You appear to have completed the tracked required courses for the "
                        f"{st.session_state.major} path in the current tool."
                    )
                else:
                    missing = ", ".join(degree_result["missing_required_courses"])
                    if missing:
                        return (
                            f"For the {st.session_state.major} track, you are still missing these "
                            f"tracked required courses in the current tool: {missing}."
                        )

    # -------- EASINESS ROUTE --------
    if route == "easiness" and course_code:
        difficulty_result = get_course_difficulty(course_code)
        difficulty = difficulty_result["difficulty"]

        if difficulty != "Unknown":
            return (
                f"{course_code} is labeled as {difficulty} difficulty in the current tool data. "
                f"If you want, I can also help you think about whether it fits with the rest of your schedule."
            )

    # -------- GENERAL / FALLBACK RAG --------
    db = load_db()
    results = db.similarity_search_with_relevance_scores(query_text, k=4)

    if len(results) == 0 or results[0][1] < 0.7:
        return "I couldn't find relevant information in the database."

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
    history_text = "\n".join(
        [f"{m['role']}: {m['content']}" for m in st.session_state.messages[-4:]]
    )

    major_text = st.session_state.major if st.session_state.major else "Not specified"
    completed_classes_text = (
        ", ".join(st.session_state.completed_classes)
        if st.session_state.completed_classes
        else "None specified"
    )

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
import streamlit as st
import random
import datetime
import time

# Motivational Prompts
motivation_prompts = [
    "Believe in yourself! You are capable of amazing things. 🌟",
    "Every day is a new beginning. Take a deep breath and start again. 🌅",
    "Success is the sum of small efforts, repeated daily. 📈",
    "Keep going. Everything you need will come to you at the perfect time. ⏳",
    "Difficulties in life are intended to make us better, not bitter. 💪",
    "You are stronger than you think. Keep pushing forward! 🔥",
    "Your potential is limitless. Never stop exploring your capabilities. 🚀",
    "The only way to achieve the impossible is to believe it is possible. 🎯",
    "Challenges are what make life interesting. Overcoming them is what makes life meaningful. 🏆",
    "You are capable, you are strong, and you can do this! 💖"
]


# Anxiety Relief Prompts
anxiety_relief_prompts = [
    "Take a deep breath. Inhale for 4 seconds, hold for 4, and exhale for 6. 🌬",
    "Close your eyes and picture your happy place. Stay there for a moment. 🌄",
    "Write down what’s bothering you and set it aside for later. ✍",
    "Try progressive muscle relaxation – tense each muscle, then relax it. 🧘",
    "Listen to calming music or nature sounds to ease your mind. 🎵",
    "Step outside and take a short walk to clear your thoughts. 🚶",
    "Drink a warm cup of tea or water. Hydration helps relaxation. 🍵",
    "Focus on the present. What are five things you can see and hear? 👀👂",
    "Talk to someone you trust about what’s making you anxious. 💬",
    "Remind yourself: You have overcome challenges before, and you will again. 🌟"
]


# Study Tips
study_tips = [
    "Use the Pomodoro technique – study for 25 mins, take a 5-min break. ⏳",
    "Teach what you learn to someone else. It helps retain information! 🗣",
    "Summarize notes in your own words to enhance understanding. 📝",
    "Practice active recall – test yourself instead of rereading notes. 📚",
    "Break large tasks into smaller chunks to avoid feeling overwhelmed. 🏗",
    "Use mnemonic devices to memorize complex concepts. 🧠",
    "Find a distraction-free study environment for better focus. 🎧",
    "Use visual aids like mind maps and diagrams to remember better. 🖼",
    "Get enough sleep! Rest is crucial for memory retention. 😴",
    "Stay hydrated and take regular breaks to keep your mind fresh. 💧"
]


# Self-care Tips
self_care_tips = [
    "Take a 5-minute stretch break to ease your muscles. 🏋",
    "Maintain a good posture while studying to avoid back pain. 🪑",
    "Eat brain-boosting foods like nuts, fruits, and dark chocolate. 🍫",
    "Avoid excessive caffeine; try herbal tea instead. ☕",
    "Get sunlight exposure to boost your mood and energy levels. ☀",
    "Set realistic goals and celebrate small achievements. 🎉",
    "Listen to calming music while studying to reduce stress. 🎶",
    "Practice gratitude – write down three things you are grateful for. 💖",
    "Take a deep breath and remind yourself it’s okay to take breaks. 🌿",
    "Limit screen time before bed to ensure better sleep quality. 📵"
]



# Streamlit App Layout
st.set_page_config(page_title="MindEase", layout="wide")
st.title("🌿 Welcome to MindEase")
st.subheader("Your personal companion for motivation, study tips, and self-care.")

# Sidebar
st.sidebar.markdown("""
    <style>
        [data-testid="stSidebar"] {
            background-color: #E6E6FA;
        }
        .sidebar-title {
            color: black;
            font-size: 20px;
            font-weight: bold;
        }
        .sidebar-text {
            color: black;
        }
    </style>
""", unsafe_allow_html=True)

st.sidebar.markdown("<span class='sidebar-title'>💜 MindEase Tools</span>", unsafe_allow_html=True)

if st.sidebar.button("💡 Need a boost? Inspire Me!"):
    st.sidebar.markdown(f"<span class='sidebar-text'>{random.choice(motivation_prompts)}</span>", unsafe_allow_html=True)

if st.sidebar.button("😌 Feeling anxious? Anxiety Relief"):
    st.sidebar.markdown(f"<span class='sidebar-text'>{random.choice(anxiety_relief_prompts)}</span>", unsafe_allow_html=True)

if st.sidebar.button("📖 Study Tips"):
    st.sidebar.markdown(f"<span class='sidebar-text'>{random.choice(study_tips)}</span>", unsafe_allow_html=True)

if st.sidebar.button("💆 Self-care Tips"):
    st.sidebar.markdown(f"<span class='sidebar-text'>{random.choice(self_care_tips)}</span>", unsafe_allow_html=True)

# Timer Functionality with Styled Sidebar
st.sidebar.markdown("<span class='sidebar-title'>⏳ Study Timer</span>", unsafe_allow_html=True)
timer_input = st.sidebar.number_input(
    "Set timer (minutes):", min_value=1, max_value=120, value=25, step=1, key="timer_input"
)
if st.sidebar.button("Start Timer", key="start_timer"):
    st.sidebar.markdown(
        f"<span class='sidebar-text'>Timer started for {timer_input} minutes. Stay focused! 🎯</span>",
        unsafe_allow_html=True,
    )

# Study Plan Generator in Sidebar
st.sidebar.markdown("<span class='sidebar-title'>📋 Study Plan Generator</span>", unsafe_allow_html=True)
# Input: Total study time
total_study_time = st.sidebar.number_input(
    "Total study time (hours):", min_value=1, step=1, value=4, key="total_study_time"
)

# Input: Number of subjects
num_subjects = st.sidebar.number_input(
    "Number of subjects:", min_value=1, max_value=10, step=1, value=3, key="num_subjects"
)

# Dynamic subject inputs and difficulty levels
subjects = []
difficulty_levels = []

for i in range(int(num_subjects)):
    subject_name = st.sidebar.text_input(
        f"Enter Subject {i + 1}:", key=f"subject_name_{i}"
    )
    difficulty = st.sidebar.selectbox(
        f"Difficulty for {subject_name or f'Subject {i + 1}'}:",
        options=["Easy", "Medium", "Hard"],
        key=f"difficulty_{i}",
    )
    subjects.append(subject_name)
    difficulty_levels.append(difficulty)

# Generate Study Plan Button
if st.sidebar.button("Generate Study Plan", key="generate_study_plan"):
    if not all(subjects):  # Check for empty subject names
        st.sidebar.error("Please provide names for all subjects.")
    else:
        # Weight Mapping
        difficulty_weights = {"Easy": 1, "Medium": 2, "Hard": 3}
        total_weight = sum(difficulty_weights[difficulty] for difficulty in difficulty_levels)
        total_study_minutes = total_study_time * 60

        # Generate and Display Study Plan
        study_plan = {}
        for i, subject in enumerate(subjects):
            allocated_time = (difficulty_weights[difficulty_levels[i]] / total_weight) * total_study_minutes
            study_plan[subject] = f"{int(allocated_time)} minutes"

        st.sidebar.markdown("**Your Study Plan:**", unsafe_allow_html=True)
        for subject, time in study_plan.items():
            st.sidebar.markdown(f"- **{subject}:** {time}", unsafe_allow_html=True)



# Emotion-Based Prompt System
st.subheader("How are you feeling today?")
total_emotion_prompts = {
    "Happy": ["Keep spreading the joy! Happiness is contagious. 😊", "Enjoy the little things in life! 🌼", "Happiness shared is happiness doubled. 💕", "Smile more, worry less. 😊", "Every moment is a fresh beginning. 🌟"],
    "Sad": ["It’s okay to feel sad. Take it one step at a time. 💙", "This too shall pass. 🌈", "You are stronger than you think. 💪", "Tough times never last, but tough people do. 🏆", "Let yourself feel, then let yourself heal. 💖"],
    "Anxious": anxiety_relief_prompts,
    "Motivated": ["Keep up the great work! Channel your motivation into your goals. 🚀", "Your hard work will pay off. 💰", "Success is built on consistency. 📈", "You’ve got this! 🔥", "Dream big, work hard, stay focused. 🎯"],
    "Frustrated": ["Take a deep breath. A short break might help clear your mind. 🌬", "Let go of what you can’t control. 🍃", "One step at a time, you’ll get there. 🏁", "Obstacles are stepping stones to growth. ⛰", "Frustration is temporary, progress is permanent. 💡"],
    "Tired": ["Rest is just as important as work. Recharge yourself. 💤", "Listen to your body, it’s okay to slow down. 🛌", "Energy comes from balance. Take care of yourself. ⚖", "Small breaks can make a big difference. ⏳", "You deserve to rest and recover. 🌙"]
}

emotion = st.selectbox("Select your emotion:", list(total_emotion_prompts.keys()))
st.write(random.choice(total_emotion_prompts[emotion]))

# Daily Affirmations
st.subheader("✨ Daily Affirmation")
current_date = datetime.datetime.now().day
affirmation = motivation_prompts[current_date % len(motivation_prompts)]
st.markdown(f'<div style="color:black; font-size:18px; padding:10px; border-radius:10px; background-color:#F5F5F5;">{affirmation}</div>', unsafe_allow_html=True)




from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain import vectorstores
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
import os

# Initialize Chatbot
def initialize_llm():
    llm = ChatGroq(
        temperature=0,
        groq_api_key="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",
        model_name="llama-3.3-70b-versatile"
    )
    return llm



def setup_qa_chain(vector_db, llm):
    retriever = vector_db.as_retriever()
    prompt_templates = """You are a compassionate mental health chatbot. Respond thoughtfully to the following questions:
    {context}
    user: {question}
    chatbot: """
    PROMPT = PromptTemplate(template=prompt_templates, input_variables=['context', 'question'])
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT}
    )
    return qa_chain

# Initialize LLM and Vector Database
llm = initialize_llm()

from langchain.vectorstores import Chroma


def create_vector_db():
    # Define embeddings model
    embeddings = HuggingFaceBgeEmbeddings(model_name='all-mpnet-base-v2')

    # Initialize the vector database and specify the directory
    vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    return vector_db


db_path = "./chroma_db"

import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

if not os.path.exists(db_path):
    # Create and initialize the vector database
    vector_db = create_vector_db()
else:
    # Load existing vector database
    embeddings = HuggingFaceBgeEmbeddings(model_name='all-mpnet-base-v2')
    vector_db = vectorstores.Chroma(persist_directory=db_path, embedding_function=embeddings)
qa_chain = setup_qa_chain(vector_db, llm)

# Chatbot Response Logic
def chatbot_response(user_input, history=[]):
    if not user_input.strip():
        return "Please provide a valid input", history
    response = qa_chain.run(user_input)
    history.append((user_input, response))
    return response, history

    
# Initialize Chat History
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# User Input
user_message = st.text_input("You:", placeholder="Ask me anything!", key="user_input")
if user_message:
    bot_response, st.session_state["chat_history"] = chatbot_response(user_message, st.session_state["chat_history"])
    st.session_state["chat_history"].append(("You", user_message))  # Store user message
    st.session_state["chat_history"].append(("Bot", bot_response))  # Store bot response

# Display Chat History
for sender, message in st.session_state["chat_history"]:
    if sender == "You":
        st.markdown(f"**{sender}:** {message}")
    elif sender == "Bot":
        st.markdown(f"🤖 **{sender}:** {message}")
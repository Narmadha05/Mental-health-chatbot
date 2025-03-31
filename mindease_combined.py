import streamlit as st
import random
import datetime

# Motivational Prompts, Study Tips, etc.
motivation_prompts = [
    "Believe in yourself! You are capable of amazing things.",
    "Every day is a new beginning. Take a deep breath and start again."
]
study_tips = [
    "Use the Pomodoro technique – study for 25 mins, take a 5-min break.",
    "Summarize notes in your own words to enhance understanding."
]
self_care_tips = [
    "Take a 5-minute stretch break to ease your muscles.",
    "Eat brain-boosting foods like nuts, fruits, and dark chocolate."
]

# Streamlit App Layout
st.set_page_config(page_title="MindEase", layout="wide")
st.title("🌿 Welcome to MindEase")
st.sidebar.title("MindEase Tools")

from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
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
db_path = "./chroma_db"
if not os.path.exists(db_path):
    vector_db = create_vector_db()
else:
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


# Sidebar Navigation
st.sidebar.title("Navigation")
selected_option = st.sidebar.radio("Go to:", ["Home", "Motivation", "Anxiety Relief", "Study Tips", "Self-Care", "Chatbot"])

if selected_option == "Home":
    st.write("Welcome to MindEase! Your one-stop app for mental wellness and productivity.")

elif selected_option == "Motivation":
    st.write("💡 **Motivational Prompt:**")
    st.write(random.choice(motivation_prompts))

elif selected_option == "Study Tips":
    st.write("📖 **Study Tip:**")
    st.write(random.choice(study_tips))

elif selected_option == "Self-Care":
    st.write("🌸 **Self-Care Tip:**")
    st.write(random.choice(self_care_tips))

elif selected_option == "Chatbot":
    st.write("🤖 **Chat with MindEase Bot:**")
    
    # Initialize Chat History
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    
    # User Input
    user_message = st.text_input("You:", placeholder="Ask me anything!", key="user_input")
    if user_message:
        bot_response, st.session_state["chat_history"] = chatbot_response(user_message, st.session_state["chat_history"])
    # Directly display the updated chat history instead of rerunning the script
    for sender, message in st.session_state["chat_history"]:
        if sender == "You":
            st.write(f"**{sender}:** {message}")
        else:
            st.write(f"🤖 **{sender}:** {message}")
    
    # Display Chat History
    for sender, message in st.session_state["chat_history"]:
        if sender == "You":
            st.write(f"**{sender}:** {message}")
        else:
            st.write(f"🤖 **{sender}:** {message}")
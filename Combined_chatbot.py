import streamlit as st
import random
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain import vectorstores
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
import os

# Motivational Prompts and Other Tips
motivation_prompts = [
    "Believe in yourself! You are capable of amazing things.",
    "Every day is a new beginning. Take a deep breath and start again."
]
anxiety_relief_prompts = [
    "Take a deep breath. Inhale for 4 seconds, hold for 4, and exhale for 6.",
    "Close your eyes and picture your happy place."
]
study_tips = [
    "Use the Pomodoro technique – study for 25 mins, take a 5-min break.",
    "Summarize notes in your own words to enhance understanding."
]
self_care_tips = [
    "Take a 5-minute stretch break to ease your muscles.",
    "Eat brain-boosting foods like nuts, fruits, and dark chocolate."
]

# Initialize the Language Model
def initialize_llm():
    llm = ChatGroq(
        temperature=0,
        groq_api_key="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",
        model_name="llama-3.3-70b-versatile"
    )
    return llm

# Create Vector Database
def create_vector_db():
    os.makedirs('./datas', exist_ok=True)
    loader = DirectoryLoader('./datas', glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    embeddings = HuggingFaceBgeEmbeddings(model_name='all-mpnet-base-v2')
    vector_db = vectorstores.Chroma.from_documents(texts, embeddings, persist_directory='./chroma_db')
    vector_db.persist()
    return vector_db

# Setup QA Chain
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

# Initialize components
llm = initialize_llm()
db_path = "./chroma_db"
if not os.path.exists(db_path):
    vector_db = create_vector_db()
else:
    embeddings = HuggingFaceBgeEmbeddings(model_name='all-mpnet-base-v2')
    vector_db = vectorstores.Chroma(persist_directory=db_path, embedding_function=embeddings)
qa_chain = setup_qa_chain(vector_db, llm)

# Streamlit Interface
st.title("MindEase - Your Mental Health Companion 🌿")
st.sidebar.title("Features")
options = st.sidebar.radio("Choose a feature:", ["Motivation", "Anxiety Relief", "Study Tips", "Self-Care", "Ask a Question"])

# Functionality for each feature
if options == "Motivation":
    st.write("💡 **Motivational Prompt:**")
    st.write(random.choice(motivation_prompts))

elif options == "Anxiety Relief":
    st.write("🕊️ **Anxiety Relief Tip:**")
    st.write(random.choice(anxiety_relief_prompts))

elif options == "Study Tips":
    st.write("📖 **Study Tip:**")
    st.write(random.choice(study_tips))

elif options == "Self-Care":
    st.write("🌸 **Self-Care Tip:**")
    st.write(random.choice(self_care_tips))

elif options == "Ask a Question":
    user_question = st.text_input("What’s on your mind?")
    if st.button("Submit"):
        if user_question.strip():
            response = qa_chain.run(user_question)
            st.write("🤖 **Chatbot Response:**")
            st.write(response)
        else:
            st.write("Please enter a valid question.")
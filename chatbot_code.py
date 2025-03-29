!pip install langchain-groq # Use langchain-groq instead of langchain_groq
!huggingface-cli login # Log in to Hugging Face Hub to access the model
!pip install gradio

import os
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain import vectorstores
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq 
import gradio as gr  
import sys  

def initialize_llm():
    llm = ChatGroq(
        temperature=0,
        groq_api_key="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",
        model_name="llama-3.3-70b-versatile"
    )
    return llm

def create_vector_db():
    # Creating the directory if it doesn't exist
    os.makedirs('/content/datas/', exist_ok=True)  
    
    loader = DirectoryLoader('/content/datas/', glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()
    # Check if any documents were loaded and print the number of documents
    if documents:
        print(f"Loaded {len(documents)} documents.")
    else:
        print("No documents found in the directory.")
        return None  # Return None to indicate failure
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    # Try a different model name that is publicly available
    embeddings = HuggingFaceBgeEmbeddings(model_name='all-mpnet-base-v2')  
    vector_db = vectorstores.Chroma.from_documents(texts, embeddings, persist_directory='./chroma_db')
    vector_db.persist()

    print("chromaDB created and datas saved")
    return vector_db

# The setup_qa_chain function definition needs proper indentation 
def setup_qa_chain(vector_db, llm):
    retriever = vector_db.as_retriever()  
    prompt_templates = """You are a compassionate mental health chatbot.Respond thoughtfully to the following questions:
    {context}
    user:  {question}
    chatbot: """
    PROMPT = PromptTemplate(template=prompt_templates, input_variables=['context', 'question'])
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT}
    )
    return qa_chain


print("Initializing chatbot.........")
llm = initialize_llm()



db_path = "./chroma_db"

if not os.path.exists(db_path):
    vector_db = create_vector_db()
    # Handle the case where create_vector_db returns None
    if vector_db is None:
        print("Error: Failed to create the vector database.")
        sys.exit()  # Exit the script if the database creation fails
else:
    # Use the same model name here as in create_vector_db
    embeddings = HuggingFaceBgeEmbeddings(model_name='all-mpnet-base-v2')
    vector_db = vectorstores.Chroma(persist_directory=db_path, embedding_function=embeddings)  # using vectorstores.Chroma
qa_chain = setup_qa_chain(vector_db, llm)  

def chatbot_response(user_input, history=[]):
    if not user_input.strip():
        return "Please provide a valid input", history
    response = qa_chain.run(user_input) 
    history.append((user_input, response))
    return response, history  

# Create the Gradio interface
with gr.Blocks(theme='Respair/shiki@1.2.1') as demo:
    chatbot = gr.Chatbot()  # Initialize Chatbot component
    msg = gr.Textbox()  # Add a Textbox for user input
    
    # Update chatbot state and clear the input on submit
    def user(user_message, history):
        # Get chatbot response
        bot_message, history = chatbot_response(user_message, history)  
        return "", history + [[user_message, bot_message]]

    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False)  
    
    demo.title = "MindEase"  
    demo.launch(share=True) # Launch and share the app

import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
import os
import time
import json
import datetime


# Theme Configuration
current_hour = datetime.datetime.now().hour
is_daytime = 6 <= current_hour < 18

theme_css = """
    <style>
    .stApp {
        background-color: #A9A9A9;  /* Gray background */
        color: #000000;           /* Black letter */
    }
    h1, h2, h3 { color: #FF0000 !important; }  /* Red titles */
    .stTextInput, .stTextArea {
        background-color: #F8F9FA !important;
        color: #000000 !important;  /* black letter color */
    }
    .stMarkdown, .stMarkdown p {
        color: #000000 !important;  /* AI response black */
    }
    .stFileUploader label {
        color: #000000 !important;  /* Upload letter black */
    }
    .css-1d391kg, .css-1d391kg p {
        color: #000000 !important;  /* Black Letter on Sidebar */
    }
    .stSidebar {
        background-color: #2E3440 !important;  /* Sidebar background gray */
    }
    .stSidebar .stMarkdown, .stSidebar .stMarkdown p {
        color: #FFFFFF !important;  /* White letter on Sidebar */
    }
    </style>
""" if is_daytime else """
    <style>
    .stApp {
        background-color: #2E3440;  /* Gray background */
        color: #00FF00;           /* Green letter */
    }
    h1, h2, h3 { color: #00FF00 !important; }  /* Green titles */
    .stTextInput, .stTextArea {
        background-color: #3B4252 !important;  /*  gray input chatbox */
        color: #00FF00 !important;  /* green letter */
    }
    </style>
"""
st.markdown(theme_css, unsafe_allow_html=True)

# Constants
Pdf_Storage_Path = 'document_store/pdfs/'
Chat_History_Path = 'chat_history.json'

# Model Selection
model_options = ["deepseek-r1:8b", "deepseek-coder-v2:latest", "mistral:latest", "llama3.2:latest", "gemma2:2b", "qwen2.5:3b", "phi3:latest"]
selected_model = st.sidebar.selectbox("Select AI Model", model_options)


Embedding_Model = OllamaEmbeddings(model=selected_model)
Document_Vector_DB = InMemoryVectorStore(Embedding_Model)
Language_Model = OllamaLLM(model=selected_model)

# History loading function
def load_chat_history():
    if os.path.exists(Chat_History_Path):
        with open(Chat_History_Path, 'r') as file:
            return json.load(file)
    return []

# History saving
def save_chat_history(chat_history):
    with open(Chat_History_Path, 'w') as file:
        json.dump(chat_history, file)

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = load_chat_history()

# Sidebar - Chat History & Clear Button
st.sidebar.subheader("📜 Chat History")
if st.sidebar.button("🗑️ Clear Chat"):
    st.session_state.chat_history.clear()
    save_chat_history(st.session_state.chat_history)

for chat in st.session_state.chat_history[-10:]:
    st.sidebar.write(f"👤 **{chat['user']}**")
    st.sidebar.write(f"🤖 {chat['assistant']}")
    st.sidebar.markdown("---")  # Ayrım çizgisi



def ensure_directory_exists(file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

def save_uploaded_file(uploaded_file):
    file_path = Pdf_Storage_Path + uploaded_file.name
    ensure_directory_exists(file_path)
    with open(file_path, "wb") as file:
        file.write(uploaded_file.getbuffer())
    return file_path

def load_pdf_documents(file_path):
    document_loader = PDFPlumberLoader(file_path)
    return document_loader.load()

def chunk_documents(raw_documents):
    text_processor = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    return text_processor.split_documents(raw_documents)

def index_documents(document_chunks):
    Document_Vector_DB.add_documents(document_chunks)

def find_related_documents(query):
    return Document_Vector_DB.similarity_search(query)

def generate_answer(user_query, context_documents):
    context_text = "\n\n".join([doc.page_content for doc in context_documents])
    prompt_template = ChatPromptTemplate.from_template("""
        You are an expert research assistant. Use the provided context to answer the query. 
        If unsure, state that you don't know. Be concise and factual. If user want to you speak Turkish,
        you can use the Turkish language.

        Query: {user_query}
        Context: {document_context}
        Answer:
    """)
    response_chain = prompt_template | Language_Model
    return response_chain.stream({"user_query": user_query, "document_context": context_text})

# UI Design
st.title("🗂️🔍 PDF Whisperer: Ask, Learn, and Discover")
st.markdown("Your Gateway to Smarter Document Handling. 📖")

# File Upload Section
uploaded_pdf = st.file_uploader(
    "📤 Upload, Ask, Discover! ✨",
    type="pdf",
    help="Select a PDF document for analysis",
    accept_multiple_files=True
)

if uploaded_pdf:
    try:
        for pdf_file in uploaded_pdf:
            saved_path = save_uploaded_file(pdf_file)
            raw_docs = load_pdf_documents(saved_path)
            processed_chunks = chunk_documents(raw_docs)
            index_documents(processed_chunks)
        st.success("✅ Documents Uploaded – Let’s Chat! 👇")
    except Exception as e:
        st.error(f"❌ An error occurred: {e}")

# Chat Section
st.subheader("🤖 AI Chat")
for chat in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(chat["user"])
    with st.chat_message("assistant", avatar="🤖"):
        st.write(chat["assistant"])

# Chat Input
user_input = st.chat_input("🤖 Ask me anything – I’m your PDF assistant!")

if user_input:
    with st.chat_message("user"):
        st.write(user_input)
    
    with st.spinner("🔍 Unlocking the Secrets of Your Document..."):
        relevant_docs = find_related_documents(user_input)
        ai_response = generate_answer(user_input, relevant_docs)
    
    with st.chat_message("assistant", avatar="🤖"):
        thinking_placeholder = st.empty()
        thinking_placeholder.info("🤔 AI thinking...")

        full_response = "".join(chunk for chunk in ai_response)
        thinking_placeholder.empty()

        response_placeholder = st.empty()
        displayed_text = ""

        for word in full_response.split():
            displayed_text += word + " "
            response_placeholder.markdown(f"**{displayed_text}**")  
            time.sleep(0.01)

    # Add chat history
    st.session_state.chat_history.append({"user": user_input, "assistant": full_response})
    save_chat_history(st.session_state.chat_history)
import os
import streamlit as st
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Move these outside main() to prevent recreation on each rerun
if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key='answer'
    )

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def set_openai_api_key():
    if "api_key" not in st.session_state:
        st.session_state.api_key = None
    
    st.sidebar.header("API Key Configuration")
    api_key = st.sidebar.text_input(
        "Enter your OpenAI API key:",
        type="password",
        placeholder="sk-...",
        key="api_key_input"
    )
    
    if api_key:
        st.session_state.api_key = api_key
        os.environ["OPENAI_API_KEY"] = api_key
        return api_key
    return None

def initialize_llm():
    return ChatOpenAI(
        model_name="gpt-4",
        temperature=0.7
    )

def load_existing_vectorstore(persist_directory='db'):
    try:
        embedding = OpenAIEmbeddings()
        return Chroma(persist_directory=persist_directory, embedding_function=embedding)
    except Exception as e:
        st.error(f"Error loading vector store: {e}")
        return None

def create_vectorstore_from_path(file_path, persist_directory='db'):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        documents = [Document(page_content=chunk) for chunk in text_splitter.split_text(content)]

        embedding = OpenAIEmbeddings()
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embedding,
            persist_directory=persist_directory
        )
        vectorstore.persist()
        return vectorstore

    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        return None

qa_template = """
Use the following conversation history and context to answer the question at the end.
If you don't know the answer, just say that you don't know; don't try to make up an answer.
give descriptive answers.

Conversation History:
{chat_history}

Context:
{context}

Question: {question}
Helpful Answer:"""

def create_chain(vectorstore, llm):
    prompt = PromptTemplate(
        input_variables=["context", "chat_history", "question"],
        template=qa_template
    )
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=st.session_state.memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={'prompt': prompt},
        verbose=True
    )

def main():
    st.set_page_config(page_title="RAG QA Bot", page_icon="🤖", layout="wide")

    st.markdown("""
        <style>
        .chat-box { border: 2px solid #e0e0e0; border-radius: 10px; padding: 10px; margin-bottom: 20px; }
        .user-message { background-color: #e8f5e9; border: 1px solid #c8e6c9; padding: 10px; border-radius: 10px; margin-bottom: 10px; }
        .bot-message { background-color: #f3e5f5; border: 1px solid #e1bee7; padding: 10px; border-radius: 10px; margin-bottom: 10px; }
        .input-container { position: fixed; bottom: 0; width: 100%; background-color: #ffffff; padding: 10px; }
        </style>
    """, unsafe_allow_html=True)

    st.title("🤖 RAG Composite AI Tutor")

    api_key = set_openai_api_key()
    if not api_key:
        return

    if 'qa_chain' not in st.session_state:
        llm = initialize_llm()
        file_path = ""
        vectordb = load_existing_vectorstore() or create_vectorstore_from_path(file_path)
        
        if vectordb is None:
            st.error("Failed to initialize vector store")
            return
            
        st.session_state.qa_chain = create_chain(vectordb, llm)

    # Display chat history
    for question, answer in st.session_state.chat_history:
        st.markdown(f'<div class="chat-box user-message"><strong>You:</strong> {question}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="chat-box bot-message"><strong>Bot:</strong> {answer}</div>', unsafe_allow_html=True)

    # User input
    user_input = st.text_input("Ask a question:", key="user_input")

    if user_input and user_input != st.session_state.get('last_input'):
        st.session_state.last_input = user_input
        
        with st.spinner("Processing..."):
            result = st.session_state.qa_chain.invoke({"question": user_input})
            response = result['answer']
            st.session_state.chat_history.append((user_input, response))
            
        st.rerun()

if __name__ == "__main__":
    main()

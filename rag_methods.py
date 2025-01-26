import os
import dotenv
import streamlit as st
from time import time
import shutil
import zipfile

from langchain_community.document_loaders.text import TextLoader
from langchain_community.document_loaders import (
    WebBaseLoader, 
    PyPDFLoader, 
    Docx2txtLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, AzureOpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma  # Updated import

dotenv.load_dotenv()

os.environ["USER_AGENT"] = "myagent"
DB_DOCS_LIMIT = 10

def stream_llm_response(llm_stream, messages):
    """Function to stream the response of the LLM"""
    response_message = ""

    for chunk in llm_stream.stream(messages):
        response_message += chunk.content
        yield chunk

    st.session_state.messages.append({"role": "assistant", "content": response_message})

def initialize_vector_db(docs):
    """Initialize the vector database with documents"""
    if "AZ_OPENAI_API_KEY" not in os.environ:
        embedding = OpenAIEmbeddings(api_key=st.session_state.openai_api_key)
    else:
        embedding = AzureOpenAIEmbeddings(
            api_key=os.getenv("AZ_OPENAI_API_KEY"), 
            azure_endpoint=os.getenv("AZ_OPENAI_ENDPOINT"),
            model="text-embedding-3-large",
            openai_api_version="2024-02-15-preview",
        )

    vector_db = Chroma.from_documents(
        documents=docs,
        embedding=embedding,
        collection_name=f"{str(time()).replace('.', '')[:14]}_" + st.session_state['session_id'],
    )

    return vector_db

def load_pretrained_db():
    """Load the pretrained Chroma vector database"""
    temp_dir = "./temp_vectordb"
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        if not os.path.exists("db_jones.zip"):
            st.error("Composite materials database not found")
            return None

        with zipfile.ZipFile("db_jones.zip", 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        # Get collection name dynamically
        collections = [d for d in os.listdir(temp_dir) 
                      if os.path.isdir(os.path.join(temp_dir, d))]
        
        if not collections:
            st.error("No collections found in the database")
            return None

        collection_name = collections[0]

        # Initialize embedding model
        if "AZ_OPENAI_API_KEY" not in os.environ:
            embedding = OpenAIEmbeddings(api_key=st.session_state.openai_api_key)
        else:
            embedding = AzureOpenAIEmbeddings(
                api_key=os.getenv("AZ_OPENAI_API_KEY"), 
                azure_endpoint=os.getenv("AZ_OPENAI_ENDPOINT"),
                model="text-embedding-3-large",
                openai_api_version="2024-02-15-preview",
            )

        # Modern Chroma client configuration
        import chromadb
        from chromadb.config import Settings
        
        client = chromadb.PersistentClient(
            path=os.path.join(temp_dir, collection_name),
            settings=Settings(
                allow_reset=True,
                anonymized_telemetry=False
            ),
            tenant="default_tenant",
            database="default_database"
        )

        vector_db = Chroma(
            client=client,
            collection_name=collection_name,
            embedding_function=embedding
        )
        
        return vector_db
    
    except Exception as e:
        st.error(f"Error loading pretrained vector database: {e}")
        return None


def cleanup_temp_directory():
    """Clean up temporary directory when session ends"""
    temp_dir = "./temp_vectordb"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)

def load_doc_to_db():
    """Load documents to the vector database"""
    if "rag_docs" in st.session_state and st.session_state.rag_docs:
        docs = [] 
        for doc_file in st.session_state.rag_docs:
            if doc_file.name not in st.session_state.rag_sources:
                if len(st.session_state.rag_sources) < DB_DOCS_LIMIT:
                    os.makedirs("source_files", exist_ok=True)
                    file_path = f"./source_files/{doc_file.name}"
                    with open(file_path, "wb") as file:
                        file.write(doc_file.read())

                    try:
                        if doc_file.type == "application/pdf":
                            loader = PyPDFLoader(file_path)
                        elif doc_file.name.endswith(".docx"):
                            loader = Docx2txtLoader(file_path)
                        elif doc_file.type in ["text/plain", "text/markdown"]:
                            loader = TextLoader(file_path)
                        else:
                            st.warning(f"Document type {doc_file.type} not supported.")
                            continue

                        docs.extend(loader.load())
                        st.session_state.rag_sources.append(doc_file.name)

                    except Exception as e:
                        st.toast(f"Error loading document {doc_file.name}: {e}", icon="⚠️")
                        print(f"Error loading document {doc_file.name}: {e}")
                    
                    finally:
                        os.remove(file_path)

                else:
                    st.error(f"Maximum number of documents reached ({DB_DOCS_LIMIT}).")

        if docs:
            _split_and_load_docs(docs)
            st.toast(f"Document *{str([doc_file.name for doc_file in st.session_state.rag_docs])[1:-1]}* loaded successfully.", icon="✅")

def load_url_to_db():
    """Load URL content to the vector database"""
    if "rag_url" in st.session_state and st.session_state.rag_url:
        url = st.session_state.rag_url
        docs = []
        if url not in st.session_state.rag_sources:
            if len(st.session_state.rag_sources) < 10:
                try:
                    loader = WebBaseLoader(url)
                    docs.extend(loader.load())
                    st.session_state.rag_sources.append(url)

                except Exception as e:
                    st.error(f"Error loading document from {url}: {e}")

                if docs:
                    _split_and_load_docs(docs)
                    st.toast(f"Document from URL *{url}* loaded successfully.", icon="✅")

            else:
                st.error("Maximum number of documents reached (10).")

def _split_and_load_docs(docs):
    """Split documents and load them into the vector database"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=5000,
        chunk_overlap=1000,
    )

    document_chunks = text_splitter.split_documents(docs)

    if "vector_db" not in st.session_state:
        st.session_state.vector_db = initialize_vector_db(docs)
    else:
        st.session_state.vector_db.add_documents(document_chunks)

def _get_context_retriever_chain(vector_db, llm):
    """Create a retriever chain for getting context"""
    retriever = vector_db.as_retriever()
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="messages"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation, focusing on the most recent messages."),
    ])
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

    return retriever_chain

def get_conversational_rag_chain(llm):
    """Create a conversational RAG chain"""
    retriever_chain = _get_context_retriever_chain(st.session_state.vector_db, llm)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
        """You are a helpful assistant. if you do not know an answer-say you dont know please dont give or generate false answers'. Give descriptive answers\n
        {context}"""),
        MessagesPlaceholder(variable_name="messages"),
        ("user", "{input}"),
    ])
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)

    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

def stream_llm_rag_response(llm_stream, messages):
    """Stream the RAG-enhanced LLM response"""
    conversation_rag_chain = get_conversational_rag_chain(llm_stream)
    response_message = "*(RAG Response)*\n"
    for chunk in conversation_rag_chain.pick("answer").stream({"messages": messages[:-1], "input": messages[-1].content}):
        response_message += chunk
        yield chunk

    st.session_state.messages.append({"role": "assistant", "content": response_message})

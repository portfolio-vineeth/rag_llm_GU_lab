import streamlit as st
import os
import dotenv
import uuid

if os.name == 'posix':
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.schema import HumanMessage, AIMessage

from rag_methods import (
    load_doc_to_db, 
    load_url_to_db,
    stream_llm_response,
    stream_llm_rag_response,
    load_pretrained_db,
    cleanup_temp_directory
)

dotenv.load_dotenv()

if "AZ_OPENAI_API_KEY" not in os.environ:
    MODELS = [
        "openai/gpt-4",
        "openai/gpt-3.5-turbo",
        "anthropic/claude-3-sonnet-20240307",
    ]
else:
    MODELS = ["azure-openai/gpt-4"]

st.set_page_config(
    page_title="Composites Class app", 
    page_icon="📚", 
    layout="centered", 
    initial_sidebar_state="expanded"
)

# --- Header ---
st.html("""<h2 style="text-align: center;">📚🔍 <i> Composites Class AI Tutor </i> 🤖💬</h2>""")

# --- Initial Setup ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "rag_sources" not in st.session_state:
    st.session_state.rag_sources = []

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there! How can I assist you today?"}
    ]

# --- Side Bar LLM API Tokens ---
with st.sidebar:
    if "AZ_OPENAI_API_KEY" not in os.environ:
        default_openai_api_key = os.getenv("OPENAI_API_KEY") if os.getenv("OPENAI_API_KEY") is not None else ""
        with st.popover("🔐 OpenAI"):
            openai_api_key = st.text_input(
                "Introduce your OpenAI API Key (https://platform.openai.com/)", 
                value=default_openai_api_key, 
                type="password",
                key="openai_api_key",
            )

        default_anthropic_api_key = os.getenv("ANTHROPIC_API_KEY") if os.getenv("ANTHROPIC_API_KEY") is not None else ""
        with st.popover("🔐 Anthropic"):
            anthropic_api_key = st.text_input(
                "Introduce your Anthropic API Key (https://console.anthropic.com/)", 
                value=default_anthropic_api_key, 
                type="password",
                key="anthropic_api_key",
            )
    else:
        openai_api_key, anthropic_api_key = None, None
        st.session_state.openai_api_key = None
        az_openai_api_key = os.getenv("AZ_OPENAI_API_KEY")
        st.session_state.az_openai_api_key = az_openai_api_key

# --- Main Content ---
missing_openai = openai_api_key == "" or openai_api_key is None or "sk-" not in openai_api_key
missing_anthropic = anthropic_api_key == "" or anthropic_api_key is None
if missing_openai and missing_anthropic and ("AZ_OPENAI_API_KEY" not in os.environ):
    st.write("#")
    st.warning("⬅️ Please introduce an API Key to continue...")

else:
    # Sidebar
    with st.sidebar:
        st.divider()
        models = []
        for model in MODELS:
            if "openai" in model and not missing_openai:
                models.append(model)
            elif "anthropic" in model and not missing_anthropic:
                models.append(model)
            elif "azure-openai" in model:
                models.append(model)

        st.selectbox(
            "🤖 Select a Model", 
            options=models,
            key="model",
        )

        cols0 = st.columns(2)
        with cols0[0]:
            is_vector_db_loaded = ("vector_db" in st.session_state and st.session_state.vector_db is not None)
            st.toggle(
                "Use RAG", 
                value=is_vector_db_loaded, 
                key="use_rag", 
                disabled=not is_vector_db_loaded,
            )

        with cols0[1]:
            if st.button("Clear Chat", type="primary"):
                st.session_state.messages = []
                cleanup_temp_directory()

        # Knowledge base selection
        kb_option = st.radio(
            "📚 Knowledge Base",
            ["Custom Documents", "Composite Materials"],
            key="kb_option"
        )

        if kb_option == "Custom Documents":
            st.header("RAG Sources:")
            
            # File upload input for RAG with documents
            st.file_uploader(
                "📄 Upload a document", 
                type=["pdf", "txt", "docx", "md"],
                accept_multiple_files=True,
                on_change=load_doc_to_db,
                key="rag_docs",
            )

            # URL input for RAG with websites
            st.text_input(
                "🌐 Introduce a URL", 
                placeholder="https://example.com",
                on_change=load_url_to_db,
                key="rag_url",
            )

            with st.expander(f"📚 Documents in DB ({0 if not is_vector_db_loaded else len(st.session_state.rag_sources)})"):
                st.write([] if not is_vector_db_loaded else [source for source in st.session_state.rag_sources])
        
        else:  # Composite Materials selected
            composite_loaded = (
                hasattr(st.session_state, 'vector_db') and 
                getattr(st.session_state.vector_db, '_is_composite_db', False)
            )
            
            if not composite_loaded:
                with st.spinner("Loading Composite Materials Knowledge Base..."):
                    new_vector_db = load_pretrained_db()
                    if new_vector_db:
                        # Clear previous non-composite DB
                        if 'vector_db' in st.session_state:
                            del st.session_state.vector_db
                        
                        new_vector_db._is_composite_db = True
                        st.session_state.vector_db = new_vector_db
                        st.session_state.rag_sources = ["Composite Materials Knowledge Base"]
                        st.rerun()
            
            st.info("Using pre-trained Composite Materials knowledge base")
    
    # Main chat app
    model_provider = st.session_state.model.split("/")[0]
    if model_provider == "openai":
        llm_stream = ChatOpenAI(
            api_key=openai_api_key,
            model_name=st.session_state.model.split("/")[-1],
            temperature=0.3,
            streaming=True,
        )
    elif model_provider == "anthropic":
        llm_stream = ChatAnthropic(
            api_key=anthropic_api_key,
            model=st.session_state.model.split("/")[-1],
            temperature=0.3,
            streaming=True,
        )
    elif model_provider == "azure-openai":
        llm_stream = AzureChatOpenAI(
            azure_endpoint=os.getenv("AZ_OPENAI_ENDPOINT"),
            openai_api_version="2024-02-15-preview",
            model_name=st.session_state.model.split("/")[-1],
            openai_api_key=os.getenv("AZ_OPENAI_API_KEY"),
            openai_api_type="azure",
            temperature=0.3,
            streaming=True,
        )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Your message"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            messages = [HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"]) for m in st.session_state.messages]

            if not st.session_state.use_rag:
                st.write_stream(stream_llm_response(llm_stream, messages))
            else:
                st.write_stream(stream_llm_rag_response(llm_stream, messages))

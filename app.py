import os
import streamlit as st

if os.name == 'posix':
   __import__('pysqlite3')
   import sys
   sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

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
       model_name="gpt-3.5-turbo",
       temperature=0.7
   )

def load_existing_vectorstore(persist_directory='db_vin'):
   try:
       if not os.environ.get("OPENAI_API_KEY"):
           st.error("OpenAI API key not found")
           return None
       embedding = OpenAIEmbeddings()
       return Chroma(persist_directory=persist_directory, embedding_function=embedding)
   except Exception as e:
       st.error(f"Error loading vector store: {e}")
       return None

def create_chain(vectorstore, llm):
   prompt = PromptTemplate(
       input_variables=["context", "chat_history", "question"],
       template="""Use the following conversation history and context to answer the question at the end.
If you don't know the answer, just say that you don't know; don't try to make up an answer.
give descriptive answers.

Conversation History:
{chat_history}

Context:
{context}

Question: {question}
Helpful Answer:"""
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
   st.title("🤖 RAG Composite AI Tutor")

   api_key = set_openai_api_key()
   if not api_key:
       return

   if 'qa_chain' not in st.session_state:
       llm = initialize_llm()
       vectordb = load_existing_vectorstore()
       
       if vectordb is None:
           st.error("Failed to initialize vector store")
           return
           
       st.session_state.qa_chain = create_chain(vectordb, llm)

   for question, answer in st.session_state.chat_history:
       st.markdown(f'<div class="chat-box user-message"><strong>You:</strong> {question}</div>', unsafe_allow_html=True)
       st.markdown(f'<div class="chat-box bot-message"><strong>Bot:</strong> {answer}</div>', unsafe_allow_html=True)

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

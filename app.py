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

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = "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"

# Initialize OpenAI LLM
turbo_llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0.7
)

def load_existing_vectorstore(persist_directory='db_vin'):
    try:
        embedding = OpenAIEmbeddings()
        return Chroma(persist_directory=persist_directory, embedding_function=embedding)
    except Exception as e:
        st.error(f"Error loading vector store: {e}")
        return None
def create_vectorstore_from_path(file_path, persist_directory='db'):
    """Create vector store from a preloaded .txt file"""
    st.info("Processing preloaded file and creating vector store...")
    try:
        # Read file content
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()

        # Split content into manageable chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Adjust chunk size
            chunk_overlap=200  # Ensure context continuity
        )
        documents = [Document(page_content=chunk) for chunk in text_splitter.split_text(content)]

        # Create embeddings and vector store
        embedding = OpenAIEmbeddings()
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embedding,
            persist_directory=persist_directory
        )
        vectorstore.persist()
        st.success(f"Vector store created successfully with {len(documents)} chunks.")
        return vectorstore

    except Exception as e:
        st.error("Error while creating vector store from preloaded file.")
        st.error(f"Details: {e}")
        return None

# Define QA template
qa_template = """
Use the following conversation history and context to answer the question at the end.
If you don't know the answer, just say that you don't know; don't try to make up an answer.
give descriptive answers. Also explain it in easy terms.
Conversation History:
{chat_history}

Context:
{context}

Question: {question}
Helpful Answer:
"""

# Create prompt template
prompt = PromptTemplate(
    input_variables=["context", "chat_history", "question"],
    template=qa_template
)

# Initialize memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key='answer'
)

def format_source_documents(source_documents):
    sources = []
    for doc in source_documents:
        if 'source' in doc.metadata:
            sources.append(f"- {doc.metadata['source']}")

    if sources:
        return "\n\nSources:\n" + "\n".join(sources)
    return ""

def create_chain(vectorstore):
    # Set up retriever with k=10
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    return ConversationalRetrievalChain.from_llm(
        llm=turbo_llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={'prompt': prompt},
        verbose=True
    )

# Streamlit app setup
def main():
    st.title("Composite AI tutor")
    st.write("Ask questions regarding composites from MECHANICS OF COMPOSITE MATERIALS textbook.")

    # Path to preloaded .txt file
    file_path = "/Users/gurunanmapurushotam/Documents/rag_llm/rag_llm_app/Mechanics of Composite Materials 2nd Ed 1999 BY [Taylor & Francis] (1) (1).txt"
    persist_directory = 'db_vin'

    # Load existing vector store or create a new one
    vectordb = load_existing_vectorstore(persist_directory)
    # if vectordb is None:
    #     vectordb = create_vectorstore_from_path(file_path, persist_directory)

    if vectordb is None:
        st.error("Vector store could not be loaded or created. Please check the file and try again.")
        return

    # Create the chain
    qa_chain = create_chain(vectordb)

    # Initialize chat history in Streamlit session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # User input
    user_input = st.text_input("Ask a question:")

    if user_input:
        # Get response from the chain
        result = qa_chain.invoke({"question": user_input})
        response = result['answer']

        # Format sources and add to response
        sources = format_source_documents(result.get('source_documents', []))
        if sources:
            response += sources

        # Display response
        st.session_state.chat_history.append((user_input, response))

    # Display chat history
    if st.session_state.chat_history:
        for question, answer in st.session_state.chat_history:
            st.markdown(f"**You:** {question}")
            st.markdown(f"**Bot:** {answer}")

if __name__ == "__main__":
    main()

import os
import pathlib
import logging
import streamlit as st
from streamlit_chat import message
from langchain_community.vectorstores import Chroma  # Updated import
from langchain_community.embeddings import OpenAIEmbeddings  # Updated import
from langchain_community.llms import OpenAI  # Updated import
from langchain.text_splitter import PythonCodeTextSplitter
from langchain.chains import RetrievalQA  # Directly use RetrievalQA
from langchain.schema import Document

# Configure logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

# Streamlit configuration
st.set_option('client.showErrorDetails', False)
st.set_page_config(page_title="CODE CHAT", page_icon=":robot_face:")
st.markdown("<h1 style='text-align: center; color: red;'>CODE CHAT</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Query Your GIT REPO</h3>", unsafe_allow_html=True)

# Initialize session state
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []

# OpenAI API key validation
if "OPENAI_API_KEY" not in os.environ:
    st.error("Please set your OPENAI_API_KEY environment variable.")
    st.stop()

# Paths
REPO_PATH = '/Users/ravi/projects/bijucyborg-rag-app/chatwithcsv'
CHROMA_DB_PATH = f'./chroma/{os.path.basename(REPO_PATH)}'

# Load repository files
def get_repo_docs(repo_path):
    repo = pathlib.Path(repo_path)
    logging.debug("Loading repository files...")
    #for codefile in repo.glob("**/*.ipynb"):
    for codefile in repo.glob("**/*.py"):
        logging.debug(f"Processing file: {codefile}")
        with open(codefile, "r") as file:
            rel_path = codefile.relative_to(repo)
            yield Document(page_content=file.read(), metadata={"source": str(rel_path)})

# Split files into chunks
def get_source_chunks(repo_path):
    source_chunks = []
    splitter = PythonCodeTextSplitter(chunk_size=1024, chunk_overlap=30)
    for source in get_repo_docs(repo_path):
        for chunk in splitter.split_text(source.page_content):
            logging.debug(f"Chunk created: {chunk[:100]}...")
            source_chunks.append(Document(page_content=chunk, metadata=source.metadata))
    return source_chunks

# Generate response
def generate_response(input_text):
    # Load or create ChromaDB
    if not os.path.exists(CHROMA_DB_PATH):
        source_chunks = get_source_chunks(REPO_PATH)
        #Embedding Model: Model-1 Used:  'OpenAIEmbeddings'
        vector_db = Chroma.from_documents(source_chunks, OpenAIEmbeddings(), persist_directory=CHROMA_DB_PATH)
        vector_db.persist()
    else:
        vector_db = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=OpenAIEmbeddings())

    # Create QA chain
    retriever = vector_db.as_retriever()
    #LLM for Generating Response: Model-2 Used:  'OpenAI'
    qa_chain = RetrievalQA.from_chain_type(
        llm=OpenAI(temperature=0.5),
        retriever=retriever,
        chain_type="stuff",
    )

    # Debug retrieved documents
    results = retriever.get_relevant_documents(input_text)
    logging.debug(f"Retrieved documents: {[result.metadata['source'] for result in results]}")

    # Run the chain
    query_response = qa_chain.run(input_text)
    logging.debug(f"Generated response: {query_response}")
    return query_response

# Streamlit app layout
response_container = st.container()
input_container = st.container()

with input_container:
    with st.form(key='input_form', clear_on_submit=True):
        user_input = st.text_area("Enter your query:", key='input', height=100)
        submit_button = st.form_submit_button(label='Submit')

    if submit_button and user_input:
        try:
            response = generate_response(user_input)
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(response)
        except Exception as e:
            logging.error(f"An error occurred: {e}")
            st.error(f"An error occurred: {e}")

if st.session_state['generated']:
    with response_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i], is_user=True, key=f"user_{i}")
            st.code(st.session_state["generated"][i], language="python")

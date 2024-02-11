from langchain_community.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

import streamlit as st 
from wxai_langchain.llm import LangChainInterface
from wxai_langchain.credentials import Credentials
creds =Credentials(
    api_key= '65x4dxLCwZl4ALJsMvDYieZoS_QZqlDvnvRwJ3dHRyVZ',
    api_endpoint=  'https://us-south.ml.cloud.ibm.com',
    project_id = '17ed1163-812a-4d9f-b9b4-45f83af41994'
)

llm = LangChainInterface(
    credentials=creds,
    model = 'ibm/granite-13b-chat-v1',
    params={
        'decoding_tokens' : 'sample',
        'max_new_tokens' :200,
        'temperature' : 0.5,
    })

@st.cache_resource
def load_pdf(file):
    loaders = [PyPDFLoader(file)]
    index = VectorstoreIndexCreator(
        embedding=HuggingFaceEmbeddings(model_name = 'all-mpnet-base-v2'),
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=150)
    ).from_loaders(loaders)
    return index

# File upload
pdf_file = st.file_uploader("Upload PDF file", type=["pdf"])

if pdf_file is not None:
    # Save the PDF file temporarily
    temp_pdf_path = "temp_pdf.pdf"
    with open(temp_pdf_path, "wb") as temp_file:
        temp_file.write(pdf_file.read())

    # Load PDF and create index
    index = load_pdf(temp_pdf_path)

    # Create a Q&A chain
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=index.vectorstore.as_retriever(),
        input_key='question'
    )

    # Remove the temporary PDF file
    os.remove(temp_pdf_path)

st.title('AI Grader')
st.write("Streamline Your Grading with AI Grader: Smarter, Faster, Fairer!")

if 'messages' not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])


prompt = st.chat_input("Your prompt here")

if prompt:
    st.chat_message('user').markdown(prompt)
    st.session_state.messages.append({'role':'user' , 'content': prompt})

    # Add prompt to the user's input
    prompt_with_user_input = f"""You've been tasked with grading a student's assignment. The assignment and grading criteria are provided in {pdf_file}. Below is the solution submitted by the student for you to grading:

{prompt}

Please provide Points scored in each question by a student.

"""
    response = chain.run(prompt_with_user_input)  

    st.chat_message('assistant').markdown(response)
    st.session_state.messages.append({'role':'assistant' , 'content': response})
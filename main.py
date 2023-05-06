import streamlit as st
from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback


load_dotenv()
key = os.getenv('OPENAI_API_KEY')


st.set_page_config(page_title='ASKPDF-GPT',page_icon=':chat:')
st.header('I have a question about the pdf :point_up:')
pdf_file = st.file_uploader('Upload your PDF ',accept_multiple_files=False,type='pdf')

if pdf_file is not None:
    text=''
    for page in PdfReader(pdf_file).pages:
        text += page.extract_text()

    splitter = CharacterTextSplitter(separator='\n',chunk_size = 800,chunk_overlap=150,length_function = len)

    chunks= splitter.split_text(text)

    embeddings =  OpenAIEmbeddings()

    thelibrary =FAISS.from_texts(chunks,embedding=embeddings) 


    quest = st.text_input(label='Ask your question about your pdf')

    if quest:
        found_doc = thelibrary.similarity_search(quest)

    
        with get_openai_callback() as cb:
            chain = load_qa_chain(OpenAI(), chain_type="stuff")
            ans = chain.run(input_documents=found_doc, question=quest)


        st.write(ans)
        st.write(cb)


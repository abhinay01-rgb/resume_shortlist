import openai
import os
import pickle
import time
import langchain
from langchain import OpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

openai_api_key = "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"
llm = OpenAI(openai_api_key=openai_api_key)

def save_faiss_index(index, file_path):
    index.serialize(file_path)

def load_faiss_index(file_path):
    index = FAISS.deserialize(file_path)
    return index

def pdf_to_qa_retrieve(pdf_path, question):
    loader = PyPDFLoader(pdf_path)
    doc = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(separators = ["\n\n","\n"," "],chunk_size=200, chunk_overlap=0)
    docs = text_splitter.split_documents(doc)
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    vectorindex_openai = FAISS.from_documents(docs, embeddings)

    file_path = "vector_index.faiss"  # Use a different file extension
    save_faiss_index(vectorindex_openai, file_path)

    if os.path.exists(file_path):
        vectorindex_openai = load_faiss_index(file_path)

    chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorindex_openai.as_retriever())
    
    langchain.debug=True

    input_data = {"question": question}
    result = chain(input_data, return_only_outputs=True)
    return result['answer']

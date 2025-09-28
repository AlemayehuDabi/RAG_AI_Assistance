import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from loaders import load_and_split_docs

VECTORSTORE_PATH = "vectorstore/faiss_index"

def build_vectorstore():
    docs = load_and_split_docs()
    embeddings = HuggingFaceEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    os.makedirs("vectorstore", exist_ok=True)
    vectorstore.save_local(VECTORSTORE_PATH)
    return vectorstore

def load_vectorstore():
    embeddings = HuggingFaceEmbeddings()
    return FAISS.load_local(VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True)

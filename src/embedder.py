import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from loaders import load_and_split_docs

VECTORSTORE_PATH = "vectorstore/index"

def build_vectorstore():
    docs = load_and_split_docs()
    if not docs:
        raise ValueError("❌ No documents found in data/publications. Check file path and format.")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(docs, embeddings)
    os.makedirs("vectorstore", exist_ok=True)
    vectorstore.save_local(VECTORSTORE_PATH)
    return vectorstore

def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return Chroma.load_local(VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True)

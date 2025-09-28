import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_and_split_docs(data_path: str = "data/publications"):
    # Load all text/markdown files
    loader = DirectoryLoader(data_path, glob="**/*.txt", loader_cls=TextLoader)
    docs = loader.load()

    # Split docs into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    split_docs = splitter.split_documents(docs)
    return split_docs

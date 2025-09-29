from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from retriever import get_retriever

def create_rag_pipeline():
    retriever = get_retriever()
    llm = ChatGroq(model="llama3-8b-8192", temperature=0)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
    )
    return qa_chain

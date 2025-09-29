from langchain.chains import RetrievalQA
from retriever import get_retriever
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import os

load_dotenv()

def create_rag_pipeline():
    retriever = get_retriever()
    groq_api_key = os.getenv("GOOGLE_API_KEY")
    if not groq_api_key:
        raise ValueError(
            "❌ GOOGLE_API_KEY is not set. Please add it to your .env file!"
        )
        
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash",
        temperature=0.5)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
    )
    return qa_chain

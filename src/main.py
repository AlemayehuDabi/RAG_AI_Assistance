from rag_pipeline import create_rag_pipeline

def run_cli():
    print("📚 RAG Assistant (type 'exit' to quit)\n")
    qa = create_rag_pipeline()
    while True:
        query = input(">> You: ")
        if query.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break

        result = qa(query)
        answer = result["result"]
        sources = result.get("source_documents", [])

        print("\n🤖 Assistant:", answer)
        if sources:
            print("\n📖 Sources:")
            for doc in sources:
                print(f" - {doc.metadata.get('source', 'Unknown')}")
        print("-" * 50)

if __name__ == "__main__":
    run_cli()

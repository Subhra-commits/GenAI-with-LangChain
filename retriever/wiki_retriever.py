from langchain_community.retrievers import WikipediaRetriever

retriever = WikipediaRetriever(
    top_k_results= 2,
    lang= 'en'
)

user_query = "What is the difference between shallow copy and deep copy in Python?"

docs = retriever.invoke(user_query)


for i, doc in enumerate(docs):
    print(f"\n--- Result {i+1} ---")
    print(f"Content: - {doc.page_content}")
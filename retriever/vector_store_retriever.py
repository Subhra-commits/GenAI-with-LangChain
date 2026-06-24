from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain_chroma import Chroma
from vector_db.chroma_db import vector_store

load_dotenv()

hf_embeddings = HuggingFaceEmbeddings(
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
)

# Use akready created vector store
vector_store = Chroma(
    embedding_function= hf_embeddings, 
    persist_directory= 'Chroma_db',
    collection_name= 'Players'
)

retriever = vector_store.as_retriever(search_kwargs={"k": 2})

user_query = "Who is the best bowler?"
result = retriever.invoke(user_query)

for i, doc in enumerate(result):
    print(f"\n--- Result {i+1} ---")
    print(f"Content: - {doc.page_content}")
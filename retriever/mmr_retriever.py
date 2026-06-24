from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

load_dotenv()

hf_embeddings = HuggingFaceEmbeddings(
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
)

doc1 = Document(
    page_content = "Mahendra Singh Dhoni is one of India's most successful cricket captains, known for his calm leadership, exceptional wicketkeeping, and finishing ability. He led India to victories in the 2007 T20 World Cup, 2011 Cricket World Cup, and 2013 Champions Trophy, making him one of the greatest captains in cricket history.",
    metadata = {"team" : "IPL - CSK"}
)

doc2 = Document(
    page_content = "Virat Kohli is one of the greatest modern-day cricketers, known for his exceptional batting consistency, aggressive mindset, and remarkable fitness standards.He has captained the Indian cricket team across formats and holds numerous records in international cricket, particularly in ODI run-scoring and successful run chases.",
    metadata = {"team" : "IPL - RCB"}
)

doc3 = Document(
    page_content= "Jasprit Bumrah is one of the world's premier fast bowlers, known for his unique bowling action, pinpoint yorkers, and ability to perform under pressure in all formats of cricket. He has been a key match-winner for India, consistently leading the bowling attack with his pace, accuracy, and wicket-taking ability.",
    metadata = {"team" : "IPL - MI"}
)

docs = [doc1, doc2, doc3]

vector_store = FAISS.from_documents(
    documents= docs, 
    embedding= hf_embeddings
)

retriever = vector_store.as_retriever(
    search_type = "mmr", 
    search_kwargs = {"k":2, "lambda_mult" : 0.2}
)

user_query = "Who is the best bowler?"
result = retriever.invoke(user_query)

for i, doc in enumerate(result):
    print(f"\n--- Result {i+1} ---")
    print(f"Content: - {doc.page_content}")
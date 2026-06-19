from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')

user_query = "What is the capital of India?"
docs = [
    "Jakarta is the capital of Indonesia",
    "Kolkata is the capital of British India",
    "Delhi is the capital of India",
    "Paris is the capital of France",
    "Madrid is the capital of Spain"    
]

query_embedding = embeddings.embed_query(user_query)
doc_embeddings = embeddings.embed_documents(docs)

similarity_score = cosine_similarity([query_embedding], doc_embeddings)[0]
print(similarity_score)

index, score = sorted(list(enumerate(similarity_score)), key = lambda x:x[1])[-1]

print(user_query)
print(docs[index])
print("Similarity score: ", score*100)



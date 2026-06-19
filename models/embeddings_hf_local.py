from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')

text = "What is the capital of India?"
docs = [
    "What is the capital of India?",
    "What is the capital of France?"
    "What is the capital of Spain?"
]

vector = embeddings.embed_documents(docs)

print(str(vector))

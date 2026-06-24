from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain_chroma import Chroma
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

vector_store = Chroma(
    embedding_function= hf_embeddings, 
    persist_directory= 'Chroma_db',
    collection_name= 'Players'
)

# Add document embeddings into vector db
docs_id = vector_store.add_documents(docs)
print("Embedded id of documents: ", docs_id)

# Retrieve embddings from vector db
print(vector_store.get(include=['embeddings', 'documents', 'metadatas']))

# Search query in vector db
query_res = vector_store.similarity_search(
    query='What is the full name of Dhoni?',
    k=1 # K value determines how many results we want to display in our result, k=1 means 1 result, k=2 means 2 etc
)
print(query_res)

# Search query in vector db with similarity score
res_with_score= vector_store.similarity_search_with_score(
    query='Who is the best bowler?',
    k=2
)
print(res_with_score)

# Search query in vector db with meta data filtering
filtered_res = vector_store.similarity_search_with_score(
    query='',
    filter={'team':'IPL - RCB'}
)
print(filtered_res)

# Updating an existing document
updated_doc = Document(
    page_content = 'MS Dhoni is a legendary Indian wicketkeeper-batter and one of the most successful cricket captains of all time. Known as "Captain Cool," he led India to victories in the 2007 T20 World Cup, 2011 ODI World Cup, and 2013 Champions Trophy, while earning a reputation as one of the best finishers in cricket history.',
    metadata = {"team" : "India"}
)

vector_store.update_document(document_id= '646b25a5-84a3-4b20-95b4-20ffc3b9ab6b', document=updated_doc)

print(vector_store.get(ids = ['646b25a5-84a3-4b20-95b4-20ffc3b9ab6b']))

# Deleting an exsiting document
deleted_id = vector_store.delete(ids = ['c2b995b1-fb57-445b-8586-ceb8e0c56304'])
print(vector_store.get(ids = ['c2b995b1-fb57-445b-8586-ceb8e0c56304'])) # Returning None
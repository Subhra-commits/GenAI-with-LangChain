from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from dotenv import load_dotenv

load_dotenv()

llm_model = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm_model)

hf_embeddings = HuggingFaceEmbeddings(
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
)

doc1 = Document(
    page_content = ("""Virat Kohli is one of India's greatest batsmen, known for his consistency, aggressive mindset, and ability to chase targets. Mahendra Singh Dhoni is famous for his calm leadership, wicketkeeping skills, and match-finishing ability. While Kohli excels in batting records, Dhoni is remembered for leading India to major tournament victories. 
            Both have played crucial roles in Indian cricket and share many memorable partnerships. """),
    metadata = ({"team" : "India"})
)

doc2 = Document(
    page_content = ("""
        Virat Kohli is one of India's most successful batters, known for his consistency, aggressive mindset, and ability to chase targets under pressure. Jasprit Bumrah is India's premier fast bowler, famous for his unique bowling action, pinpoint yorkers, and effectiveness across all formats. 
                    Together, Kohli's batting excellence and Bumrah's bowling brilliance have played a crucial role in many of India's victories on the international stage.
        """),
    metadata = ({"team" : "India"})
)

doc3 = Document(
    page_content= ("""
        MS Dhoni is one of India's most successful cricket captains, known for his calm leadership, finishing abilities, and exceptional wicketkeeping skills. Jasprit Bumrah is India's premier fast bowler, renowned for his unique bowling action, deadly yorkers, and ability to perform under pressure. 
                   Together, Dhoni's strategic mindset and Bumrah's match-winning bowling have contributed significantly to India's success in international cricket.
        """),
    metadata = ({"team" : "India"})
)

docs = [doc1, doc2, doc3]

vector_store = FAISS.from_documents(
    documents= docs,
    embedding= hf_embeddings
)

retriever = vector_store.as_retriever(
    search_type = "mmr",
    search_kwargs = {"k":2, "lambda_mult" : 0.6}
)
compressor = LLMChainExtractor.from_llm(model)

compression_retriever = ContextualCompressionRetriever(
    base_retriever= retriever,
    base_compressor= compressor
)

user_query = "Tell me about MS Dhoni."
result = compression_retriever.invoke(user_query)

for i, doc in enumerate(result):
    print(f"\n--- Result {i+1} ---")
    print(f"Content: - {doc.page_content}")
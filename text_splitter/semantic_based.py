from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

text_sample = """Interview Definition

A code snippet is a small, reusable block of source code that demonstrates a particular functionality, 
logic, or programming concept. Developers often use code snippets for quick reference, testing, debugging, or sharing examples."""


embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')

text_splitter = SemanticChunker(
    embeddings = embeddings,
    breakpoint_threshold_type= "percentile"
)

print(text_splitter.create_documents([text_sample]))
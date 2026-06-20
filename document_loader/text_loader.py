from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm_model= HuggingFaceEndpoint(
    repo_id = "Qwen/Qwen2.5-72B-Instruct"
)

model = ChatHuggingFace(llm = llm_model)

prompt = PromptTemplate(
    template= "Generate a one liner from the below document: {document}",
    input_variables= ['document']
)

parser = StrOutputParser()

document = TextLoader("./document_loader/langchain.txt", encoding= 'utf-8')
loader = document.load()

# print(loader[0].page_content)

chain = prompt | model | parser

print(chain.invoke({'document' : loader[0].page_content}))


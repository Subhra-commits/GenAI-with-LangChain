from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm_model = HuggingFaceEndpoint(
    repo_id = "Qwen/Qwen2.5-72B-Instruct"
)

model = ChatHuggingFace(llm = llm_model)

prompt1 = PromptTemplate(
    template = "Generate a detailed report on {topic}",
    input_variables= ['topic']
)

prompt2 = PromptTemplate(
    template = "Find the few key persons from the below report \n {report}",
    input_variables= ['report']
)

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

print(chain.invoke({'topic' : 'cricket'}))
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm_model = HuggingFaceEndpoint(
    repo_id = "Qwen/Qwen2.5-72B-Instruct"
)

model = ChatHuggingFace(llm = llm_model)

# 1st Prompt
template1 = PromptTemplate(
    template = "Write a detailed report on: {topic}",
    input_variables= ['topic'],
    validate_template= True
)

# 2nd Prompt
template2 = PromptTemplate(
    template= "Write a 5 line summary on the below text: {text}",
    input_variables= ['text'],
    validate_template= True
)

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({'topic': 'black hole'})
print(result)

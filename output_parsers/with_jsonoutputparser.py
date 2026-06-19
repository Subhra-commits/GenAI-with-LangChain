from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

llm_model = HuggingFaceEndpoint(
    repo_id = "Qwen/Qwen2.5-72B-Instruct"
)

model = ChatHuggingFace(llm = llm_model)

parser = JsonOutputParser()

# 1st Prompt
template1 = PromptTemplate(
    template = "Give me name, age and origin of Captain America. \n {format_structure}",
    input_variables= [],
    partial_variables= {'format_structure' : parser.get_format_instructions()},
    validate_template= True
)

chain = template1 | model | parser
result = chain.invoke({})
print(result)
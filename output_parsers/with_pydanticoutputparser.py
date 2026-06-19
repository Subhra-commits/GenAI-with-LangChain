from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()

llm_model = HuggingFaceEndpoint(
    repo_id = "Qwen/Qwen2.5-72B-Instruct"
)

model = ChatHuggingFace(llm = llm_model)

class Person(BaseModel):
    name : str = Field(description="Name of the person")
    age : int = Field(gt= 18, description="Age of the person")
    city : str = Field (description="City from which person belongs to")

parser = PydanticOutputParser(pydantic_object=Person)

# 1st Prompt
template1 = PromptTemplate(
    template = "Give me name, age and city of an famous {nationality} person. \n {format_structure}",
    input_variables= ['nationality'],
    partial_variables= {'format_structure' : parser.get_format_instructions()},
    validate_template= True
)

chain = template1 | model | parser
result = chain.invoke({'nationality' : 'American'})
print(result)
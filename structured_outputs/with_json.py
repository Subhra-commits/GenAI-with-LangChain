from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import TypedDict, Annotated, Optional

load_dotenv()

llm_model = HuggingFaceEndpoint(
    repo_id = "Qwen/Qwen2.5-72B-Instruct"
)

model = ChatHuggingFace(llm = llm_model)

# Define schema
json_schema = {
    "title" : "Review", 
    "type" : "Object", 
    "properties" : {
        "name" : {
            "type" : ["string", "null"],
            "description" : "Name of the reviewer"
        },
        "summary" : {
            "type" : "string", 
            "description" : "Brief summary of the review"
        }, 
        "sentiment" : {
            "type" : "string", 
            "description" : "positive, negative or neutral"
        }
    },
    "required" : ["summary", "sentiment"]
}


structured_model = model.with_structured_output(json_schema)

result = structured_model.invoke("""Iphones are good. Reviewed by Subhradeep""")
print(result)
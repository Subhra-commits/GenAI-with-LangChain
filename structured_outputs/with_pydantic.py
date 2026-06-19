from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import TypedDict, Annotated, Optional

load_dotenv()

llm_model = HuggingFaceEndpoint(
    repo_id = "Qwen/Qwen2.5-32B-Instruct"
)

model = ChatHuggingFace(llm = llm_model)

# Define schema
class Review(BaseModel):
    name: Optional[str] = Field(
        default="Anonymous",
        description="Name of the reviewer"
    )

    summary: str = Field(
        description="Brief summary of the review"
    )

    sentiment: str = Field(
        description="positive, negative or neutral"
    )


structured_model = model.with_structured_output(Review)

result = structured_model.invoke("""Iphones are good.""")
print(result)
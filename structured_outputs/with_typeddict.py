from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional

load_dotenv()

llm_model = HuggingFaceEndpoint(
    repo_id = "Qwen/Qwen2.5-72B-Instruct"
)

model = ChatHuggingFace(llm = llm_model)

# Define schema
class Review(TypedDict):
    name: Annotated[Optional[str], "Name of the reviewer"]
    summary: Annotated[str, "Brief summary of the review"]
    sentiment: Annotated[str, "Sentiment of the review - positive, negative or neutral"]


structured_model = model.with_structured_output(Review)

result = structured_model.invoke("""Iphones are good.""")
print(result)
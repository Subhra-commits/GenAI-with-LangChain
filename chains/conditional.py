from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

llm_model_1 = HuggingFaceEndpoint(
    repo_id = "Qwen/Qwen2.5-72B-Instruct"
)

model_1 = ChatHuggingFace(llm = llm_model_1)

class Feedback(BaseModel):
    sentiment: Literal['Positive', 'Negative'] = Field(description= "user sentiment about feedback")


pydantic_parser = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template = "Find the user sentiment from the feeback: {feedback} \n {format_structure}",
    input_variables= ['feedback'],
    partial_variables= {'format_structure' : pydantic_parser.get_format_instructions()}
)

sentiment_chain = prompt1 | model_1 | pydantic_parser

prompt2 = PromptTemplate(
    template = "Write an appropriate response on : {feedback} feedback",
    input_variables= ['feedback']
)

branch_chain = RunnableBranch(
    (lambda x:x.sentiment == 'Positive', prompt2 | model_1 | StrOutputParser()),
    (lambda x:x.sentiment == 'Negative', prompt2 | model_1 | StrOutputParser()),
    RunnableLambda(lambda x:'Could not find sentiment')
)

final_chain = sentiment_chain | branch_chain

print(final_chain.invoke({'feedback': 'Iphone is a so-so smartphone. And this is a Neutral feedback.'}))
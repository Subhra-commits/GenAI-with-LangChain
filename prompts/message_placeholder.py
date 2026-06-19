from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage
from dotenv import load_dotenv

load_dotenv()

llm_model = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation", 
    temperature= 0.7
)

model = ChatHuggingFace(llm = llm_model)

chat_template = ChatPromptTemplate([
    ('system', 'You are s smart AI assistent'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human', '{query}')
])

chat_history = []

with open('prompts/chat_hist.txt') as f:
    chat_history.extend(f.readlines())

prompt = chat_template.invoke({
    'chat_history' : chat_history,
    'query' : 'What is the status of my refund?'
})

result = model.invoke(prompt)
print(AIMessage(content = result.content))
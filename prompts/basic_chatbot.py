from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

llm_model = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation", 
    temperature= 0.7
)

model = ChatHuggingFace(llm = llm_model)

chat_history = [
    SystemMessage(content='You are a smart AI assistant')
]

while True:
    user_query = input('You: ')
    chat_history.append(HumanMessage(content= user_query))
    if user_query.lower() == 'exit':
        break

    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    print('AI: ', result.content)

print(chat_history)
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import load_prompt


load_dotenv()

llm_model = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation", 
    temperature= 0.7
)

model = ChatHuggingFace(llm = llm_model)


# streamlit dropdowns
role = st.selectbox("Selcet your role", ['Trainer', 'Mentor', 'Student'])
topic = st.selectbox("Select the topic", ['Java', 'Python', 'C++', 'C'])
level = st.selectbox("Select difficulty level", ['Beginner', 'Intermediate', 'Advanced', 'Expert'])

# template
prompt_template = load_prompt("prompts/prompt_template.json")

# Fill the placholder
prompt = prompt_template.invoke({
    'role' : role, 
    'topic' : topic,
    'level' : level
})


if st.button("Result"):
    result = model.invoke(prompt)
    st.write(result.content)

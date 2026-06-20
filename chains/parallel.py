from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()

llm_model_1 = HuggingFaceEndpoint(
    repo_id = "Qwen/Qwen2.5-72B-Instruct"
)

model_1 = ChatHuggingFace(llm = llm_model_1)


prompt1 = PromptTemplate(
    template = "Generate a detailed report on - {topic}",
    input_variables= ['topic']
)

prompt2 = PromptTemplate(
    template = "Generate 5 important qestions and answers from the following report: \n {topic}",
    input_variables= ['topic']
)

prompt3 = PromptTemplate(
    template = "Merge the below repot and quiz into a single document: \n Report -> {report} \n Quiz -> {quiz}",
    input_variables= ['report', 'quiz'],
    validate_template= True
)

parallel_chain = RunnableParallel(
    report = prompt1 | model_1 | StrOutputParser(),
    quiz = prompt2 | model_1 | StrOutputParser()
)

merge_chain = prompt3 | model_1 | StrOutputParser()

chain = parallel_chain | merge_chain

print(chain.invoke({'topic':'cricket'}))

chain.get_graph().print_ascii()

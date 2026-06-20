from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnableSequence, RunnableLambda, RunnablePassthrough, RunnableParallel

load_dotenv()

llm_model= HuggingFaceEndpoint(
    repo_id = "Qwen/Qwen2.5-72B-Instruct"
)

model = ChatHuggingFace(llm = llm_model)

# Generate our prompt template and output parser
prompt = PromptTemplate(
    template = "Generate a joke on : {topic}",
    input_variables= ['topic']
)

parser = StrOutputParser()

# Create a runnable sequence that generates a joke
joke_generator = RunnableSequence(prompt, model, parser)

# create a custom function to count the number of words in the joke
def word_counter(text):
    return len(text.split())

# Create a parallel runnable that passes through a joke and counts the number of words in the joke
parallel_chain = RunnableParallel(
    joke = RunnablePassthrough(),
    word_count = RunnableLambda(word_counter)
)

final_chain = joke_generator | parallel_chain
result = final_chain.invoke({'topic' : 'Technology'})

print("Joke: ", result['joke'])
print("Word Count: ", result['word_count'])
from langchain_text_splitters import CharacterTextSplitter

text = """What is a Chain in LangChain?
A Chain is a workflow that connects multiple LangChain components together, where the output of one component becomes the input of the next component.

Think of it as a pipeline."""

splitter = CharacterTextSplitter(
    separator= '', 
    chunk_size = 50,
    chunk_overlap = 5
)

print(splitter.split_text(text))


from langchain_text_splitters import RecursiveCharacterTextSplitter

text = """What is a Chain in LangChain?
A Chain is a workflow that connects multiple LangChain components together, where the output of one component becomes the input of the next component.

Think of it as a pipeline."""

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 50,
    chunk_overlap = 0
)

chunks = splitter.split_text(text)

print(len(chunks))
print(chunks)


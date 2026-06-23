from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

text = """employee = {
    "name": "John",
    "role": "Data Engineer",
    "experience": 4
}

for key, value in employee.items():
    print(f"{key}: {value}")"""

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size = 100,
    chunk_overlap = 0
)

chunks = splitter.split_text(text)

print(len(chunks))
print(chunks)


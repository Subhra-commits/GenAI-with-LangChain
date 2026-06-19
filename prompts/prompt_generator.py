from langchain_core.prompts import PromptTemplate

prompt_template = PromptTemplate(
    template = """
            You are a {role}. Explain the topic "{topic}" in a {level} manner.
        """,
    input_variables = ['role', 'topic', 'level'],
    validate_template = True
)

prompt_template.save("prompts/prompt_template.json")
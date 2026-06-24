# GenAI-with-LangChain

Hands-on exercises and practical examples to learn LangChain. This repository collects small, focused examples for working with chat models, embedding models, prompts, tool agents, RAG pipelines, retrievers, and integrations with closed-source providers (OpenAI, Anthropic, Google) as well as open-source models from Hugging Face.

## Features

- Practical example scripts for building LangChain pipelines
- Examples for chat & embedding models, prompt engineering, and output parsers
- Retriever and vector-store examples (Chroma, FAISS)
- Streamlit demo apps illustrating basic and advanced usage
- Utilities for document loading and text splitting

## Quick Start

1. Create and activate a Python environment (recommended).

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Try a simple runnable (example):

```bash
# Run a small example script
python Runnables/joke_generator_word_counter.py

# Or run the basic Streamlit demo (Windows PowerShell example)
streamlit run "StreamLit/Basic Concepts/first_app.py"
```

Note: Some examples require API keys or additional environment configuration for cloud models. Check the individual module docstrings or source files for required environment variables.

## Repository Structure

- `models/` — Model integration examples (chat, embeddings, similarity search).
- `prompts/` — Reusable prompt templates and generators.
- `structured_outputs/` — More structured output examples.
- `chains/` — Example LangChain flows: `conditional.py`, `parallel.py`, `sequential.py`.
- `output_parsers/` — Examples of structured output parsing (JSON, Pydantic, strings).
- `Runnables/` — Small runnable scripts demonstrating focused functionality.

- `document_loader/` — Helpers and loaders for text, PDF, and web content.
- `text_splitter/` — Different text-splitting strategies for document processing.
- `vector_db/` — Example vector store adapters (Chroma, FAISS).
- `retriever/` — Retriever examples and strategies (MMR, multi-query, contextual compression).

- `StreamLit/` — Streamlit demo apps (Basic and Advanced directories).



## How to Use

- Explore the `Runnables/` and `StreamLit/` folders for runnable examples.
- Open loader and model files in `document_loader/` and `models/` to see how documents are indexed and how embeddings/search are performed.
- Use the prompt templates in `prompts/` to adapt example prompts for your models.

## Contributing

Contributions and improvements are welcome. Please open issues or PRs with focused changes or example additions.

## Notes

- This repository is intended as a learning resource and collection of small examples — not a production-ready project. Review and adapt examples before using in production.

----

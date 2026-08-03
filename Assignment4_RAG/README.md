# Assignment 4 — Retrieval-Augmented Generation

This project builds a small question-answering system over a cat-facts knowledge base. It compares local generation models, embedding models, and prompts using a ten-question case-insensitive substring evaluation.

## Pipeline

1. Load one cat fact per line as a LangChain document.
2. Embed documents and store them in Chroma.
3. Retrieve context with maximal marginal relevance (MMR).
4. Generate an answer with a local Ollama model.
5. Check whether an accepted answer string appears in the response.

The default tracked configuration uses:

- Generator: `llama3.2:3b`
- Embeddings: `jinaai/jina-embeddings-v2-base-en`
- Vector store: Chroma
- Retrieval: MMR

## Reported results

### Generator and prompt comparison

| Generator | Best correct answers (out of 10) |
| --- | ---: |
| `llama3.2:1b` | 8 |
| `gemma2:2b` | 7 |
| `llama3.2:3b` | **9** |
| `phi3:mini` | 7 |

### Embedding comparison with `llama3.2:3b`

| Embedding model | Correct answers |
| --- | ---: |
| `bert-base-multilingual-uncased` | 3 |
| `bert-base-multilingual-cased` | 4 |
| `sentence-transformers/all-MiniLM-L12-v2` | 4 |
| `sentence-transformers/paraphrase-MiniLM-L6-v2` | **8** |
| `distilbert-base-uncased` | 6 |
| `distilbert-base-cased` | 4 |
| `jinaai/jina-embedding-b-en-v1` | 7 |
| `all-mpnet-base-v2` | 7 |
| `FacebookAI/roberta-base` | 4 |
| `FacebookAI/roberta-large` | 2 |
| `FacebookAI/xlm-roberta-base` | 5 |

Without retrieval, the tracked `llama3.2:3b` run answered 4 of 10 questions correctly. These figures are transcribed from the course report and were not regenerated during repository cleanup.

## Run locally

1. Install and start [Ollama](https://ollama.com/), then pull the default model:

   ```bash
   ollama pull llama3.2:3b
   ```

2. Install the Python dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

3. Download the assignment's `cat-facts.txt` into this directory.
4. Run the evaluation:

   ```bash
   python main.py --facts cat-facts.txt --output results.json
   ```

Use `--model` or `--embedding-model` to select another tracked configuration. The current script requires no hard-coded Hugging Face token and does not save credentials.

## Limitations

- The knowledge-base file and local Ollama weights are not tracked.
- Substring matching is a lightweight course metric; it does not measure factual completeness or answer quality.
- Local model output can vary with runtime and model versions, so exact counts may differ from the historical report.

## Attribution

Starter code and data instructions: [IKMLab NTHU NLP Assignment 4](https://github.com/IKMLab/NTHU_Natural_Language_Processing/tree/main/Assignments/Assignment4).

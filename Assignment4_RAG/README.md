# CS563100 Natural Language Processing - Assignment 4: Retrieval-Augmented Generation with LangChain

## Experimental Setup

**Running Environment:** Colab
**Python Version:** Colab

## Experiments

### RAG System Components

My RAG system incorporates the following key components:

* **Retrieval:**
    * **Vector Database:** Chroma
    * **Retrieval Method:** MMR
    * **Embedding Models (Evaluated):** Jina AI (`jinaai/jina-embeddings-v2-base-en`, `jinaai/jina-embedding-b-en-v1`), Sentence Transformers (`all-MiniLM-L12-v2`, `paraphrase-MiniLM-L6-v2`), BERT (`bert-base-multilingual-cased`, `bert-base-multilingual-uncased`), Hugging Face (`distilbert-base-cased/uncased`, `roberta-base`, `roberta-large`, `xlm-roberta-base`, `all-mpnet-base-v2`).
* **Generation Model (Evaluated):** Llama 3.2 (`llama3.2:1b`, `llama3.2:3b`), Gemma 2 (`gemma2:2b`), Phi (`phi3:mini`).

The primary retrieval model used in the experiments was `jinaai/jina-embeddings-v2-base-en`.

### Prompt Engineering

Eight different prompt scenarios were tested, each appended with the common suffix: "Answer the question shortly and professionally using the most relevant information from the provided documents." Table 1-4 detail the prompts and the number of correct answers obtained using `llama3.2:1b`, `gemma2:2b`, `llama3.2:3b`, and `phi3:mini` respectively. Subsequent experiments utilized `llama3.2:3b` and the prompt: "You are an expert on feline biology and behavior. Your role is to provide precise and accurate answers to questions about cats based on the most relevant context from the provided documents."

**TABLE 1.** Prompt Performance with `llama3.2:1b`
| Prompt                                                                                                                                                                                                                            | Correct numbers |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------|
| "You are a highly knowledgeable assistant. Your task is to answer questions about cats. "                                                                                                                                        | 7               |
| "You are an expert on feline biology and behavior. Your role is to provide precise and accurate answers to questions about cats based on the most relevant context from the provided documents."                                   | 8               |
| "You are a concise and precise assistant. Answer the questions about cats using the context provided, keeping your answers as brief and accurate as possible."                                                                     | 5               |
| "You are a knowledgeable assistant specializing in cat-related information. Use the provided context to answer questions in a bullet-point format, emphasizing key facts."                                                          | 7               |
| "You are a cat expert educating an audience. Provide comprehensive and easy-to-understand answers about cats, using information from the provided documents."                                                                      | 7               |
| "You are a detail-oriented assistant. When answering questions about cats, provide not only the answer but also an explanation of the reasons or mechanisms behind it, using the context provided."                               | 7               |
| "Imagine you are a cat explaining your life and behavior. Use the provided context to answer questions in a way that reflects a cat's perspective."                                                                                | 7               |
| "You are a storyteller specializing in cats. When answering questions, incorporate a short narrative to explain the information in an engaging way."                                                                               | 7               |

**TABLE 2.** Prompt Performance with `gemma2:2b`
| Prompt                                                                                                                                                                                                                            | Correct numbers |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------|
| "You are a highly knowledgeable assistant. Your task is to answer questions about cats."                                                                                                                                         | 6               |
| "You are an expert on feline biology and behavior. Your role is to provide precise and accurate answers to questions about cats based on the most relevant context from the provided documents."                                   | 6               |
| "You are a concise and precise assistant. Answer the questions about cats using the context provided, keeping your answers as brief and accurate as possible."                                                                     | 6               |
| "You are a knowledgeable assistant specializing in cat-related information. Use the provided context to answer questions in a bullet-point format, emphasizing key facts."                                                          | 7               |
| "You are a cat expert educating an audience. Provide comprehensive and easy-to-understand answers about cats, using information from the provided documents."                                                                      | 7               |
| "You are a detail-oriented assistant. When answering questions about cats, provide not only the answer but also an explanation of the reasons or mechanisms behind it, using the context provided."                               | 7               |
| "Imagine you are a cat explaining your life and behavior. Use the provided context to answer questions in a way that reflects a cat's perspective."                                                                                | 1               |
| "You are a storyteller specializing in cats. When answering questions, incorporate a short narrative to explain the information in an engaging way."                                                                               | 5               |

**TABLE 3.** Prompt Performance with `llama3.2:3b`
| Prompt                                                                                                                                                                                                                            | Correct numbers |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------|
| "You are a highly knowledgeable assistant. Your task is to answer questions about cats."                                                                                                                                         | 9               |
| "You are an expert on feline biology and behavior. Your role is to provide precise and accurate answers to questions about cats based on the most relevant context from the provided documents."                                   | 8               |
| "You are a concise and precise assistant. Answer the questions about cats using the context provided, keeping your answers as brief and accurate as possible."                                                                     | 5               |
| "You are a knowledgeable assistant specializing in cat-related information. Use the provided context to answer questions in a bullet-point format, emphasizing key facts."                                                          | 7               |
| "You are a cat expert educating an audience. Provide comprehensive and easy-to-understand answers about cats, using information from the provided documents."                                                                      | 7               |
| "You are a detail-oriented assistant. When answering questions about cats, provide not only the answer but also an explanation of the reasons or mechanisms behind it, using the context provided."                               | 7               |
| "Imagine you are a cat explaining your life and behavior. Use the provided context to answer questions in a way that reflects a cat's perspective."                                                                                | 7               |
| "You are a storyteller specializing in cats. When answering questions, incorporate a short narrative to explain the information in an engaging way."                                                                               | 7               |

**TABLE 4.** Prompt Performance with `phi3:mini`
| Prompt                                                                                                                                                                                                                            | Correct numbers |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------|
| "You are a highly knowledgeable assistant. Your task is to answer questions about cats."                                                                                                                                         | 5               |
| "You are an expert on feline biology and behavior. Your role is to provide precise and accurate answers to questions about cats based on the most relevant context from the provided documents."                                   | 7               |
| "You are a concise and precise assistant. Answer the questions about cats using the context provided, keeping your answers as brief and accurate as possible."                                                                     | 7               |
| "You are a knowledgeable assistant specializing in cat-related information. Use the provided context to answer questions in a bullet-point format, emphasizing key facts."                                                          | 7               |
| "You are a cat expert educating an audience. Provide comprehensive and easy-to-understand answers about cats, using information from the provided documents."                                                                      | 7               |
| "You are a detail-oriented assistant. When answering questions about cats, provide not only the answer but also an explanation of the reasons or mechanisms behind it, using the context provided."                               | 6               |
| "Imagine you are a cat explaining your life and behavior. Use the provided context to answer questions in a way that reflects a cat's perspective."                                                                                | 7               |
| "You are a storyteller specializing in cats. When answering questions, incorporate a short narrative to explain the information in an engaging way."                                                                               | 7               |




The highest score achieved across the ten questions was 9 correct answers, obtained with `llama3.2:3b` as the generation model, `jina-embeddings-v2-base-en` as the retrieval model, and the prompt: "You are a highly knowledgeable assistant. Your task is to answer questions about cats. Answer the question shortly and professionally using the most relevant information from the provided documents."

### RAG Performance Analysis with Different Prompts

Table 5 shows the number of prompts that resulted in incorrect answers for each question when using `llama3.2:3b`. Questions 3 and 5 were the most frequently answered incorrectly across different prompts (a similar trend was observed with `phi3:mini`), while Question 10 had the fewest incorrect answers.

**TABLE 5.** Incorrect Answers per Question with `llama3.2:3b`

| QuestionID | # of prompts that make the mistake |
|------------|-----------------------------------|
| 1          | 1                                 |
| 2          | 0                                 |
| 3          | 5                                 |
| 4          | 2                                 |
| 5          | 6                                 |
| 6          | 0                                 |
| 7          | 1                                 |
| 8          | 3                                 |
| 9          | 1                                 |
| 10         | 3                                 |

### Comparison with Different Retrieval Models and Without RAG

The RAG performance was compared using various retrieval models (with `llama3.2:3b` as the generator):

* `bert-base-multilingual-uncased`: 3 correct answers
* `bert-base-multilingual-cased`: 4 correct answers
* `sentence-transformers/all-MiniLM-L12-v2`: 4 correct answers
* `sentence-transformers/paraphrase-MiniLM-L6-v2`: **8 correct answers**
* `distilbert-base-uncased`: 6 correct answers
* `distilbert-base-cased`: 4 correct answers
* `jinaai/jina-embedding-b-en-v1`: 7 correct answers
* `all-mpnet-base-v2`: 7 correct answers
* `FacebookAI/roberta-base`: 4 correct answers
* `FacebookAI/roberta-large`: 2 correct answers
* `FacebookAI/xlm-roberta-base`: 5 correct answers

**Without using RAG** (only `llama3.2:3b`), the number of correct answers was 4. The language model's direct responses often deviated from the expected answers and provided incorrect information. 

## Reference

Sample code / Data: [IKMLab Github](https://github.com/IKMLab/NTHU_Natural_Language_Processing/tree/main/Assignments/Assignment4)
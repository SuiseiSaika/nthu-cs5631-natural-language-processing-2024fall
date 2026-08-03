# Natural Language Processing — Fall 2024

Coursework for NTHU CS5631, taught by Prof. Hung-Yu Kao. The repository covers distributional semantics, character-level sequence generation, multi-task sentence understanding, retrieval-augmented generation, and a team project on mathematical misconception retrieval.

## Projects

| Project | Focus | Main approach | Reported outcome |
| --- | --- | --- | --- |
| [Assignment 1](Assignment1_Word_Analogy/) | Word analogy | Pretrained GloVe and custom Word2Vec | GloVe-200: 73.44% semantic, 67.02% syntactic accuracy |
| [Assignment 2](Assignment2_Arithmetic_as_a_Language/) | Arithmetic as language | Character-level RNN, GRU, and LSTM | GRU/LSTM substantially outperform the vanilla RNN in the tracked plots |
| [Assignment 3](Assignment3_Multi-output_Learning/) | Multi-output learning | BERT with relatedness and entailment heads | Test Spearman 0.8395; entailment accuracy 0.8815 |
| [Assignment 4](Assignment4_RAG/) | Retrieval-augmented generation | Chroma, MMR retrieval, local Ollama models | Best tracked run: 9/10 exact-match questions |
| [Term project](Term_Project/NTHU_NLP_Final%28Group13%29.pdf) | Mathematical misconception retrieval | Embedding retrieval and multi-stage reranking | Team report preserved as submitted |

The numeric results above are transcribed from the tracked reports and figures. They were not regenerated during repository cleanup because the required datasets, model weights, and compute environments are not bundled.

## Repository layout

```text
.
├── Assignment1_Word_Analogy/
├── Assignment2_Arithmetic_as_a_Language/
├── Assignment3_Multi-output_Learning/
├── Assignment4_RAG/
└── Term_Project/
```

Each assignment directory contains its own dependency list, implementation, experiment report, and available figures. Start with the assignment README for data and runtime requirements.

## Reproduction scope

- Large datasets, pretrained weights, generated checkpoints, and caches are intentionally excluded.
- Assignment 1 requires the Google analogy data and a prepared Wikipedia text corpus.
- Assignment 2 requires the course arithmetic CSV files.
- Assignment 3 downloads SemEval 2014 Task 1 and BERT through Hugging Face.
- Assignment 4 requires a local Ollama server plus the separately downloaded cat-facts file.
- The term-project PDF is a team-authored report; its implementation and competition data are not included.

## Attribution and license

Assignment starter code and data instructions are attributed to the [IKMLab NTHU NLP course repository](https://github.com/IKMLab/NTHU_Natural_Language_Processing). Third-party datasets, models, and libraries remain subject to their original terms.

This repository does not assert a project-wide software license. It is published for portfolio and educational review.

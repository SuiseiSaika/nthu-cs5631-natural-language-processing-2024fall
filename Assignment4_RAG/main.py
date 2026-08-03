"""Evaluate a local Ollama model with retrieval over the cat-facts dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.llms import Ollama
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings


DEFAULT_MODEL = "llama3.2:3b"
DEFAULT_EMBEDDING_MODEL = "jinaai/jina-embeddings-v2-base-en"
DEFAULT_PROMPT = (
    "You are an expert on feline biology and behavior. Answer briefly and "
    "accurately using only the most relevant information in the supplied context."
)

QUERIES = (
    "How much of a day do cats spend sleeping on average?",
    "What is the technical term for a cat's hairball?",
    "What do scientists believe caused cats to lose their sweet tooth?",
    "What is the top speed a cat can travel over short distances?",
    "What is the name of the organ in a cat's mouth that helps it smell?",
    "Which wildcat is considered the ancestor of all domestic cats?",
    "What is the group term for cats?",
    "How many different sounds can cats make?",
    "What is the name of the first cat in space?",
    "How many toes does a cat have on its back paws?",
)

EXPECTED_ANSWERS: tuple[str | tuple[str, ...], ...] = (
    "2/3",
    "Bezoar",
    "a mutation in a key taste receptor",
    ("31 mph", "49 km"),
    "Jacobson’s organ",
    "the African Wild Cat",
    "clowder",
    "100",
    ("Felicette", "Astrocat"),
    "four",
)


def load_documents(path: Path) -> list[Document]:
    """Load one fact per non-empty line."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Cat-facts file not found: {path}. See README.md for download instructions."
        )
    facts = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    return [
        Document(page_content=fact, metadata={"id": index})
        for index, fact in enumerate(facts)
        if fact
    ]


def build_chain(
    documents: list[Document], model_name: str, embedding_model_name: str
):
    """Build the MMR retriever and retrieval-augmented generation chain."""
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs={"trust_remote_code": True},
        encode_kwargs={"normalize_embeddings": False},
    )
    vector_store = Chroma.from_documents(documents=documents, embedding=embeddings)
    retriever = vector_store.as_retriever(search_type="mmr")
    llm = Ollama(model=model_name)
    prompt = ChatPromptTemplate.from_messages(
        (
            ("system", DEFAULT_PROMPT + "\n\nContext:\n{context}"),
            ("human", "{input}"),
        )
    )
    document_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    return create_retrieval_chain(retriever=retriever, combine_docs_chain=document_chain)


def answer_is_correct(response: str, expected: str | tuple[str, ...]) -> bool:
    """Apply the assignment's case-insensitive substring evaluation."""
    candidates = (expected,) if isinstance(expected, str) else expected
    lowered = response.lower()
    return any(candidate.lower() in lowered for candidate in candidates)


def evaluate(chain) -> dict[str, object]:
    """Run the ten tracked questions and return serializable results."""
    results: list[dict[str, object]] = []
    correct = 0
    for question_id, (query, expected) in enumerate(
        zip(QUERIES, EXPECTED_ANSWERS), start=1
    ):
        response = chain.invoke({"input": query})["answer"]
        is_correct = answer_is_correct(response, expected)
        correct += int(is_correct)
        results.append(
            {
                "question_id": question_id,
                "query": query,
                "response": response,
                "correct": is_correct,
            }
        )
        print(f"[{question_id:02d}] {'correct' if is_correct else 'incorrect'}: {response}")
    return {"correct": correct, "total": len(QUERIES), "results": results}


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--facts", type=Path, default=project_dir / "cat-facts.txt"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    documents = load_documents(args.facts)
    chain = build_chain(documents, args.model, args.embedding_model)
    summary = evaluate(chain)
    summary.update({"model": args.model, "embedding_model": args.embedding_model})
    print(f"Score: {summary['correct']}/{summary['total']}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

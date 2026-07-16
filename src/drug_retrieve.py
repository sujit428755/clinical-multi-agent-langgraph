"""
Retrieval module for the FDA Drug Label RAG Assistant.

Loads the TF-IDF index built by ingest.py and returns the top-k most
similar chunks for a given query, along with a similarity score. A
minimum-similarity threshold is used to support an honest "not found in
provided labels" fallback rather than forcing the generator to answer from
weak or irrelevant context.
"""
import json
import pickle
from dataclasses import dataclass
from pathlib import Path

from sklearn.metrics.pairwise import cosine_similarity

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "drug_rag_data" / "processed"

# Below this cosine similarity, we treat retrieval as "no relevant match"
MIN_SIMILARITY = 0.05


@dataclass
class RetrievedChunk:
    chunk_id: str
    drug: str
    section: str
    text: str
    source_url: str
    score: float


class Retriever:
    def __init__(self):
        with open(PROCESSED_DIR / "vectorizer.pkl", "rb") as f:
            self.vectorizer = pickle.load(f)
        with open(PROCESSED_DIR / "matrix.pkl", "rb") as f:
            self.matrix = pickle.load(f)
        with open(PROCESSED_DIR / "chunks.json", "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.matrix).flatten()
        top_indices = sims.argsort()[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(sims[idx])
            if score < MIN_SIMILARITY:
                continue
            c = self.chunks[idx]
            results.append(RetrievedChunk(
                chunk_id=c["chunk_id"],
                drug=c["drug"],
                section=c["section"],
                text=c["text"],
                source_url=c["source_url"],
                score=score,
            ))
        return results

    def list_drugs(self) -> list[str]:
        return sorted(set(c["drug"] for c in self.chunks))


if __name__ == "__main__":
    # Quick manual smoke test
    r = Retriever()
    print("Drugs in corpus:")
    for d in r.list_drugs():
        print(" -", d)

    test_queries = [
        "Can I take metformin with furosemide?",
        "What is the max dose of atorvastatin with clarithromycin?",
        "Is amoxicillin safe with warfarin?",
        "What is the capital of France?",  # should trigger low-similarity fallback
    ]
    for q in test_queries:
        print(f"\nQuery: {q}")
        hits = r.retrieve(q, top_k=3)
        if not hits:
            print("  -> NOT FOUND IN PROVIDED LABELS (below similarity threshold)")
        for h in hits:
            print(f"  [{h.score:.3f}] {h.drug} :: {h.section}")

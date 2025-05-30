import json
import pickle
import numpy as np
import string
from rank_bm25 import BM25Okapi
from pyvi.ViTokenizer import tokenize
from sentence_transformers import SentenceTransformer


def split_text(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.lower().split()
    return [w for w in words if w.strip()]


class Retriever:
    def __init__(self, corpus, corpus_emb_path, model_name):
        self.corpus = corpus
        self.corpus_emb_path = corpus_emb_path
        self.model_name = model_name

        # Load model embedding
        self.embedder = SentenceTransformer(model_name)

        # Load embedding trực tiếp ở đây
        with open(self.corpus_emb_path, "rb") as f:
            self.embeddings = pickle.load(f)

        # Normalize embeddings
        self.embeddings = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        
        # BM25
        # for doc in self.corpus:
        #     if "passage" not in doc:
        #         doc["passage"] = doc.get("context", "")
        self.tokenized_corpus = [split_text(doc["passage"]) for doc in self.corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def retrieve(self, question: str, topk: int = 10):
        segmented_question = tokenize(question)
        query_emb = self.embedder.encode([segmented_question])[0]
        query_emb = query_emb / np.linalg.norm(query_emb)

        sim_scores = np.dot(self.embeddings, query_emb)

        tokenized_query = split_text(question)
        bm25_scores = self.bm25.get_scores(tokenized_query)
        max_bm25, min_bm25 = max(bm25_scores), min(bm25_scores)
        normalize = lambda x: (x - min_bm25 + 0.1) / (max_bm25 - min_bm25 + 0.1)

        results = []
        for i, doc in enumerate(self.corpus):
            bm25_score = bm25_scores[i]
            bm25_normed = normalize(bm25_score)
            sem_score = sim_scores[i]
            combined_score = bm25_normed * 0.4 + sem_score * 0.6
            results.append({
                "id": doc["id"],
                "title": doc.get("title", ""),
                "context": doc.get("context", ""),
                "bm25_score": bm25_score,
                "semantic_score": sem_score,
                "combined_score": combined_score
            })

        return sorted(results, key=lambda x: x["combined_score"], reverse=True)[:topk]


import os
import json
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity


def _safe_label(label: str) -> str:
    return str(label).strip().upper().replace(" ", "_")


@dataclass
class SimilarityResult:
    label: str
    score: float
    scores: Dict[str, float]
    margin: float


class SimilarityEngine:
    def __init__(
        self,
        model_dir: str,
        lexicon_filename: str = "keyword_dimension.json",
        vectorizer_filename: str = "tfidf_vectorizer_dimensi.joblib",
    ):
        self.model_dir = model_dir
        self.lexicon_path = os.path.join(model_dir, lexicon_filename)
        self.vectorizer_path = os.path.join(model_dir, vectorizer_filename)

        self.labels: List[str] = []
        self.dim_docs: List[str] = []
        self.vectorizer = None
        self.X_dim = None

        self._load_assets()

    def _load_assets(self) -> None:
        if not os.path.exists(self.lexicon_path):
            raise FileNotFoundError(f"Lexicon tidak ketemu: {self.lexicon_path}")

        with open(self.lexicon_path, "r", encoding="utf-8") as f:
            lex = json.load(f)

        if not isinstance(lex, dict):
            raise ValueError("keyword_dimension.json harus dict {DIMENSI: [keyword,...]}")

        self.labels = list(lex.keys())
        self.dim_docs = []
        for lab in self.labels:
            kws = lex.get(lab, [])
            if isinstance(kws, list):
                kws = [str(x).strip().lower() for x in kws if str(x).strip()]
            else:
                kws = [str(kws).strip().lower()] if str(kws).strip() else []
            self.dim_docs.append(" ".join(kws))

        if not os.path.exists(self.vectorizer_path):
            raise FileNotFoundError(
                f"Vectorizer tidak ketemu: {self.vectorizer_path}. "
                f"Pastikan sudah tersimpan sebagai tfidf_vectorizer_dimensi.joblib"
            )

        self.vectorizer = joblib.load(self.vectorizer_path)
        self.X_dim = self.vectorizer.transform(self.dim_docs)

    def score_one(self, text: str) -> SimilarityResult:
        text = "" if text is None else str(text)

        X_text = self.vectorizer.transform([text])
        sim = cosine_similarity(X_text, self.X_dim).ravel()

        scores = {_safe_label(self.labels[i]): float(sim[i]) for i in range(len(self.labels))}

        idx_sorted = np.argsort(sim)[::-1]
        top1 = idx_sorted[0]
        top2 = idx_sorted[1] if len(idx_sorted) > 1 else top1

        best_label = _safe_label(self.labels[top1])
        best_score = float(sim[top1])
        margin = float(sim[top1] - sim[top2]) if len(idx_sorted) > 1 else best_score

        return SimilarityResult(label=best_label, score=best_score, scores=scores, margin=margin)

    def predict_one(self, text: str, unknown_threshold: float = 0.0) -> SimilarityResult:
        res = self.score_one(text)
        if res.score <= float(unknown_threshold):
            return SimilarityResult(label="UNKNOWN", score=res.score, scores=res.scores, margin=res.margin)
        return res

    def predict_many(self, texts: List[str], unknown_threshold: float = 0.0, drop_unknown: bool = False) -> List[SimilarityResult]:
        out: List[SimilarityResult] = []
        for t in texts:
            r = self.predict_one(t, unknown_threshold=unknown_threshold)
            if drop_unknown and r.label == "UNKNOWN":
                continue
            out.append(r)
        return out


import os
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import joblib


@dataclass
class SentimentResult:
    label: str
    score: float
    scores: Dict[str, float]


class SentimentEngine:
    def __init__(
        self,
        model_dir: str,
        vectorizer_filename: str = "tfidf_vectorizer_sentiment.joblib",
        model_filename: str = "logreg_sentiment.joblib",
    ):
        self.vectorizer_path = os.path.join(model_dir, vectorizer_filename)
        self.model_path = os.path.join(model_dir, model_filename)

        if not os.path.exists(self.vectorizer_path):
            raise FileNotFoundError(f"Vectorizer tidak ketemu: {self.vectorizer_path}")

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model tidak ketemu: {self.model_path}")

        self.vectorizer = joblib.load(self.vectorizer_path)
        self.model = joblib.load(self.model_path)

    def predict_one(self, text: str, unknown_threshold: float = 0.5) -> SentimentResult:
        text = "" if text is None else str(text)

        X_text = self.vectorizer.transform([text])
        probas = self.model.predict_proba(X_text)[0]

        idx = np.argmax(probas)
        label = str(self.model.classes_[idx])
        score = float(probas[idx])

        if score < unknown_threshold:
            label = "UNKNOWN"

        scores = {
            str(self.model.classes_[i]): float(probas[i])
            for i in range(len(self.model.classes_))
        }

        return SentimentResult(label=label, score=score, scores=scores)

    def predict_many(
        self,
        texts: List[str],
        unknown_threshold: float = 0.1,
        drop_unknown: bool = False,
    ) -> List[SentimentResult]:

        results: List[SentimentResult] = []

        for t in texts:
            res = self.predict_one(t, unknown_threshold=unknown_threshold)
            if drop_unknown and res.label == "UNKNOWN":
                continue
            results.append(res)

        return results

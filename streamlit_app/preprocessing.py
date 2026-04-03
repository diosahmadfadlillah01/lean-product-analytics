
# preprocessing.py
import re
import json
import pandas as pd
from typing import Dict, Set, Optional, Union, List

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory


def load_slang_map_from_excel(xlsx_path: str) -> Dict[str, str]:
    kamus = pd.read_excel(xlsx_path)

    required_cols = {"tidak_baku", "kata_baku"}
    if not required_cols.issubset(set(kamus.columns)):
        raise ValueError(
            f"Kolom kamus harus ada {required_cols}, tetapi ketemu: {kamus.columns.tolist()}"
        )

    kamus = kamus[["tidak_baku", "kata_baku"]].dropna()
    kamus["tidak_baku"] = kamus["tidak_baku"].astype(str).str.strip().str.lower()
    kamus["kata_baku"]  = kamus["kata_baku"].astype(str).str.strip().str.lower()

    kamus = kamus[
        (kamus["tidak_baku"].str.len() > 0) & (kamus["kata_baku"].str.len() > 0)
    ].copy()

    return dict(zip(kamus["tidak_baku"], kamus["kata_baku"]))


def load_slang_map_from_json(json_path: str) -> Dict[str, str]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return {str(k).lower().strip(): str(v).lower().strip() for k, v in data.items()}


def build_stopwords(extra: Optional[Set[str]] = None, keep: Optional[Set[str]] = None) -> Set[str]:
    factory = StopWordRemoverFactory()
    stopwords = set(factory.get_stop_words())

    if extra:
        stopwords |= set(map(lambda x: str(x).strip().lower(), extra))

    if keep:
        stopwords -= set(map(lambda x: str(x).strip().lower(), keep))

    return stopwords


class Preprocessor:
    def __init__(self, slang_map: Optional[Dict[str, str]] = None, stopwords: Optional[Set[str]] = None):
        self.slang_map = slang_map or {}
        self.stopwords = stopwords or set()
        self.stemmer = StemmerFactory().create_stemmer()

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    def cleaning_casefold(self, text: str) -> str:
        text = "" if text is None else str(text)
        text = text.lower()

        text = re.sub(r"http\S+|www\.\S+", " ", text)
        text = re.sub(r"@\w+", " ", text)
        text = text.replace("#", " ")
        text = re.sub(r"\brt\b", " ", text)

        text = re.sub(r"[^a-z0-9\s]", " ", text)
        return self._normalize_whitespace(text)

    def normalize(self, text: str) -> str:
        text = re.sub(r"(.)\1{2,}", r"\1\1", text)
        text = self._normalize_whitespace(text)
        if not text:
            return ""

        tokens = text.split(" ")
        tokens = [self.slang_map.get(t, t) for t in tokens]
        return " ".join(tokens)

    def tokenize(self, text: str) -> List[str]:
        text = self._normalize_whitespace(text)
        return [] if not text else text.split(" ")

    def stem(self, tokens: List[str]) -> List[str]:
        return [self.stemmer.stem(t) for t in tokens]

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        return [t for t in tokens if t not in self.stopwords and len(t) > 1]

    def preprocess_text(self, raw_text: str) -> str:
        t1 = self.cleaning_casefold(raw_text)
        if not t1:
            return ""

        t2 = self.normalize(t1)
        if not t2:
            return ""

        tokens = self.tokenize(t2)
        if not tokens:
            return ""

        stems = self.stem(tokens)
        final_tokens = self.remove_stopwords(stems)
        return " ".join(final_tokens)

    def preprocess_many(self, texts: Union[List[str], "pd.Series"]) -> List[str]:
        return [self.preprocess_text(t) for t in list(texts)]

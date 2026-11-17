"""English–Bengali similarity for one line or couplet. Load once, then classify.

Tuples become newline-separated text. Overlong input is rejected. Similarity
is not a probability; semantic_match is None unless you supply a threshold.
"""

import math
from numbers import Integral, Real
from pathlib import Path

__all__ = ["SemanticClassifier"]


def _text(value, name):
    if isinstance(value, tuple):
        if len(value) != 2:
            raise ValueError(f"{name} must contain two text lines")
        if any(not isinstance(line, str) for line in value):
            raise TypeError(f"Each {name} line must be a string")
        if any(not line.strip() for line in value):
            raise ValueError(f"{name} contains an empty line")
        value = "\n".join(line.strip() for line in value)
    if not isinstance(value, str):
        raise TypeError(f"{name} must be text or a tuple of two lines")
    if not value.strip():
        raise ValueError(f"{name} must not be empty")
    return value.strip()


def _cosine(vectors):
    if len(vectors) != 2 or not len(vectors[0]) or len(vectors[0]) != len(vectors[1]):
        raise ValueError("Model must return two equal-dimensional embeddings")
    first, second = vectors
    scale = math.sqrt(sum(float(x) ** 2 for x in first) * sum(float(x) ** 2 for x in second))
    if not math.isfinite(scale) or scale == 0:
        raise ValueError("Model returned a zero or nonfinite embedding")
    return max(-1.0, min(1.0, sum(float(a) * float(b) for a, b in zip(first, second)) / scale))


class SemanticClassifier:
    def __init__(self, model, *, threshold=None):
        if not callable(getattr(model, "encode", None)) or not callable(getattr(model, "tokenizer", None)):
            raise TypeError("model must provide encode() and a tokenizer")
        if threshold is not None:
            if isinstance(threshold, bool) or not isinstance(threshold, Real):
                raise TypeError("threshold must be a number or None")
            if not math.isfinite(threshold) or not -1 <= threshold <= 1:
                raise ValueError("threshold must be finite and in [-1, 1]")
        self.model = model
        self.threshold = float(threshold) if threshold is not None else None
        self._limit()

    @classmethod
    def from_pretrained(cls, model_path, *, device=None, threshold=None, local_files_only=True):
        """Load weights once; downloads are disabled by default."""
        path = Path(model_path).expanduser()
        if local_files_only and not path.is_dir():
            raise FileNotFoundError(f"Local embedding checkpoint not found: {path}")
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(str(path.resolve()) if path.is_dir() else str(model_path),
                                    device=device, local_files_only=local_files_only,
                                    trust_remote_code=False)
        return cls(model, threshold=threshold)

    def _limit(self):
        limit = getattr(self.model, "max_seq_length", None)
        if isinstance(limit, bool) or not isinstance(limit, Integral) or limit < 1:
            raise ValueError("Model must expose a positive max_seq_length")
        return int(limit)

    def classify(self, english, bengali):
        """Compare the complete English meaning with the Bengali line/couplet."""
        texts = {"english": _text(english, "english"), "bengali": _text(bengali, "bengali")}
        limit = self._limit()
        counts = {}
        for name, text in texts.items():
            counts[name] = len(self.model.tokenizer(text, add_special_tokens=True,
                                                   truncation=False)["input_ids"])
            if counts[name] > limit:
                raise ValueError(f"{name} has {counts[name]} tokens; exceeds the {limit}-token model limit")

        vectors = self.model.encode(list(texts.values()), normalize_embeddings=True,
                                   convert_to_numpy=True, show_progress_bar=False,
                                   batch_size=2, prompt="")
        similarity = _cosine(vectors)
        return {"cosine_similarity": similarity,
                "semantic_match": None if self.threshold is None else similarity >= self.threshold,
                "threshold": self.threshold, "token_counts": counts, "max_seq_length": limit}

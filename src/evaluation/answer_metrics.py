"""Non-authoritative reference-overlap diagnostics for open-ended QA."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any


def normalize_answer(text: str | None) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", str(text or "").casefold()).split())


def _ngrams(tokens: list[str], n: int) -> Counter[tuple[str, ...]]:
    return Counter(tuple(tokens[index:index + n]) for index in range(max(0, len(tokens) - n + 1)))


def sentence_bleu(reference: str, prediction: str) -> float:
    """BLEU-4, uniform weights, brevity penalty, method-1-style 0.1 smoothing."""
    ref, hyp = normalize_answer(reference).split(), normalize_answer(prediction).split()
    if not hyp:
        return 0.0
    precisions = []
    for n in range(1, 5):
        hc, rc = _ngrams(hyp, n), _ngrams(ref, n)
        denominator = sum(hc.values())
        if denominator == 0:
            precisions.append(0.1)
            continue
        matches = sum(min(count, rc[gram]) for gram, count in hc.items())
        precisions.append(matches / denominator if matches else 0.1 / denominator)
    bp = 1.0 if len(hyp) > len(ref) else math.exp(1.0 - len(ref) / max(len(hyp), 1))
    return bp * math.exp(sum(0.25 * math.log(max(value, 1e-15)) for value in precisions))


def single_reference_cider(reference: str, prediction: str, reference_corpus: list[str]) -> float:
    """Exploratory single-reference CIDEr-like TF-IDF cosine, n=1..4, scaled by 10.

    This is intentionally labelled non-canonical: it uses smoothed IDF over the
    actual single-reference corpus and never fabricates additional references.
    """
    ref_tokens, hyp_tokens = normalize_answer(reference).split(), normalize_answer(prediction).split()
    corpus_tokens = [normalize_answer(item).split() for item in reference_corpus]
    scores = []
    for n in range(1, 5):
        ref_counts, hyp_counts = _ngrams(ref_tokens, n), _ngrams(hyp_tokens, n)
        vocabulary = set(ref_counts) | set(hyp_counts)
        if not vocabulary:
            scores.append(0.0)
            continue
        documents = [_ngrams(tokens, n) for tokens in corpus_tokens]
        def vector(counts: Counter) -> dict[tuple[str, ...], float]:
            total = max(sum(counts.values()), 1)
            return {gram: count / total * (math.log((len(documents) + 1) / (sum(gram in doc for doc in documents) + 1)) + 1) for gram, count in counts.items()}
        rv, hv = vector(ref_counts), vector(hyp_counts)
        dot = sum(rv.get(gram, 0.0) * hv.get(gram, 0.0) for gram in vocabulary)
        denom = math.sqrt(sum(value * value for value in rv.values())) * math.sqrt(sum(value * value for value in hv.values()))
        scores.append(dot / denom if denom else 0.0)
    return 10.0 * sum(scores) / 4.0


def evaluate_answer(case: dict[str, Any], reference_corpus: list[str]) -> dict[str, Any]:
    gold, prediction = case.get("gold_dataset_answer"), case.get("validated_prediction")
    return {
        "case_id": case["case_id"], "question": case["question"], "gold_reference_answer": gold,
        "raw_prediction": case.get("raw_model_prediction"), "validated_prediction": prediction,
        "answer_status": case.get("answer_status"), "confidence": case.get("confidence"),
        "abstain": case.get("abstain"), "uncertainties": case.get("uncertainties", []),
        "reference_metrics": {
            "normalized_exact_match": normalize_answer(gold) == normalize_answer(prediction),
            "bleu": sentence_bleu(str(gold or ""), str(prediction or "")),
            "bleu_variant": "sentence BLEU-4; uniform weights; method-1-style 0.1 smoothing",
            "cider": single_reference_cider(str(gold or ""), str(prediction or ""), reference_corpus),
            "cider_variant": "exploratory single-reference CIDEr-like TF-IDF cosine; n=1..4; smoothed corpus IDF; scale=10; non-canonical",
        },
        "metric_interpretation": {"automatic_correctness": "not_determined", "metric_role": "diagnostic_reference_overlap"},
        "human_review": {"status": "unreviewed", "correctness": None, "notes": None},
    }


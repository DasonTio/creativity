from collections import Counter


def _tokens(text: str) -> list[str]:
    return [token.lower() for token in text.split() if token.strip()]


def distinct_n(outputs: list[str], n: int) -> float:
    ngrams: list[tuple[str, ...]] = []
    for output in outputs:
        tokens = _tokens(output)
        ngrams.extend(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))
    if not ngrams:
        return 0.0
    return len(set(ngrams)) / len(ngrams)


def _overlap_score(candidate: str, references: list[str]) -> float:
    candidate_tokens = _tokens(candidate)
    if not candidate_tokens or not references:
        return 0.0
    reference_counts = Counter(token for ref in references for token in _tokens(ref))
    overlap = sum(1 for token in candidate_tokens if reference_counts[token] > 0)
    return overlap / len(candidate_tokens)


def self_bleu(outputs: list[str]) -> float:
    if len(outputs) < 2:
        return 0.0
    scores = []
    for index, output in enumerate(outputs):
        refs = outputs[:index] + outputs[index + 1 :]
        scores.append(_overlap_score(output, refs))
    return sum(scores) / len(scores)

import numpy as np


def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    safe = np.where(norms == 0, 1.0, norms)
    normalized = embeddings / safe
    return normalized @ normalized.T


def average_pairwise_cosine(embeddings: np.ndarray) -> float:
    if len(embeddings) < 2:
        return 0.0
    matrix = cosine_similarity_matrix(embeddings)
    upper = matrix[np.triu_indices(len(embeddings), k=1)]
    return float(upper.mean()) if len(upper) else 0.0

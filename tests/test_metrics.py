import numpy as np

from mop_divpo.metrics.diversity import distinct_n, self_bleu
from mop_divpo.metrics.semantic import average_pairwise_cosine, cosine_similarity_matrix


def test_distinct_n_is_higher_for_varied_text():
    repeated = ["alpha alpha", "alpha alpha"]
    varied = ["alpha beta", "gamma delta"]
    assert distinct_n(varied, n=1) > distinct_n(repeated, n=1)


def test_distinct_n_handles_empty_outputs():
    assert distinct_n([], n=1) == 0.0
    assert distinct_n([""], n=2) == 0.0


def test_self_bleu_is_lower_for_diverse_outputs():
    repeated = ["the same idea", "the same idea", "the same idea"]
    varied = ["use a map", "borrow from biology", "remove the premise"]
    assert self_bleu(varied) < self_bleu(repeated)


def test_cosine_similarity_matrix_and_average():
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    matrix = cosine_similarity_matrix(embeddings)
    assert matrix.shape == (3, 3)
    assert matrix[0, 2] == 1.0
    assert round(average_pairwise_cosine(embeddings), 3) == 0.333

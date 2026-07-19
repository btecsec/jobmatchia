"""Tests de la self-attention NumPy — Chapitre 2.

On ne teste pas 'le modèle est intelligent' (non mesurable) mais les
PROPRIÉTÉS MATHÉMATIQUES que la théorie garantit. Réflexe à garder :
en IA aussi, un test vérifie un contrat précis.
"""

import numpy as np
import pytest

from jobmatch.nlp.attention import self_attention, softmax


@pytest.fixture
def toy_setup():
    rng = np.random.default_rng(seed=0)
    n_tokens, dim = 5, 8
    embeddings = rng.normal(size=(n_tokens, dim))
    weights = [rng.normal(size=(dim, dim)) for _ in range(3)]
    return embeddings, weights


def test_softmax_rows_are_probabilities():
    scores = np.array([[2.3, 0.1, -1.0], [0.0, 0.0, 0.0]])
    probs = softmax(scores)
    assert np.all(probs > 0)
    assert np.allclose(probs.sum(axis=-1), 1.0)


def test_softmax_is_numerically_stable():
    # Sans l'astuce du max, exp(1000) déborderait en inf/NaN.
    probs = softmax(np.array([[1000.0, 999.0]]))
    assert np.all(np.isfinite(probs))
    assert np.allclose(probs.sum(axis=-1), 1.0)


def test_attention_weights_sum_to_one(toy_setup):
    embeddings, (w_q, w_k, w_v) = toy_setup
    _, attn = self_attention(embeddings, w_q, w_k, w_v)
    assert np.allclose(attn.sum(axis=-1), 1.0)


def test_output_shapes(toy_setup):
    embeddings, (w_q, w_k, w_v) = toy_setup
    out, attn = self_attention(embeddings, w_q, w_k, w_v)
    n_tokens, dim = embeddings.shape
    assert out.shape == (n_tokens, dim)
    assert attn.shape == (n_tokens, n_tokens)


def test_attention_is_deterministic(toy_setup):
    # Le mécanisme est un calcul pur : mêmes entrées, mêmes sorties.
    # (La variabilité des LLM vient de l'échantillonnage de génération,
    # PAS de l'attention — nuance d'entretien classique.)
    embeddings, (w_q, w_k, w_v) = toy_setup
    out1, attn1 = self_attention(embeddings, w_q, w_k, w_v)
    out2, attn2 = self_attention(embeddings, w_q, w_k, w_v)
    assert np.array_equal(out1, out2)
    assert np.array_equal(attn1, attn2)


def test_context_changes_token_representation():
    # LE test qui prouve l'intérêt de l'attention : le MÊME token,
    # placé dans deux contextes différents, ressort avec deux vecteurs
    # différents. Un embedding statique en serait incapable.
    rng = np.random.default_rng(seed=1)
    dim = 4
    w_q, w_k, w_v = (rng.normal(size=(dim, dim)) for _ in range(3))

    python_token = np.array([0.9, 0.9, 0.0, 0.0])
    ctx_dev = np.stack([python_token, [0.8, 0.0, 0.2, 0.0]])      # + "backend"
    ctx_animal = np.stack([python_token, [0.0, 0.8, 0.1, 0.0]])   # + "royal"

    out_dev, _ = self_attention(ctx_dev, w_q, w_k, w_v)
    out_animal, _ = self_attention(ctx_animal, w_q, w_k, w_v)

    assert not np.allclose(out_dev[0], out_animal[0])

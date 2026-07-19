"""Self-attention en NumPy pur — Chapitre 2.

Implémentation minimale mais mathématiquement exacte du mécanisme
d'attention de l'article « Attention Is All You Need » (Vaswani et al.,
2017) : scores par produit scalaire Query·Key, softmax, moyenne
pondérée des Values. Aucune bibliothèque de Deep Learning : le but est
de prouver que le cœur d'un LLM tient en trois opérations de lycée.
"""

import numpy as np


def softmax(scores: np.ndarray) -> np.ndarray:
    """Transforme des scores en probabilités (positives, somme = 1 par ligne).

    L'astuce du max soustrait est standard : elle évite les débordements
    numériques sans changer le résultat (le softmax est invariant par
    translation des scores).
    """
    shifted = scores - scores.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def self_attention(
    embeddings: np.ndarray,
    w_query: np.ndarray,
    w_key: np.ndarray,
    w_value: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Une tête de self-attention.

    Args:
        embeddings: matrice (n_tokens, dim) — un vecteur par token.
        w_query, w_key, w_value: matrices apprises (dim, dim) qui
            fabriquent les Query, Key et Value de chaque token.
            (Dans un vrai Transformer, ces poids sont ajustés par
            l'entraînement ; ici on les fournit.)

    Returns:
        (nouveaux_vecteurs, poids_attention) :
        - nouveaux_vecteurs (n_tokens, dim) : chaque token enrichi du contexte
        - poids_attention (n_tokens, n_tokens) : qui a écouté qui, ligne = 1
    """
    queries = embeddings @ w_query   # "ce que je cherche"
    keys = embeddings @ w_key        # "ce que je propose"
    values = embeddings @ w_value    # "ce que je transmets"

    dim = queries.shape[-1]
    # Scores : produit scalaire de chaque Query avec chaque Key.
    # Division par sqrt(dim) : sans elle, les scores grossissent avec la
    # dimension et le softmax sature (détail de l'article de 2017).
    scores = queries @ keys.T / np.sqrt(dim)

    attention_weights = softmax(scores)

    # Chaque token devient la moyenne des Values, pondérée par son attention.
    new_vectors = attention_weights @ values
    return new_vectors, attention_weights


def demo() -> None:
    """Le token ambigu 'python' écoute son contexte et change de sens.

    Embeddings jouets en 4 dimensions, choisis à la main pour l'intuition :
      axe 0 : "programmation"   axe 1 : "animal"
      axe 2 : "action"          axe 3 : "lieu"
    """
    tokens = ["python", "royal", "vivarium"]
    embeddings = np.array([
        [0.9, 0.9, 0.0, 0.0],   # python : ambigu (langage ET serpent)
        [0.0, 0.8, 0.1, 0.0],   # royal : animalier
        [0.0, 0.7, 0.0, 0.9],   # vivarium : animalier + lieu
    ])
    rng = np.random.default_rng(seed=42)  # poids fixés pour la reproductibilité
    dim = embeddings.shape[1]
    w_q, w_k, w_v = (rng.normal(size=(dim, dim)) for _ in range(3))

    _, weights = self_attention(embeddings, w_q, w_k, w_v)

    print(f"{'':>10}" + "".join(f"{t:>10}" for t in tokens))
    for token, row in zip(tokens, weights):
        cells = "".join(f"{w:>10.2f}" for w in row)
        print(f"{token:>10}{cells}")
    print("\nChaque ligne somme à 1 : c'est la répartition d'attention du token.")


if __name__ == "__main__":
    demo()

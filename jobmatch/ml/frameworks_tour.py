"""Le même problème, deux outils : scikit-learn vs Keras — Chapitre 3.

On rejoue le détecteur de compétences du Chapitre 1 avec un réseau de
neurones Keras, sur les MÊMES données et les MÊMES vecteurs TF-IDF,
pour comparer honnêtement les deux approches.

Leçon visée : un framework de Deep Learning n'est ni magique ni
obligatoire ; la puissance ne se justifie que si le problème la réclame.

Règle d'or JobMatch AI : uniquement des textes synthétiques et anonymisés.
"""

import numpy as np
import keras
from keras import layers
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Continuité du dépôt : mêmes données qu'au Chapitre 1.
from jobmatch.skills.detector import TRAINING_LABELS, TRAINING_LINES

# Reproductibilité : graine fixée pour NumPy, Python et le backend.
# Le versioning de la Partie III transformera ce réflexe en discipline.
SEED = 42


def build_features(lines: list[str]) -> tuple[TfidfVectorizer, np.ndarray]:
    """Vectorise les lignes de CV en TF-IDF (mêmes réglages qu'au Ch.1)."""
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True)
    features = vectorizer.fit_transform(lines).toarray()
    return vectorizer, features


def build_sklearn_model(features: np.ndarray, labels: list[bool]) -> LogisticRegression:
    """La baseline du Chapitre 1 : une régression logistique."""
    model = LogisticRegression(max_iter=1000)
    model.fit(features, labels)
    return model


def build_keras_model(n_features: int) -> keras.Sequential:
    """Le réseau de la Figure 1.4, en huit lignes exécutables.

    Deux couches cachées (relu), une sortie sigmoïde qui produit une
    probabilité entre 0 et 1. 'adam' est une variante raffinée de la
    descente de gradient ; 'binary_crossentropy' est la fonction de
    perte standard de la classification binaire.
    """
    model = keras.Sequential([
        keras.Input(shape=(n_features,)),
        layers.Dense(16, activation="relu"),
        layers.Dense(8, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model


def train_both(lines: list[str], labels: list[bool]):
    """Entraîne les deux modèles sur les mêmes vecteurs TF-IDF."""
    keras.utils.set_random_seed(SEED)
    vectorizer, features = build_features(lines)
    y = np.array(labels, dtype="float32")

    sk_model = build_sklearn_model(features, labels)

    nn_model = build_keras_model(features.shape[1])
    # 300 epochs : à chaque passage, prédiction -> perte -> ajustement
    # des poids par descente de gradient (Chapitre 1, section 1.4).
    nn_model.fit(features, y, epochs=300, verbose=0)

    return vectorizer, sk_model, nn_model


def predict_both(vectorizer, sk_model, nn_model, line: str) -> tuple[float, float]:
    """Probabilité 'compétence' selon chaque modèle, pour une ligne."""
    vector = vectorizer.transform([line]).toarray()
    p_sklearn = float(sk_model.predict_proba(vector)[0][1])
    p_keras = float(nn_model.predict(vector, verbose=0)[0][0])
    return p_sklearn, p_keras


def demo() -> None:
    """Verdicts côte à côte sur les lignes pièges du Chapitre 1."""
    tricky_lines = [
        "Développement backend avec le framework Django",
        "Soigneur animalier : nourrissage du python royal",
        "Automatisation de tests et intégration continue",
    ]
    vectorizer, sk_model, nn_model = train_both(TRAINING_LINES, TRAINING_LABELS)

    print(f"{'LIGNE':<50} {'SKLEARN':>8} {'KERAS':>8}")
    for line in tricky_lines:
        p_sk, p_nn = predict_both(vectorizer, sk_model, nn_model, line)
        print(f"{line:<50} {p_sk:>8.3f} {p_nn:>8.3f}")
    print("\nMêmes conclusions, confiances différentes : plus de paramètres")
    print("= sorties plus tranchées, pas forcément un meilleur modèle.")


if __name__ == "__main__":
    demo()

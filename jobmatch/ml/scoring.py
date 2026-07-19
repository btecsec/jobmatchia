"""Modèle de scoring de compatibilité CV/offre — Chapitre 5.

Le cœur métier de JobMatch AI : étant donné un profil de candidat
(anonymisé, cf. Chapitre 4) et une offre d'emploi (publique), prédire
si la paire est compatible — c'est-à-dire si la candidature vaut la
peine d'être envoyée.

Leçon centrale du chapitre : sur des données déséquilibrées (la vraie
vie — la plupart des offres ne matchent PAS), l'accuracy est un
indicateur trompeur. Un modèle idiot qui répond toujours « non
compatible » affiche une accuracy superbe et une utilité nulle.
Les vraies métriques : précision, rappel, F1, matrice de confusion.

Règle d'or : ce module vit en zone IA. Il ne voit que des profils
anonymisés (compétences, années d'expérience) et des offres publiques.
Jamais de nom, d'âge, d'adresse ni de coordonnée.
"""

import random
from dataclasses import dataclass

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

SEED = 42

# Vivier de compétences synthétiques (aucune donnée réelle).
SKILLS_POOL = [
    "python", "sql", "docker", "linux", "git", "react", "kubernetes",
    "django", "fastapi", "postgresql", "aws", "terraform", "pandas",
    "scikit-learn", "javascript", "typescript", "ci/cd", "ansible",
    "monitoring", "api-rest",
]


@dataclass(frozen=True)
class Profile:
    """Profil candidat ANONYMISÉ : ce que la zone IA a le droit de voir."""

    skills: frozenset[str]
    years: int


@dataclass(frozen=True)
class Offer:
    """Offre d'emploi publique : compétences requises et séniorité minimale."""

    required: frozenset[str]
    min_years: int


def _ground_truth(profile: Profile, offer: Offer, rng: random.Random) -> bool:
    """Le « jugement recruteur » simulé qui étiquette nos paires.

    Règle métier plausible + 2 % de bruit : les recruteurs humains ne
    sont pas parfaitement cohérents, nos étiquettes non plus. Ce bruit
    borne la performance atteignable — aucun modèle honnête ne fera
    100 % sur ces données, et c'est réaliste.
    """
    coverage = len(profile.skills & offer.required) / len(offer.required)
    compatible = coverage >= 0.6 and profile.years >= offer.min_years
    if rng.random() < 0.02:
        compatible = not compatible
    return compatible


def pair_features(profile: Profile, offer: Offer) -> list[float]:
    """Traduit une paire (profil, offre) en features numériques (Ch.4).

    Chaque feature encode une intuition de recruteur, vérifiable :
    couverture des compétences requises, manques, marge de séniorité.
    """
    overlap = profile.skills & offer.required
    return [
        len(overlap) / len(offer.required),          # couverture [0..1]
        float(len(offer.required - profile.skills)),  # compétences manquantes
        float(profile.years - offer.min_years),       # marge d'expérience
        float(len(profile.skills)),                   # largeur du profil
    ]


def generate_dataset(
    n_profiles: int = 60, n_offers: int = 20, seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Génère toutes les paires profil × offre, étiquetées.

    Le déséquilibre n'est pas fabriqué : il ÉMERGE de la règle métier.
    Un candidat donné n'est compatible qu'avec une minorité d'offres —
    exactement comme dans la vraie vie.
    """
    rng = random.Random(seed)
    profiles = [
        Profile(
            skills=frozenset(rng.sample(SKILLS_POOL, k=rng.randint(4, 8))),
            years=rng.randint(1, 12),
        )
        for _ in range(n_profiles)
    ]
    offers = [
        Offer(
            required=frozenset(rng.sample(SKILLS_POOL, k=rng.randint(3, 5))),
            min_years=rng.randint(0, 8),
        )
        for _ in range(n_offers)
    ]
    features = [pair_features(p, o) for p in profiles for o in offers]
    labels = [_ground_truth(p, o, rng) for p in profiles for o in offers]
    return np.array(features), np.array(labels)


def train_scoring_model(
    x_train: np.ndarray, y_train: np.ndarray,
) -> LogisticRegression:
    """Régression logistique avec class_weight='balanced'.

    Sans ce paramètre, le modèle optimise l'accuracy globale et apprend
    surtout à dire « non » (la classe majoritaire). 'balanced' repondère
    les rares « oui » pour qu'ils pèsent autant que les « non ».
    """
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(x_train, y_train)
    return model


def evaluate(model, x_test: np.ndarray, y_test: np.ndarray) -> dict[str, float]:
    """Les quatre métriques du chapitre, sur le jeu de test uniquement."""
    y_pred = model.predict(x_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }


def demo() -> None:
    """Le piège de l'accuracy, chiffres en main, puis les vraies métriques."""
    x, y = generate_dataset()
    n_pos = int(y.sum())
    print(f"Paires générées : {len(y)} — compatibles : {n_pos} "
          f"({n_pos / len(y):.0%})")

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, stratify=y, random_state=SEED)

    # Le « modèle » naïf : répond toujours « non compatible ».
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(x_train, y_train)
    model = train_scoring_model(x_train, y_train)

    print(f"\n{'MÉTRIQUE':<12} {'TOUJOURS-NON':>14} {'SCORING':>10}")
    m_dummy, m_model = evaluate(dummy, x_test, y_test), evaluate(
        model, x_test, y_test)
    for key in ("accuracy", "precision", "recall", "f1"):
        print(f"{key:<12} {m_dummy[key]:>14.3f} {m_model[key]:>10.3f}")

    tn, fp, fn, tp = confusion_matrix(
        y_test, model.predict(x_test)).ravel()
    print(f"\nMatrice de confusion du modèle de scoring (test) :")
    print(f"  Vrais négatifs  : {tn:>3}   Faux positifs  : {fp:>3}")
    print(f"  Faux négatifs   : {fn:>3}   Vrais positifs : {tp:>3}")

    # Le curseur précision/rappel : même modèle, trois seuils.
    proba = model.predict_proba(x_test)[:, 1]
    print(f"\n{'SEUIL':<8} {'PRÉCISION':>10} {'RAPPEL':>8}")
    for seuil in (0.3, 0.5, 0.7):
        y_seuil = proba >= seuil
        p = precision_score(y_test, y_seuil, zero_division=0)
        r = recall_score(y_test, y_seuil, zero_division=0)
        print(f"{seuil:<8} {p:>10.3f} {r:>8.3f}")


if __name__ == "__main__":
    demo()

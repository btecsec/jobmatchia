"""Audit de biais du scoring — Chapitre 6 : « L'affaire du CV fantôme ».

Reproduction, sur données 100 % synthétiques, du mécanisme du scandale
Amazon 2018 : un modèle entraîné sur un historique de recrutement biaisé
apprend à pénaliser un mot proxy (« féminine ») qui n'a aucun rapport
avec la compétence. Puis correction : neutralisation déterministe du
proxy avant vectorisation, et mesures avant/après.

Trois instruments d'audit :
- les coefficients du modèle (l'arme du crime, mot par mot) ;
- le test du CV fantôme : deux CV identiques au mot près, deux scores ;
- le disparate impact ratio (règle des 80 % de l'EEOC).

Règle d'or : ce module ne voit que des textes de CV anonymisés. Le tag
de groupe (A/B) n'existe que dans la zone d'audit hors-ligne, pour
MESURER le biais — il n'est jamais une feature du modèle, et n'atteint
jamais la production.
"""

import random
import re
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

SEED = 42

TECH_SKILLS = [
    "python", "sql", "docker", "linux", "git", "kubernetes", "aws",
    "fastapi", "terraform", "pandas", "react", "ansible",
]

# Le mot proxy : aucun rapport avec la compétence, corrélé au groupe.
GENDER_MARKER_RE = re.compile(r"\bféminine?s?\b", re.IGNORECASE)

# Paires de loisirs appariées : seul le mot « féminine » diffère.
HOBBIES_NEUTRAL = [
    "capitaine de l'équipe d'échecs",
    "bénévole dans une association",
    "joue dans une ligue de volley",
]
HOBBIES_MARKED = [
    "capitaine de l'équipe féminine d'échecs",
    "bénévole dans une association féminine",
    "joue dans une ligue féminine de volley",
]


@dataclass(frozen=True)
class CV:
    """Un CV synthétique anonymisé, plus les métadonnées d'audit.

    `group` et `competent` sont la vérité terrain du simulateur : ils
    servent à MESURER le biais, jamais à entraîner le modèle.
    """

    text: str
    group: str        # "A" ou "B" — zone d'audit uniquement
    competent: bool   # vérité terrain simulée


def generate_corpus(n: int = 400, seed: int = SEED) -> list[CV]:
    """Corpus synthétique : compétence tirée des skills, loisir selon le groupe.

    Les loisirs des deux groupes sont appariés mot pour mot : la SEULE
    différence textuelle systématique entre A et B est le mot proxy.
    """
    rng = random.Random(seed)
    corpus = []
    for i in range(n):
        skills = rng.sample(TECH_SKILLS, k=rng.randint(2, 7))
        competent = len(skills) >= 5
        group = "A" if i % 2 == 0 else "B"
        idx = rng.randrange(len(HOBBIES_NEUTRAL))
        hobby = (HOBBIES_NEUTRAL if group == "A" else HOBBIES_MARKED)[idx]
        text = f"Compétences : {' '.join(skills)}. Loisirs : {hobby}."
        corpus.append(CV(text=text, group=group, competent=competent))
    return corpus


def biased_historical_labels(corpus: list[CV], seed: int = SEED) -> np.ndarray:
    """Le « recruteur de 2014 » simulé : compétent → embauché… sauf que.

    Si le CV contient le marqueur, 70 % des candidatures compétentes
    sont écartées quand même. C'est le biais HUMAIN historique — le
    modèle ne fera que l'apprendre, fidèlement. Plus 2 % de bruit.
    """
    rng = random.Random(seed + 1)
    labels = []
    for cv in corpus:
        hired = cv.competent
        if hired and GENDER_MARKER_RE.search(cv.text):
            hired = rng.random() < 0.3
        if rng.random() < 0.02:
            hired = not hired
        labels.append(hired)
    return np.array(labels)


def neutralize(text: str) -> str:
    """Neutralisation déterministe (non-IA) du proxy avant vectorisation.

    Même philosophie que l'anonymiseur du Chapitre 4 : un module
    déterministe retire le canal par lequel le biais s'exprime, AVANT
    que le modèle ne voie le texte — à l'entraînement ET en inférence.
    """
    return re.sub(r"\s+", " ", GENDER_MARKER_RE.sub(" ", text)).strip()


def train_model(texts: list[str], labels: np.ndarray,
                neutralized: bool = False) -> Pipeline:
    """Sac de mots + régression logistique, avec ou sans neutralisation."""
    vectorizer = CountVectorizer(
        lowercase=True,
        preprocessor=(lambda t: neutralize(t.lower())) if neutralized else None,
    )
    model = Pipeline([
        ("vectorizer", vectorizer),
        ("classifier", LogisticRegression(max_iter=1000,
                                          class_weight="balanced")),
    ])
    model.fit(texts, labels)
    return model


def word_coefficients(model: Pipeline) -> dict[str, float]:
    """Le poids appris pour chaque mot : l'instrument d'audit n° 1."""
    vocab = model.named_steps["vectorizer"].get_feature_names_out()
    coefs = model.named_steps["classifier"].coef_[0]
    return dict(zip(vocab, coefs))


def ghost_cv_gap(model: Pipeline, base_text: str, marked_text: str) -> float:
    """Test du CV fantôme : score(CV neutre) - score(CV jumeau marqué).

    Les deux textes sont identiques au mot proxy près. Tout écart de
    score est, par construction, imputable à ce seul mot.
    """
    p_neutral, p_marked = model.predict_proba([base_text, marked_text])[:, 1]
    return float(p_neutral - p_marked)


def selection_rates(model: Pipeline, corpus: list[CV]) -> dict[str, float]:
    """Taux de sélection prédit par groupe : l'instrument d'audit n° 2."""
    rates = {}
    for group in ("A", "B"):
        texts = [cv.text for cv in corpus if cv.group == group]
        rates[group] = float(model.predict(texts).mean())
    return rates


def disparate_impact(model: Pipeline, corpus: list[CV]) -> float:
    """Ratio des taux de sélection (groupe défavorisé / favorisé).

    Règle des 80 % (EEOC, 1978) : sous 0.8, présomption d'impact
    discriminatoire. Ce seuil est un repère juridique américain, pas
    une absolution — 0.81 n'est pas « équitable », c'est « moins pire ».
    """
    rates = selection_rates(model, corpus)
    return rates["B"] / rates["A"] if rates["A"] > 0 else 0.0


def utility_vs_truth(model: Pipeline, corpus: list[CV]) -> float:
    """F1 contre la VRAIE compétence — pas contre les étiquettes biaisées.

    La question qui compte : le modèle retrouve-t-il les candidats
    réellement compétents, ou reproduit-il les préférences du passé ?
    """
    texts = [cv.text for cv in corpus]
    truth = np.array([cv.competent for cv in corpus])
    return float(f1_score(truth, model.predict(texts)))


def demo() -> None:
    """Le scandale rejoué, l'audit, la correction, les chiffres avant/après."""
    corpus = generate_corpus()
    labels = biased_historical_labels(corpus)
    idx_train, idx_test = train_test_split(
        np.arange(len(corpus)), test_size=0.25, random_state=SEED,
        stratify=labels)
    train_texts = [corpus[i].text for i in idx_train]
    test_corpus = [corpus[i] for i in idx_test]

    n_comp = sum(cv.competent for cv in corpus)
    print(f"Corpus : {len(corpus)} CV synthétiques — {n_comp} compétents, "
          f"groupes A/B à parts égales")
    hired_a = labels[[i for i, cv in enumerate(corpus)
                      if cv.group == 'A' and cv.competent]].mean()
    hired_b = labels[[i for i, cv in enumerate(corpus)
                      if cv.group == 'B' and cv.competent]].mean()
    print(f"Historique biaisé — taux d'embauche des COMPÉTENTS : "
          f"A {hired_a:.0%}, B {hired_b:.0%}")

    naive = train_model(train_texts, labels[idx_train])
    fair = train_model(train_texts, labels[idx_train], neutralized=True)

    coefs = word_coefficients(naive)
    worst = sorted(coefs.items(), key=lambda kv: kv[1])[:3]
    print("\n--- Modèle naïf : les 3 mots les plus pénalisés ---")
    for word, coef in worst:
        print(f"  {word:<12} {coef:+.2f}")

    base = ("Compétences : python sql docker kubernetes git aws. "
            "Loisirs : capitaine de l'équipe d'échecs.")
    marked = base.replace("l'équipe", "l'équipe féminine")
    print(f"\n{'':<28} {'NAÏF':>8} {'CORRIGÉ':>9}")
    print(f"{'Écart CV fantôme':<28} {ghost_cv_gap(naive, base, marked):>8.3f} "
          f"{ghost_cv_gap(fair, base, marked):>9.3f}")
    di_n, di_f = disparate_impact(naive, test_corpus), disparate_impact(
        fair, test_corpus)
    print(f"{'Disparate impact (seuil .8)':<28} {di_n:>8.3f} {di_f:>9.3f}")
    print(f"{'F1 vs vraie compétence':<28} "
          f"{utility_vs_truth(naive, test_corpus):>8.3f} "
          f"{utility_vs_truth(fair, test_corpus):>9.3f}")


if __name__ == "__main__":
    demo()

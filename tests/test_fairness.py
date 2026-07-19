"""Tests de l'audit de biais — Chapitre 6."""

import numpy as np
import pytest
from sklearn.model_selection import train_test_split

from jobmatch.fairness.audit import (
    GENDER_MARKER_RE,
    SEED,
    biased_historical_labels,
    disparate_impact,
    generate_corpus,
    ghost_cv_gap,
    neutralize,
    train_model,
    utility_vs_truth,
    word_coefficients,
)

BASE = ("Compétences : python sql docker kubernetes git aws. "
        "Loisirs : capitaine de l'équipe d'échecs.")
MARKED = BASE.replace("l'équipe", "l'équipe féminine")


@pytest.fixture(scope="module")
def corpus():
    return generate_corpus()


@pytest.fixture(scope="module")
def labels(corpus):
    return biased_historical_labels(corpus)


@pytest.fixture(scope="module")
def splits(corpus, labels):
    idx_train, idx_test = train_test_split(
        np.arange(len(corpus)), test_size=0.25, random_state=SEED,
        stratify=labels)
    train_texts = [corpus[i].text for i in idx_train]
    test_corpus = [corpus[i] for i in idx_test]
    return train_texts, labels[idx_train], test_corpus


@pytest.fixture(scope="module")
def naive_model(splits):
    train_texts, y_train, _ = splits
    return train_model(train_texts, y_train)


@pytest.fixture(scope="module")
def fair_model(splits):
    train_texts, y_train, _ = splits
    return train_model(train_texts, y_train, neutralized=True)


def test_corpus_reproductible(corpus):
    again = generate_corpus()
    assert [cv.text for cv in corpus] == [cv.text for cv in again]


def test_marqueur_present_uniquement_dans_le_groupe_b(corpus):
    """Par construction, seul le mot proxy distingue les deux groupes."""
    for cv in corpus:
        found = bool(GENDER_MARKER_RE.search(cv.text))
        assert found == (cv.group == "B")


def test_historique_biaise_contre_les_competents_du_groupe_b(corpus, labels):
    hired_a = labels[[i for i, cv in enumerate(corpus)
                      if cv.group == "A" and cv.competent]].mean()
    hired_b = labels[[i for i, cv in enumerate(corpus)
                      if cv.group == "B" and cv.competent]].mean()
    assert hired_a > 0.9
    assert hired_b < 0.5


def test_le_modele_naif_apprend_le_mot_proxy(naive_model):
    """L'arme du crime : un gros coefficient négatif sur « féminine »."""
    coefs = word_coefficients(naive_model)
    assert coefs["féminine"] < -1.0
    # Et c'est bien LE mot le plus pénalisé du vocabulaire.
    assert coefs["féminine"] == min(coefs.values())


def test_cv_fantome_demasque_le_modele_naif(naive_model):
    assert ghost_cv_gap(naive_model, BASE, MARKED) > 0.2


def test_disparate_impact_du_modele_naif_sous_le_seuil(naive_model, splits):
    _, _, test_corpus = splits
    assert disparate_impact(naive_model, test_corpus) < 0.8


def test_neutralisation_retire_le_proxy_et_garde_le_reste():
    cleaned = neutralize(MARKED.lower())
    assert not GENDER_MARKER_RE.search(cleaned)
    assert "python" in cleaned and "échecs" in cleaned
    # Déterministe et idempotente, comme l'anonymiseur du Chapitre 4.
    assert neutralize(cleaned) == cleaned


def test_modele_corrige_ecart_fantome_nul(fair_model):
    """Après neutralisation, les CV jumeaux sont le MÊME vecteur : écart 0."""
    assert ghost_cv_gap(fair_model, BASE, MARKED) == 0.0


def test_modele_corrige_atteint_la_parite(fair_model, splits):
    _, _, test_corpus = splits
    assert disparate_impact(fair_model, test_corpus) > 0.9


def test_la_correction_ne_coute_rien_en_utilite_reelle(
        naive_model, fair_model, splits):
    """Mieux : elle rapporte. Les étiquettes biaisées mentaient sur le réel."""
    _, _, test_corpus = splits
    assert utility_vs_truth(fair_model, test_corpus) >= utility_vs_truth(
        naive_model, test_corpus)

"""Tests du tour des frameworks — Chapitre 3.

Même rigueur pour les deux mondes : on vérifie les CONTRATS
(sorties bornées, précision d'entraînement, reproductibilité),
pas une 'intelligence' non mesurable.
"""

import numpy as np
import pytest

from jobmatch.skills.detector import TRAINING_LABELS, TRAINING_LINES
from jobmatch.ml.frameworks_tour import predict_both, train_both


@pytest.fixture(scope="module")
def trained():
    # scope="module" : l'entraînement Keras (300 epochs) n'est fait
    # qu'une fois pour toute la suite de tests.
    return train_both(TRAINING_LINES, TRAINING_LABELS)


def test_both_outputs_are_probabilities(trained):
    vectorizer, sk_model, nn_model = trained
    p_sk, p_nn = predict_both(
        vectorizer, sk_model, nn_model, "Administration Linux et scripting"
    )
    assert 0.0 <= p_sk <= 1.0
    assert 0.0 <= p_nn <= 1.0


def test_both_models_fit_training_data(trained):
    # Sur 20 exemples, les deux modèles doivent au moins apprendre
    # leur jeu d'entraînement (>= 90 %). Mesurer la généralisation
    # exigerait un jeu de test séparé : c'est l'objet du Chapitre 4.
    vectorizer, sk_model, nn_model = trained
    correct_sk = correct_nn = 0
    for line, label in zip(TRAINING_LINES, TRAINING_LABELS):
        p_sk, p_nn = predict_both(vectorizer, sk_model, nn_model, line)
        correct_sk += (p_sk >= 0.5) == label
        correct_nn += (p_nn >= 0.5) == label
    assert correct_sk / len(TRAINING_LINES) >= 0.9
    assert correct_nn / len(TRAINING_LINES) >= 0.9


def test_models_agree_on_clear_cases(trained):
    vectorizer, sk_model, nn_model = trained
    p_sk, p_nn = predict_both(
        vectorizer, sk_model, nn_model,
        "Développement backend en Python et Django",
    )
    assert p_sk >= 0.5 and p_nn >= 0.5  # compétence évidente

    p_sk, p_nn = predict_both(
        vectorizer, sk_model, nn_model,
        "Centres d'intérêt : cuisine, jardinage, lecture",
    )
    assert p_sk < 0.5 and p_nn < 0.5  # non-compétence évidente


def test_keras_training_is_reproducible():
    # Graine fixée => deux entraînements identiques produisent les
    # mêmes prédictions. Fondation du versioning de la Partie III.
    v1, sk1, nn1 = train_both(TRAINING_LINES, TRAINING_LABELS)
    v2, sk2, nn2 = train_both(TRAINING_LINES, TRAINING_LABELS)
    line = "Développement backend avec le framework Django"
    _, p1 = predict_both(v1, sk1, nn1, line)
    _, p2 = predict_both(v2, sk2, nn2, line)
    assert np.isclose(p1, p2, atol=1e-6)

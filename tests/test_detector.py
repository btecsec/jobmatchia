"""Tests du détecteur de compétences — Chapitre 1.

Dès le premier chapitre, le code du SaaS est testé : c'est un réflexe
de production, pas une option de fin de projet.
"""

import pytest

from jobmatch.skills.detector import (
    MLSkillDetector,
    RuleBasedSkillDetector,
    TRAINING_LABELS,
    TRAINING_LINES,
)


def build_trained_ml() -> MLSkillDetector:
    detector = MLSkillDetector()
    detector.train(TRAINING_LINES, TRAINING_LABELS)
    return detector


def test_rules_detect_explicit_keyword():
    pred = RuleBasedSkillDetector().predict("Développement Python et Docker")
    assert pred.is_skill is True


def test_rules_fail_on_implicit_skill():
    # Limite structurelle des règles : Django implique Python, mais aucune
    # règle ne le sait. Le test DOCUMENTE cette faiblesse.
    pred = RuleBasedSkillDetector().predict(
        "Développement backend avec le framework Django"
    )
    assert pred.is_skill is False


def test_rules_false_positive_on_python_snake():
    # 'python' le serpent déclenche la règle 'python' le langage.
    pred = RuleBasedSkillDetector().predict(
        "Soigneur animalier : entretien du vivarium du python royal"
    )
    assert pred.is_skill is True  # faux positif assumé et documenté


def test_ml_generalizes_to_implicit_skill():
    pred = build_trained_ml().predict(
        "Développement backend avec le framework Django"
    )
    assert pred.is_skill is True


def test_ml_refuses_prediction_before_training():
    with pytest.raises(RuntimeError):
        MLSkillDetector().predict("Python")


def test_ml_rejects_mismatched_labels():
    with pytest.raises(ValueError):
        MLSkillDetector().train(["une ligne"], [True, False])


def test_confidence_is_a_probability():
    pred = build_trained_ml().predict("Administration Linux et scripting")
    assert 0.0 <= pred.confidence <= 1.0

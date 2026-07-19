"""Tests des gates CI et du protocole LLM-as-a-Judge — Chapitre 8."""

import numpy as np
import pytest

from jobmatch.mlops.gates import data_gate, model_gate, run_gates, safety_gate
from jobmatch.mlops.llm_judge import (
    build_judge_prompt,
    heuristic_judge,
    judge_pair,
    parse_verdict,
    position_biased_judge,
)
from jobmatch.privacy.anonymizer import Identity

ALEX = Identity(prenom="Alex", nom="Martin",
                email="alex.martin@exemple.fr", telephone="06 12 34 56 78")

OFFER = ["docker", "kubernetes", "python"]
GOOD_LETTER = ("Votre offre a retenu toute mon attention : cinq années à "
               "automatiser des déploiements Docker et Kubernetes, et un "
               "quotidien outillé en Python, correspondent précisément aux "
               "compétences que vous recherchez. Je serais ravi d'échanger "
               "sur vos enjeux d'infrastructure et d'apporter à votre "
               "équipe des pipelines fiables, mesurés et documentés.")
WEAK_LETTER = ("Je suis très motivé et je m'adapte vite. Écrivez-moi sur "
               "alex.martin@exemple.fr pour en discuter.")


# ---- gates -------------------------------------------------------------

def test_data_gate_accepte_un_dataset_sain():
    x = np.ones((300, 4))
    y = np.array([1] * 30 + [0] * 270)
    assert data_gate(x, y).passed


def test_data_gate_refuse_les_nan_et_les_datasets_maigres():
    x = np.ones((300, 4))
    x[0, 0] = np.nan
    y = np.array([1] * 30 + [0] * 270)
    assert not data_gate(x, y).passed
    assert not data_gate(np.ones((10, 4)), np.ones(10)).passed


def test_data_gate_refuse_le_desequilibre_absurde():
    x = np.ones((300, 4))
    assert not data_gate(x, np.zeros(300)).passed      # 0 % de positifs
    assert not data_gate(x, np.ones(300)).passed       # 100 % de positifs


def test_model_gate_bloque_la_regression_du_rappel():
    """L'incident du Chapitre 7, cette fois arrêté par la CI."""
    champion = {"recall": 0.963, "f1": 0.612}
    challenger = {"recall": 0.741, "f1": 0.833}   # meilleur F1, pire rappel
    assert not model_gate(challenger, champion).passed


def test_model_gate_tolere_le_bruit_et_accepte_le_progres():
    champion = {"recall": 0.963}
    assert model_gate({"recall": 0.950}, champion, tolerance=0.02).passed
    assert model_gate({"recall": 0.980}, champion).passed


def test_safety_gate_attrape_la_fuite_et_la_nomme():
    result = safety_gate([GOOD_LETTER, WEAK_LETTER], ALEX)
    assert not result.passed
    assert "alex.martin@exemple.fr" in result.reason


def test_safety_gate_laisse_passer_les_sorties_propres():
    assert safety_gate([GOOD_LETTER], ALEX).passed


def test_run_gates_exige_l_unanimite():
    x = np.ones((300, 4))
    y = np.array([1] * 30 + [0] * 270)
    ok = data_gate(x, y)
    ko = safety_gate([WEAK_LETTER], ALEX)
    assert run_gates([ok]) is True
    assert run_gates([ok, ko]) is False


# ---- juge --------------------------------------------------------------

def test_parse_verdict_strict():
    assert parse_verdict('{"winner": "B", "reason": "x"}') == "B"
    with pytest.raises(ValueError):
        parse_verdict("Bien sûr ! Voici mon verdict : A gagne.")
    with pytest.raises(ValueError):
        parse_verdict('{"winner": "C", "reason": "invente"}')


def test_le_juge_heuristique_applique_la_grille():
    raw = heuristic_judge(build_judge_prompt(OFFER, GOOD_LETTER, WEAK_LETTER))
    assert parse_verdict(raw) == "A"


def test_double_passage_stable_pour_un_juge_honnete():
    assert judge_pair(heuristic_judge, OFFER, GOOD_LETTER, WEAK_LETTER) == "A"
    # L'ordre de présentation ne change pas le gagnant.
    assert judge_pair(heuristic_judge, OFFER, WEAK_LETTER, GOOD_LETTER) == "B"


def test_double_passage_neutralise_le_biais_de_position():
    """Un juge qui dit toujours « A » ne doit jamais désigner de gagnant."""
    verdict = judge_pair(position_biased_judge, OFFER,
                         GOOD_LETTER, WEAK_LETTER)
    assert verdict == "tie"

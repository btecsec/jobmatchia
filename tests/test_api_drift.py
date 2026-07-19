"""Tests de l'API de scoring et de la détection de dérive — Chapitre 9."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from jobmatch.api.app import THRESHOLD, app
from jobmatch.mlops.drift import (
    drift_report,
    drift_status,
    population_stability_index,
)

client = TestClient(app)

GOOD_REQUEST = {
    "skills": ["python", "sql", "docker", "kubernetes", "git", "aws"],
    "years": 8,
    "offer_required": ["python", "docker", "kubernetes"],
    "offer_min_years": 3,
}


# ---- API ---------------------------------------------------------------

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_score_profil_fort():
    response = client.post("/score", json=GOOD_REQUEST)
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["probability"] <= 1.0
    assert body["compatible"] is (body["probability"] >= THRESHOLD)
    assert body["compatible"] is True


def test_score_profil_faible():
    weak = dict(GOOD_REQUEST, skills=["react"], years=0)
    body = client.post("/score", json=weak).json()
    assert body["compatible"] is False


def test_champ_manquant_refuse():
    incomplete = {key: value for key, value in GOOD_REQUEST.items()
                  if key != "skills"}
    assert client.post("/score", json=incomplete).status_code == 422


def test_la_regle_d_or_au_niveau_du_schema():
    """Un champ PII inattendu n'est pas ignoré : il est REFUSÉ (422)."""
    smuggling = dict(GOOD_REQUEST, name="Alex Martin",
                     email="alex.martin@exemple.fr")
    assert client.post("/score", json=smuggling).status_code == 422


def test_bornes_de_validation():
    assert client.post("/score",
                       json=dict(GOOD_REQUEST, years=-1)).status_code == 422
    assert client.post("/score",
                       json=dict(GOOD_REQUEST, years=200)).status_code == 422


# ---- dérive ------------------------------------------------------------

def test_psi_quasi_nul_sans_derive():
    rng = np.random.default_rng(0)
    sample = rng.normal(0, 1, 5000)
    other = rng.normal(0, 1, 5000)
    assert population_stability_index(sample, other) < 0.05


def test_psi_detecte_un_decalage_de_moyenne():
    rng = np.random.default_rng(0)
    reference = rng.normal(0, 1, 5000)
    shifted = rng.normal(1.0, 1, 5000)
    assert population_stability_index(reference, shifted) > 0.2


def test_psi_supporte_les_valeurs_hors_plage():
    """La production peut déborder la plage vue à l'entraînement."""
    reference = np.linspace(0, 1, 1000)
    production = np.linspace(5, 6, 1000)   # totalement hors plage
    psi = population_stability_index(reference, production)
    assert np.isfinite(psi) and psi > 0.2


def test_statuts_aux_seuils():
    assert drift_status(0.05) == "stable"
    assert drift_status(0.15) == "A SURVEILLER"
    assert drift_status(0.25) == "ALERTE"


def test_rapport_par_feature():
    rng = np.random.default_rng(1)
    x_ref = rng.normal(0, 1, (2000, 2))
    x_prod = x_ref.copy()
    x_prod[:, 1] += 2.0    # seule la 2e feature dérive
    report = drift_report(x_ref, x_prod, ["stable_f", "derive_f"])
    statuses = {name: status for name, _, status in report}
    assert statuses["stable_f"] == "stable"
    assert statuses["derive_f"] == "ALERTE"

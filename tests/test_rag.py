"""Tests de l'Agent Chasseur d'offres (RAG) — Chapitre 10."""

import numpy as np
import pytest

from jobmatch.rag.hunter import (
    GOLD_QUERIES,
    OFFERS,
    OfferHunter,
    chunk_text,
    recall_at_k,
)


@pytest.fixture(scope="module")
def hunter():
    return OfferHunter(OFFERS)


# ---- chunking ----------------------------------------------------------

def test_chunk_court_reste_entier():
    assert chunk_text("un texte bref", max_words=40) == ["un texte bref"]


def test_chunks_couvrent_tout_avec_chevauchement():
    words = [f"mot{i}" for i in range(100)]
    chunks = chunk_text(" ".join(words), max_words=40, overlap=10)
    assert all(len(c.split()) <= 40 for c in chunks)
    # Aucun mot perdu…
    assert set(words) == {w for c in chunks for w in c.split()}
    # …et les chunks consécutifs partagent bien des mots (le chevauchement).
    assert set(chunks[0].split()) & set(chunks[1].split())


# ---- index et recherche ------------------------------------------------

def test_les_vecteurs_sont_normalises(hunter):
    vectors = hunter.embed(["docker kubernetes", "python sql"])
    norms = np.linalg.norm(vectors, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_requete_evidente_top1(hunter):
    results = hunter.retrieve("docker kubernetes terraform ci/cd aws", k=3)
    assert results[0][0].offer_id == "O1"
    # Scores décroissants : FAISS renvoie du plus proche au plus lointain.
    scores = [score for _, score in results]
    assert scores == sorted(scores, reverse=True)


def test_k_borne_le_nombre_de_resultats(hunter):
    assert len(hunter.retrieve("python", k=5)) == 5
    assert len(hunter.retrieve("python", k=1)) == 1


def test_recherche_deterministe(hunter):
    query = "fastapi postgresql pytest"
    first = [offer.offer_id for offer, _ in hunter.retrieve(query, 3)]
    second = [offer.offer_id for offer, _ in hunter.retrieve(query, 3)]
    assert first == second


# ---- prompt ------------------------------------------------------------

def test_le_prompt_contient_sources_citations_et_consignes(hunter):
    prompt = hunter.build_prompt("docker kubernetes", k=2)
    assert "UNIQUEMENT" in prompt          # anti-hallucination
    assert "[1]" in prompt and "[2]" in prompt
    assert "dis-le explicitement" in prompt
    # La requête vient du profil anonymisé : aucune PII à chercher ici,
    # mais le prompt ne doit contenir que des offres et la requête.
    assert "PROFIL RECHERCHÉ (anonymisé)" in prompt


# ---- évaluation du retriever ------------------------------------------

def test_recall_at_3_sur_le_jeu_etalonne(hunter):
    """Le contrat de qualité du R de RAG, vérifiable en CI sans LLM."""
    assert recall_at_k(hunter, GOLD_QUERIES, k=3) >= 0.85


def test_le_top_k_rattrape_une_partie_des_erreurs_du_top_1(hunter):
    r1 = recall_at_k(hunter, GOLD_QUERIES, k=1)
    r3 = recall_at_k(hunter, GOLD_QUERIES, k=3)
    assert r3 > r1


def test_le_trou_lexical_reste_un_echec(hunter):
    """« conteneurisées » n'apparaît dans aucune offre : un index lexical
    ne peut pas retrouver O1. C'est LA limite qui justifie les
    embeddings denses en production."""
    results = hunter.retrieve(
        "déploiement continu d'applications conteneurisées", k=3)
    assert "O1" not in [offer.offer_id for offer, _ in results]

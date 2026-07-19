"""Tests de la barrière PII — Chapitre 4.

Ces tests SONT la règle d'or : si l'un d'eux casse, une PII peut
atteindre la zone IA et le déploiement doit être bloqué (la CI de la
Partie III automatisera exactement cela).
"""

import pytest

from jobmatch.privacy.anonymizer import (
    AnonymizedDocument,
    DeterministicAnonymizer,
    Identity,
    PiiVault,
    find_pii_leaks,
)

ALEX = Identity(
    prenom="Alex",
    nom="Martin",
    email="alex.martin@exemple.fr",
    telephone="06 12 34 56 78",
    adresse="12 rue des Lilas, 75011 Paris",
    date_naissance="14/03/1994",
)

CV_BRUT = (
    "Alex Martin — 32 ans\n"
    "12 rue des Lilas, 75011 Paris — alex.martin@exemple.fr — 06 12 34 56 78\n"
    "linkedin.com/in/alex-martin-dev — perso : amartin.pro@exemple.fr\n"
    "Reconversion développeur backend, 10 ans d'expérience en logistique.\n"
    "Compétences : Python, SQL, Docker, API REST.\n"
)


@pytest.fixture()
def anonymizer():
    return DeterministicAnonymizer(PiiVault())


def test_aucune_pii_ne_survit(anonymizer):
    """Le contrat central : zéro fuite dans le texte anonymisé."""
    doc = anonymizer.anonymize(CV_BRUT, ALEX)
    assert find_pii_leaks(doc.text, ALEX) == []


def test_aller_retour_sans_perte(anonymizer):
    """enrich(anonymize(x)) == x : rien n'est détruit, tout est déplacé."""
    doc = anonymizer.anonymize(CV_BRUT, ALEX)
    assert anonymizer.enrich(doc) == CV_BRUT


def test_le_contenu_utile_est_preserve(anonymizer):
    """L'anonymiseur retire l'identité, pas les compétences."""
    doc = anonymizer.anonymize(CV_BRUT, ALEX)
    for mot in ("Python", "SQL", "Docker", "API REST", "logistique"):
        assert mot in doc.text


def test_l_experience_n_est_pas_confondue_avec_l_age(anonymizer):
    """« 32 ans » (PII) part, « 10 ans d'expérience » (signal) reste."""
    doc = anonymizer.anonymize(CV_BRUT, ALEX)
    assert "32 ans" not in doc.text
    assert "10 ans d'expérience" in doc.text


def test_le_filet_regex_attrape_les_pii_non_declarees(anonymizer):
    """Le second email n'est PAS dans l'Identity : la couche regex l'attrape."""
    doc = anonymizer.anonymize(CV_BRUT, ALEX)
    assert "amartin.pro@exemple.fr" not in doc.text


def test_variantes_de_telephone(anonymizer):
    """Le numéro reformaté par l'utilisateur est quand même retiré."""
    for variante in ("0612345678", "06.12.34.56.78", "+33 6 12 34 56 78"):
        doc = anonymizer.anonymize(f"Joignable au {variante}.", ALEX)
        assert variante not in doc.text
        assert anonymizer.enrich(doc) == f"Joignable au {variante}."


def test_prenom_court_ne_mange_pas_les_mots(anonymizer):
    """Frontières de mots : « Ali » ne doit pas être retiré de « qualité »."""
    ali = Identity(prenom="Ali", nom="Ben Salem",
                   email="ali@exemple.fr", telephone="06 98 76 54 32")
    doc = anonymizer.anonymize("Ali : audit qualité et tests.", ali)
    assert "qualité" in doc.text
    assert "Ali " not in doc.text


def test_le_coffre_est_isole_par_task_id(anonymizer):
    """Deux documents, deux task_id, deux mappings sans mélange."""
    doc_a = anonymizer.anonymize("Contact : alex.martin@exemple.fr", ALEX)
    doc_b = anonymizer.anonymize("Alex Martin, développeur.", ALEX)
    assert doc_a.task_id != doc_b.task_id
    vault = anonymizer._vault
    assert vault.retrieve(doc_a.task_id) != vault.retrieve(doc_b.task_id)


def test_task_id_inconnu_leve_une_erreur():
    anonymizer = DeterministicAnonymizer(PiiVault())
    faux_doc = AnonymizedDocument(task_id="inexistant", text="{{NOM}}")
    with pytest.raises(KeyError):
        anonymizer.enrich(faux_doc)

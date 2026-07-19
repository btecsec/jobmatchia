# JobMatch AI — Code officiel du livre

Code source des labos du livre **« L'IA de Zéro à Production : Du Concept à l'Agent Sécurisé »** (BTECSEC, Édition 2026, Amazon KDP).

Chapitre après chapitre, ce dépôt construit **JobMatch AI** : une plateforme SaaS de candidature automatisée multi-agents, *privacy-by-design*.

## La règle d'or du projet

> Aucun agent IA ne reçoit ni ne traite jamais de données d'identification directe (nom, prénom, âge, adresse, coordonnées). Un module déterministe (non-IA) anonymise ces données **avant** tout appel LLM ; un autre module déterministe les réinjecte **en toute fin de pipeline**. Tous les textes de ce dépôt sont synthétiques et anonymisés.

## Installation

```bash
git clone https://github.com/btecsec/jobmatchia
cd jobmatchia
python -m venv .venv
# Windows : .venv\Scripts\activate | Linux/macOS : source .venv/bin/activate
pip install -r requirements.txt
```

## Contenu par chapitre

| Chapitre | Module | Démo | Tests |
|---|---|---|---|
| 1 — C'est quoi l'IA, vraiment ? | `jobmatch/skills/detector.py` | `python -m jobmatch.skills.detector` | `pytest tests/test_detector.py` |
| 2 — Le cerveau des LLM | `jobmatch/nlp/tokenize_demo.py`, `jobmatch/nlp/attention.py` | `python -m jobmatch.nlp.tokenize_demo` · `python -m jobmatch.nlp.attention` | `pytest tests/test_attention.py` |

Lancer toute la suite de tests :

```bash
pytest tests/ -v
```

Le dépôt grandit avec le livre : chaque nouveau chapitre ajoute sa brique jusqu'au SaaS complet (anonymiseur, Agent Profileur, modèle de scoring, RAG, agents orchestrés, sécurité, déploiement).

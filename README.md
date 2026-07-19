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
| 3 — L'écosystème des outils IA | `jobmatch/ml/frameworks_tour.py` | `python -m jobmatch.ml.frameworks_tour` | `pytest tests/test_frameworks_tour.py` |
| 4 — Data cleaning et règle d'or | `jobmatch/data/cleaning.py`, `jobmatch/privacy/anonymizer.py` | `python -m jobmatch.data.cleaning` · `python -m jobmatch.privacy.anonymizer` | `pytest tests/test_cleaning.py tests/test_anonymizer.py` |
| 5 — Entraîner et évaluer le scoring CV/offre | `jobmatch/ml/scoring.py` | `python -m jobmatch.ml.scoring` | `pytest tests/test_scoring.py` |
| 6 — « L'affaire du CV fantôme » (audit de biais) | `jobmatch/fairness/audit.py` | `python -m jobmatch.fairness.audit` | `pytest tests/test_fairness.py` |
| 7 — Versioning et model registry | `jobmatch/mlops/registry.py` | `python -m jobmatch.mlops.registry` | `pytest tests/test_registry.py` |
| 8 — CI/CD et LLM-as-a-Judge | `jobmatch/mlops/gates.py`, `jobmatch/mlops/llm_judge.py` | `python -m jobmatch.mlops.gates` · `python -m jobmatch.mlops.llm_judge` | `pytest tests/test_gates_judge.py` |

> **Note Python** : le Chapitre 3 utilise TensorFlow/Keras, qui nécessite Python 3.12 au plus tant que les wheels 3.13+/3.14 ne sont pas publiées.

Lancer toute la suite de tests :

```bash
pytest tests/ -v
```

Le dépôt grandit avec le livre : chaque nouveau chapitre ajoute sa brique jusqu'au SaaS complet (anonymiseur, Agent Profileur, modèle de scoring, RAG, agents orchestrés, sécurité, déploiement).

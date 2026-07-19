"""API de scoring JobMatch AI — Chapitre 9.

Le modèle du Chapitre 5 devient un service HTTP : FastAPI valide chaque
requête contre un schéma strict, le modèle prédit, la réponse est
typée. C'est la porte d'entrée de la zone IA — et le schéma d'entrée
EST la règle d'or : il n'existe aucun champ où écrire un nom, un email
ou un âge. `extra="forbid"` rejette tout champ inattendu avec une
erreur 422 : la barrière PII est ici un contrat de schéma, vérifiable
et testé, pas une consigne de documentation.
"""

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

from jobmatch.ml.scoring import (
    Offer,
    Profile,
    generate_dataset,
    pair_features,
    train_scoring_model,
)

# Seuil de décision : un réglage PRODUIT, choisi au Chapitre 5 sur le
# jeu de validation. En production réelle, il viendrait de la config.
THRESHOLD = 0.5

app = FastAPI(
    title="JobMatch AI — API de scoring",
    description="Scoring de compatibilité profil/offre. Zone IA : "
                "aucune donnée identifiante n'entre ici.",
    version="1.0.0",
)

# Au démarrage : entraînement déterministe (seed fixe) sur le dataset
# synthétique. En production réelle, on chargerait l'alias `production`
# du registry du Chapitre 7 — même modèle, autre provenance.
_x, _y = generate_dataset()
_model = train_scoring_model(_x, _y)


class ScoringRequest(BaseModel):
    """Une paire (profil anonymisé, offre publique) — rien d'autre.

    extra="forbid" : envoyer un champ "name" ou "email" ne sera pas
    ignoré, il sera REFUSÉ (422). La règle d'or au niveau du schéma.
    """

    model_config = ConfigDict(extra="forbid")

    skills: list[str] = Field(min_length=1, max_length=50)
    years: int = Field(ge=0, le=60)
    offer_required: list[str] = Field(min_length=1, max_length=50)
    offer_min_years: int = Field(ge=0, le=60)


class ScoringResponse(BaseModel):
    probability: float
    compatible: bool
    threshold: float


@app.get("/health")
def health() -> dict:
    """Sonde de vivacité pour l'orchestrateur (Docker, Kubernetes)."""
    return {"status": "ok", "model": "scoring-v1"}


@app.post("/score", response_model=ScoringResponse)
def score(request: ScoringRequest) -> ScoringResponse:
    profile = Profile(skills=frozenset(s.lower() for s in request.skills),
                      years=request.years)
    offer = Offer(required=frozenset(s.lower()
                                     for s in request.offer_required),
                  min_years=request.offer_min_years)
    probability = float(_model.predict_proba([pair_features(profile,
                                                            offer)])[0, 1])
    return ScoringResponse(probability=round(probability, 3),
                           compatible=probability >= THRESHOLD,
                           threshold=THRESHOLD)

"""Agent Chasseur d'offres — Chapitre 10 : RAG production-grade.

RAG (Retrieval-Augmented Generation) en trois temps :
R — retrouver les documents pertinents dans une base vectorielle ;
A — les assembler dans un prompt avec consignes et citations ;
G — laisser un LLM générer la réponse à partir de CE contexte.

Ce module implémente R et A intégralement, avec un vrai index FAISS,
et prépare G sous forme d'un prompt prêt à envoyer à n'importe quel
LLM (OpenRouter et ses modèles gratuits, par exemple). Le point que
les tutoriels ratent : R se MESURE (recall@k sur un jeu de requêtes
étalonnées) — c'est la partie du RAG qu'on peut tester en CI sans
dépenser un jeton.

Embeddings : en production, des embeddings denses appris (modèles
sentence-transformers, par exemple). Ici, des vecteurs TF-IDF —
locaux, déterministes, gratuits — qui suffisent à faire tourner et
tester toute la mécanique. L'interface embed(texts) est le seul point
à changer pour brancher un vrai modèle.

Règle d'or : les offres sont des documents PUBLICS ; les requêtes sont
construites depuis le profil ANONYMISÉ (compétences, expérience).
Aucune PII n'entre dans l'index, le prompt ni les logs.
"""

from dataclasses import dataclass

import faiss
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass(frozen=True)
class Offer:
    """Une offre d'emploi publique, identifiée pour la citation."""

    offer_id: str
    title: str
    text: str


# Corpus synthétique : le « marché » où chasse l'agent.
OFFERS = [
    Offer("O1", "DevOps senior",
          "Nous cherchons un profil DevOps confirmé : Docker, Kubernetes, "
          "Terraform, pipelines CI/CD GitLab, monitoring Prometheus. "
          "Cinq ans d'expérience minimum sur des plateformes cloud AWS."),
    Offer("O2", "Data engineer",
          "Construction de pipelines de données : Python, SQL, Airflow, "
          "Spark. Modélisation d'entrepôts, qualité de données, dbt. "
          "Environnement GCP, trois ans d'expérience souhaités."),
    Offer("O3", "Développeur backend Python",
          "API REST avec FastAPI, PostgreSQL, tests pytest, architecture "
          "hexagonale. Vous rejoignez une équipe produit qui déploie en "
          "continu sur Kubernetes."),
    Offer("O4", "Frontend React",
          "Applications React et TypeScript, design system, tests "
          "Playwright, accessibilité. Sensibilité UX exigée."),
    Offer("O5", "ML engineer",
          "Industrialisation de modèles scikit-learn et PyTorch : "
          "serving FastAPI, monitoring de dérive, feature store, MLflow. "
          "Vous travaillerez avec les data scientists et l'équipe platform."),
    Offer("O6", "Administrateur systèmes Linux",
          "Parc de serveurs Debian, Ansible, supervision Zabbix, "
          "durcissement sécurité, scripting Bash et Python."),
    Offer("O7", "Ingénieur sécurité applicative",
          "Revue de code, pentest applicatif, OWASP, sécurisation "
          "d'API et de pipelines CI/CD, sensibilisation des équipes."),
    Offer("O8", "Data analyst",
          "Tableaux de bord Power BI, SQL avancé, storytelling de "
          "données, échanges quotidiens avec les équipes métier."),
    Offer("O9", "Ingénieur plateforme LLM",
          "Déploiement de modèles de langage open source, RAG, bases "
          "vectorielles FAISS et pgvector, orchestration LangChain, "
          "évaluation continue et maîtrise des coûts d'inférence."),
    Offer("O10", "SRE",
          "Fiabilité de services à fort trafic : SLO, astreintes, "
          "Kubernetes multi-cluster, Terraform, incident management."),
]

# Jeu d'évaluation étalonné : pour chaque requête-type d'un profil,
# l'offre qu'un recruteur humain jugerait LA plus pertinente.
GOLD_QUERIES = [
    ("docker kubernetes terraform ci/cd aws", "O1"),
    ("python sql airflow spark pipelines de données", "O2"),
    ("api rest fastapi postgresql pytest", "O3"),
    ("react typescript design system", "O4"),
    ("serving de modèles mlflow monitoring de dérive", "O5"),
    ("rag base vectorielle faiss langchain llm", "O9"),
    ("automatisation de serveurs linux avec ansible", "O6"),
    # Ambiguë : « monitoring » apparaît dans plusieurs offres — le
    # top-1 peut se tromper, le top-3 rattrape.
    ("monitoring de production kubernetes", "O10"),
    # Décalage lexical volontaire : « conteneurisées » ne figure dans
    # aucune offre — un recruteur pense O1 (Docker/Kubernetes), mais un
    # index purement lexical n'a que les mots pour raisonner.
    ("déploiement continu d'applications conteneurisées", "O1"),
]


def chunk_text(text: str, max_words: int = 40, overlap: int = 10) -> list[str]:
    """Découpe un document en morceaux qui se chevauchent.

    Le chevauchement évite qu'une information à cheval sur deux chunks
    devienne introuvable. Nos offres sont courtes ; sur de vrais
    documents (rapports, documentations), le chunking est LE réglage
    qui fait ou défait un RAG.
    """
    words = text.split()
    if len(words) <= max_words:
        return [text]
    chunks, start = [], 0
    while start < len(words):
        chunks.append(" ".join(words[start:start + max_words]))
        if start + max_words >= len(words):
            break
        start += max_words - overlap
    return chunks


class OfferHunter:
    """L'index vectoriel de l'Agent Chasseur : construire, chercher, citer."""

    def __init__(self, offers: list[Offer]) -> None:
        self.offers = offers
        corpus = [f"{offer.title}. {offer.text}" for offer in offers]
        self._vectorizer = TfidfVectorizer(lowercase=True)
        vectors = self._vectorizer.fit_transform(corpus).toarray()
        vectors = vectors.astype(np.float32)
        faiss.normalize_L2(vectors)          # cosinus via produit scalaire
        self._index = faiss.IndexFlatIP(vectors.shape[1])
        self._index.add(vectors)

    def embed(self, texts: list[str]) -> np.ndarray:
        """Le seul point à changer pour passer aux embeddings denses."""
        vectors = self._vectorizer.transform(texts).toarray().astype(
            np.float32)
        faiss.normalize_L2(vectors)
        return vectors

    def retrieve(self, query: str, k: int = 3) -> list[tuple[Offer, float]]:
        """Top-k des offres les plus proches de la requête (score cosinus)."""
        scores, indices = self._index.search(self.embed([query]), k)
        return [(self.offers[i], float(s))
                for i, s in zip(indices[0], scores[0]) if i != -1]

    def build_prompt(self, query: str, k: int = 3) -> str:
        """L'assemblage du contexte : consignes, sources numérotées, question.

        Les consignes anti-hallucination ne sont pas décoratives : un
        LLM sans contexte invente des offres plausibles — avec ce
        prompt, il doit citer [n] ou avouer qu'il ne sait pas.
        """
        retrieved = self.retrieve(query, k)
        sources = "\n\n".join(
            f"[{n}] ({offer.offer_id}) {offer.title} — {offer.text}"
            for n, (offer, _) in enumerate(retrieved, start=1))
        return (
            "Tu es l'Agent Chasseur d'offres de JobMatch AI.\n"
            "Réponds UNIQUEMENT à partir des offres ci-dessous.\n"
            "Cite tes sources avec leur numéro [n]. Si aucune offre ne "
            "convient, dis-le explicitement au lieu d'inventer.\n\n"
            f"OFFRES :\n{sources}\n\n"
            f"PROFIL RECHERCHÉ (anonymisé) : {query}\n")


def recall_at_k(hunter: OfferHunter,
                gold: list[tuple[str, str]], k: int) -> float:
    """La métrique du R de RAG : l'offre attendue est-elle dans le top-k ?"""
    hits = sum(
        expected in [offer.offer_id for offer, _ in hunter.retrieve(query, k)]
        for query, expected in gold)
    return hits / len(gold)


def demo() -> None:
    """Une chasse complète : recherche, scores, prompt assemblé, évaluation."""
    hunter = OfferHunter(OFFERS)

    query = "docker kubernetes terraform pipelines ci/cd cinq ans"
    print(f"Requête (profil anonymisé) : {query!r}\n")
    print(f"{'RANG':<6}{'OFFRE':<8}{'SCORE':>7}   TITRE")
    for rank, (offer, score) in enumerate(hunter.retrieve(query, 3), 1):
        print(f"{rank:<6}{offer.offer_id:<8}{score:>7.3f}   {offer.title}")

    print("\n--- Prompt assemblé (début) ---")
    print("\n".join(hunter.build_prompt(query, 2).splitlines()[:8]))

    print("\n--- Évaluation du retriever sur le jeu étalonné ---")
    for k in (1, 3):
        print(f"recall@{k} : {recall_at_k(hunter, GOLD_QUERIES, k):.2f}")


if __name__ == "__main__":
    demo()

"""Détection de lignes de compétences dans un CV anonymisé — Chapitre 1.

Deux implémentations volontairement mises en concurrence :

- RuleBasedSkillDetector : programme classique, règles écrites à la main.
- MLSkillDetector : Machine Learning, règles APPRISES à partir d'exemples.

Règle d'or JobMatch AI : ce module ne traite QUE du texte déjà anonymisé.
Aucun nom, âge, adresse ou coordonnée ne doit entrer ici (garanti par le
module d'anonymisation construit au Chapitre 3).
"""

from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


@dataclass(frozen=True)
class Prediction:
    """Résultat de classification d'une ligne de CV."""

    line: str
    is_skill: bool
    confidence: float  # 1.0 = certain ; règles à la main => toujours 1.0


class RuleBasedSkillDetector:
    """Approche 'programme classique' : le développeur écrit les règles.

    Fonctionne sur les cas prévus, échoue dès que le réel déborde des
    règles — c'est précisément ce que le labo met en évidence.
    """

    KEYWORDS = frozenset({
        "python", "javascript", "docker", "sql", "git",
        "linux", "api", "react", "kubernetes",
    })

    def predict(self, line: str) -> Prediction:
        words = set(line.lower().replace(",", " ").split())
        hit = bool(words & self.KEYWORDS)
        return Prediction(line=line, is_skill=hit, confidence=1.0)


class MLSkillDetector:
    """Approche Machine Learning : les règles sont apprises des exemples.

    TF-IDF transforme chaque ligne en vecteur de nombres (section 1.4 :
    tout devient nombres), puis une régression logistique apprend la
    frontière entre 'compétence' et 'pas compétence'.
    """

    def __init__(self) -> None:
        self._pipeline = Pipeline([
            ("vectorizer", TfidfVectorizer(ngram_range=(1, 2), lowercase=True)),
            ("classifier", LogisticRegression(max_iter=1000)),
        ])
        self._is_trained = False

    def train(self, lines: list[str], labels: list[bool]) -> None:
        """Ajuste les paramètres du modèle sur des exemples étiquetés
        (apprentissage supervisé, section 1.5)."""
        if len(lines) != len(labels):
            raise ValueError("Chaque ligne doit avoir exactement une étiquette.")
        self._pipeline.fit(lines, labels)
        self._is_trained = True

    def predict(self, line: str) -> Prediction:
        if not self._is_trained:
            raise RuntimeError("Modèle non entraîné : appelez train() d'abord.")
        proba_skill = self._pipeline.predict_proba([line])[0][1]
        return Prediction(
            line=line,
            # bool() natif : predict_proba renvoie des types numpy, et un
            # np.True_ n'est pas un True Python (piège classique en tests).
            is_skill=bool(proba_skill >= 0.5),
            confidence=round(float(proba_skill), 3),
        )


# Mini-dataset d'entraînement : des lignes de CV ANONYMISÉES et étiquetées.
# En production (Partie II), ce dataset sera un vrai fichier versionné avec DVC.
TRAINING_LINES: list[str] = [
    # Compétences (label True) — avec et sans mot-clé évident
    "Développement backend en Python et Django",
    "Administration de bases de données SQL et PostgreSQL",
    "Mise en place de pipelines CI/CD et conteneurisation Docker",
    "Développement d'API REST sécurisées",
    "Scripting d'automatisation sous Linux",
    "Intégration continue avec Git et GitHub Actions",
    "Création d'interfaces web avec React",
    "Orchestration de conteneurs avec Kubernetes",
    "Conception de microservices et architecture backend",
    "Automatisation de déploiements applicatifs",
    # Non-compétences (label False) — pièges inclus
    "Passionné de randonnée et de photographie",
    "Congé parental de deux ans",
    "Soigneur animalier : entretien du vivarium du python royal",
    "Permis B, véhicule personnel",
    "Trésorier de l'association sportive locale",
    "Recherche un poste en télétravail partiel",
    "Centres d'intérêt : cuisine, jardinage, lecture",
    "Disponible immédiatement, mobilité nationale",
    "Bénévolat en refuge animalier",
    "Membre du club d'échecs",
]
TRAINING_LABELS: list[bool] = [True] * 10 + [False] * 10


def demo() -> None:
    """Compare les deux approches sur des cas que les règles gèrent mal."""
    tricky_lines = [
        "Développement backend avec le framework Django",   # Python implicite
        "Soigneur animalier : nourrissage du python royal",  # faux positif règles
        "Automatisation de tests et intégration continue",   # aucun mot-clé
    ]

    rules = RuleBasedSkillDetector()
    ml = MLSkillDetector()
    ml.train(TRAINING_LINES, TRAINING_LABELS)

    print(f"{'LIGNE':<55} {'RÈGLES':<8} {'ML (confiance)'}")
    for line in tricky_lines:
        r, m = rules.predict(line), ml.predict(line)
        print(f"{line:<55} {str(r.is_skill):<8} {m.is_skill} ({m.confidence})")


if __name__ == "__main__":
    demo()

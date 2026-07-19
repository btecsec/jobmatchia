"""Gates de qualité pour la CI/CD de JobMatch AI — Chapitre 8.

Une gate est un contrat exécutable : une fonction qui examine un
artefact (dataset, modèle, sorties générées) et rend un verdict binaire
motivé. En CI, une gate qui échoue casse le build — la discussion
s'arrête là, quel que soit l'enthousiasme du vendredi.

Trois gates, une par famille de risque :
- data_gate     : les données d'entraînement sont-elles saines ?
- model_gate    : le challenger bat-il le champion sur LA métrique
                  produit (celle du manifeste, Ch.7) ?
- safety_gate   : les sorties générées fuient-elles des PII (Ch.4) ?
"""

from dataclasses import dataclass

import numpy as np

from jobmatch.privacy.anonymizer import Identity, find_pii_leaks


@dataclass(frozen=True)
class GateResult:
    """Un verdict de gate : binaire, motivé, loggable."""

    name: str
    passed: bool
    reason: str

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"[{status}] {self.name} — {self.reason}"


def data_gate(x: np.ndarray, y: np.ndarray,
              min_rows: int = 200,
              positive_range: tuple[float, float] = (0.02, 0.5)) -> GateResult:
    """Santé du dataset : taille, valeurs manquantes, déséquilibre plausible.

    Un dataset qui dérive silencieusement (Ch.9) commence souvent par
    violer une de ces bornes — autant le savoir avant d'entraîner.
    """
    name = "data_gate"
    if len(y) < min_rows:
        return GateResult(name, False,
                          f"{len(y)} lignes < minimum {min_rows}")
    if np.isnan(x).any():
        return GateResult(name, False, "valeurs manquantes dans les features")
    rate = float(np.mean(y))
    low, high = positive_range
    if not low <= rate <= high:
        return GateResult(
            name, False,
            f"taux de positifs {rate:.1%} hors bornes [{low:.0%}, {high:.0%}]")
    return GateResult(name, True,
                      f"{len(y)} lignes, {rate:.1%} de positifs")


def model_gate(challenger: dict, champion: dict,
               metric: str = "recall", tolerance: float = 0.02) -> GateResult:
    """Champion/challenger : promotion refusée si LA métrique produit recule.

    La métrique et la tolérance sont des décisions produit prises à
    froid (Ch.5, Ch.7) — pas des paramètres à renégocier à chaud parce
    que « l'accuracy monte ».
    """
    name = "model_gate"
    new, old = challenger[metric], champion[metric]
    if new < old - tolerance:
        return GateResult(
            name, False,
            f"{metric} du challenger {new:.3f} < champion {old:.3f} "
            f"(tolérance {tolerance})")
    return GateResult(name, True,
                      f"{metric} : challenger {new:.3f} vs champion {old:.3f}")


def safety_gate(generated_texts: list[str], identity: Identity) -> GateResult:
    """Aucune sortie générée ne doit contenir de PII : contrat du Ch.4.

    C'est la même fonction find_pii_leaks qui teste l'anonymiseur — la
    barrière est UN SEUL contrat, vérifié à chaque étage du pipeline.
    """
    name = "safety_gate"
    for i, text in enumerate(generated_texts):
        leaks = sorted(set(find_pii_leaks(text, identity)))
        if leaks:
            return GateResult(
                name, False,
                f"fuite PII dans la sortie {i + 1} : {leaks}")
    return GateResult(name, True,
                      f"{len(generated_texts)} sorties inspectées, zéro fuite")


def run_gates(results: list[GateResult]) -> bool:
    """Affiche le rapport et rend le verdict global : TOUTES doivent passer."""
    for result in results:
        print(f"  {result}")
    return all(result.passed for result in results)


def demo() -> None:
    """La CI rejoue l'incident du Chapitre 7 — et cette fois, elle le bloque."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    from jobmatch.ml.scoring import SEED, evaluate, generate_dataset

    x, y = generate_dataset()
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, stratify=y, random_state=SEED)

    champion = evaluate(LogisticRegression(
        max_iter=1000, class_weight="balanced").fit(x_train, y_train),
        x_test, y_test)
    challenger = evaluate(LogisticRegression(
        max_iter=1000).fit(x_train, y_train), x_test, y_test)

    print("Pipeline CI — candidat au déploiement : la « v2 simplifiée »")
    verdict = run_gates([
        data_gate(x, y),
        model_gate(challenger, champion),
    ])
    print(f"Verdict global : {'DÉPLOIEMENT AUTORISÉ' if verdict else 'BUILD CASSÉ'}")

    alex = Identity(prenom="Alex", nom="Martin",
                    email="alex.martin@exemple.fr", telephone="06 12 34 56 78")
    letters = [
        "Votre offre DevOps correspond à mon profil : Docker, Kubernetes, "
        "cinq ans d'expérience en automatisation.",
        "Disponible immédiatement — contactez-moi sur alex.martin@exemple.fr.",
    ]
    print("\nPipeline CI — sorties du générateur de lettres (Ch.11)")
    verdict = run_gates([safety_gate(letters, alex)])
    print(f"Verdict global : {'DÉPLOIEMENT AUTORISÉ' if verdict else 'BUILD CASSÉ'}")


if __name__ == "__main__":
    demo()

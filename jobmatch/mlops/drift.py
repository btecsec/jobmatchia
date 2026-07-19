"""Détection de dérive des données — Chapitre 9.

Un modèle en production est une photographie du monde au moment de son
entraînement. Le monde, lui, continue de bouger : profils plus juniors,
nouvelles compétences à la mode, offres plus exigeantes. Quand les
données de production s'éloignent des données d'entraînement, les
performances se dégradent EN SILENCE — aucune exception, aucun crash,
juste des prédictions de moins en moins pertinentes.

L'instrument de mesure classique : le PSI (Population Stability Index),
qui compare la distribution d'une feature entre une fenêtre de
référence (l'entraînement) et une fenêtre de production. Repères
industriels usuels : < 0.1 stable, 0.1 à 0.2 à surveiller, > 0.2 alerte
— et alerte signifie « déclencher un réentraînement », la boucle rouge
de la Figure 7.1.
"""

import numpy as np

# Seuils conventionnels du PSI (hérités du credit scoring bancaire).
PSI_WATCH = 0.1
PSI_ALERT = 0.2


def population_stability_index(reference: np.ndarray,
                               production: np.ndarray,
                               bins: int = 10) -> float:
    """PSI entre deux échantillons d'une même feature.

    Découpe la plage de référence en déciles, puis compare la part de
    population dans chaque case : PSI = somme des
    (part_prod - part_ref) * ln(part_prod / part_ref).
    Zéro si les distributions coïncident, croissant avec l'écart.
    """
    edges = np.quantile(reference, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf   # la prod peut déborder la plage
    ref_counts = np.histogram(reference, bins=edges)[0]
    prod_counts = np.histogram(production, bins=edges)[0]

    epsilon = 1e-4  # évite les log(0) sur les cases vides
    ref_share = np.maximum(ref_counts / len(reference), epsilon)
    prod_share = np.maximum(prod_counts / len(production), epsilon)
    return float(np.sum((prod_share - ref_share)
                        * np.log(prod_share / ref_share)))


def drift_status(psi: float) -> str:
    if psi >= PSI_ALERT:
        return "ALERTE"
    if psi >= PSI_WATCH:
        return "A SURVEILLER"
    return "stable"


def drift_report(x_reference: np.ndarray, x_production: np.ndarray,
                 feature_names: list[str]) -> list[tuple[str, float, str]]:
    """Le tableau de bord : une ligne par feature, PSI et statut."""
    report = []
    for j, name in enumerate(feature_names):
        psi = population_stability_index(x_reference[:, j],
                                         x_production[:, j])
        report.append((name, psi, drift_status(psi)))
    return report


def demo() -> None:
    """Trois mois plus tard : le marché a bougé, le dashboard le voit."""
    from jobmatch.ml.scoring import generate_dataset

    x_ref, _ = generate_dataset()

    # Fenêtre de production simulée : le marché s'est tendu. Les profils
    # qui candidatent sont plus juniors (marge d'expérience amputée) et
    # couvrent moins bien les compétences demandées.
    rng = np.random.default_rng(7)
    x_prod = x_ref.copy()[rng.permutation(len(x_ref))[:300]]
    x_prod[:, 2] -= 3.0        # marge d'expérience : juniorisation
    x_prod[:, 0] *= 0.75       # couverture des compétences : érosion

    names = ["couverture", "manquantes", "marge_exp", "largeur_profil"]
    print(f"{'FEATURE':<16}{'PSI':>7}   STATUT")
    for name, psi, status in drift_report(x_ref, x_prod, names):
        bar = "#" * min(int(psi * 20), 30)
        print(f"{name:<16}{psi:>7.3f}   {status:<13}{bar}")

    worst = max(drift_report(x_ref, x_prod, names), key=lambda r: r[1])
    print(f"\nPire dérive : {worst[0]} (PSI {worst[1]:.3f}) "
          f"-> déclencher le réentraînement (Ch.7 : nouvelle version, "
          f"gates du Ch.8, promotion)")


if __name__ == "__main__":
    demo()

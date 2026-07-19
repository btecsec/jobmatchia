"""Nettoyage de données, feature engineering et découpage — Chapitre 4.

Le dataset brut simule ce qu'un vrai export de données produit toujours :
doublons exacts, quasi-doublons (casse, espaces), valeurs manquantes,
espaces insécables copiés-collés depuis le web. Toutes les lignes sont
synthétiques et DÉJÀ anonymisées — la règle d'or s'applique dès le dataset.

Pipeline du module :

    lignes brutes -> clean_dataset() -> engineer_features() -> split_dataset()
                     (nettoyage)        (features)             (60/20/20)

Piège central du chapitre, rejouable dans demo() : dédoublonner APRÈS le
découpage laisse des doublons à cheval sur train et test — le modèle est
alors évalué sur des lignes qu'il a apprises par cœur (fuite de données).
"""

import re
import unicodedata

import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42

# Marqueurs de valeur manquante rencontrés dans les exports réels.
MISSING_MARKERS = {"", "n/a", "na", "null", "-"}

# Mini-lexique technique pour la feature nb_mots_techniques.
TECH_WORDS = frozenset({
    "python", "sql", "docker", "api", "rest", "git", "linux", "react",
    "kubernetes", "django", "postgresql", "ci/cd", "backend", "frontend",
})

# Dataset brut : 40 entrées, avec la saleté typique d'un vrai export.
# (  = espace insécable, souvenir d'un copier-coller depuis le web.)
RAW_ROWS: list[dict] = [
    # --- Compétences (label True) ---
    {"texte": "Développement backend en Python et Django", "label": True},
    {"texte": "Administration de bases de données SQL et PostgreSQL", "label": True},
    {"texte": "Mise en place de pipelines CI/CD et conteneurisation Docker", "label": True},
    {"texte": "Développement d'API REST sécurisées", "label": True},
    {"texte": "développement d'api rest sécurisées  ", "label": True},          # quasi-doublon (casse + espaces)
    {"texte": "Scripting d'automatisation sous Linux", "label": True},
    {"texte": "Intégration continue avec Git et GitHub Actions", "label": True},
    {"texte": "Création d'interfaces web avec React", "label": True},
    {"texte": "Orchestration de conteneurs avec Kubernetes", "label": True},
    {"texte": "Orchestration de conteneurs avec Kubernetes", "label": True},    # doublon exact
    {"texte": "Conception de microservices et architecture backend", "label": True},
    {"texte": "Automatisation de déploiements applicatifs", "label": True},
    {"texte": "Optimisation de requêtes SQL, 5 ans d'expérience", "label": True},
    {"texte": "Supervision d'infrastructure Linux et monitoring", "label": True},
    {"texte": "Développement frontend React et intégration d'API", "label": True},
    {"texte": "Modélisation de données et administration PostgreSQL", "label": True},
    {"texte": "Tests automatisés et qualité logicielle, 3 ans d'expérience", "label": True},
    {"texte": "Sécurisation d'API REST : authentification, quotas", "label": True},
    {"texte": None, "label": True},                                             # valeur manquante
    {"texte": "   ", "label": True},                                            # « vide » déguisé
    # --- Non-compétences (label False) ---
    {"texte": "Passionné de randonnée et de photographie", "label": False},
    {"texte": "Congé parental de deux ans", "label": False},
    {"texte": "Soigneur animalier : entretien du vivarium du python royal", "label": False},
    {"texte": "Permis B, véhicule personnel", "label": False},
    {"texte": "permis b, véhicule personnel", "label": False},                  # quasi-doublon (casse)
    {"texte": "Trésorier de l'association sportive locale", "label": False},
    {"texte": "Recherche un poste en télétravail partiel", "label": False},
    {"texte": "Centres d'intérêt : cuisine, jardinage, lecture", "label": False},
    {"texte": "Disponible immédiatement, mobilité nationale", "label": False},
    {"texte": "Bénévolat en refuge animalier", "label": False},
    {"texte": "Membre du club d'échecs", "label": False},
    {"texte": "Membre du club d'échecs", "label": False},                       # doublon exact
    {"texte": "Marathon de Paris terminé en 2023", "label": False},
    {"texte": "Chant choral, 10 ans de pratique", "label": False},
    {"texte": "Cuisine italienne et pâtisserie amateur", "label": False},
    {"texte": "Vice-président du club de lecture", "label": False},
    {"texte": "Séjours linguistiques réguliers en Espagne", "label": False},
    {"texte": "Collectionneur de vinyles des années 70", "label": False},
    {"texte": "N/A", "label": False},                                           # marqueur de manquant
    {"texte": "Entraîneur bénévole de l'équipe de football junior", "label": False},
]


def normalize_text(text: str) -> str:
    """Normalise un texte : unicode NFC, espaces insécables, espaces multiples."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace(" ", " ")
    return re.sub(r"\s+", " ", text).strip()


def clean_dataset(rows: list[dict]) -> pd.DataFrame:
    """Nettoie le dataset brut : manquants, normalisation, dédoublonnage.

    L'ordre des étapes n'est pas décoratif : on normalise AVANT de
    dédoublonner, sinon « Permis B » et « permis b  » comptent pour deux.
    """
    df = pd.DataFrame(rows)

    # 1. Valeurs manquantes : None, chaînes vides, marqueurs type « N/A ».
    df = df.dropna(subset=["texte"])
    df = df[~df["texte"].str.strip().str.lower().isin(MISSING_MARKERS)]

    # 2. Normalisation du texte.
    df = df.assign(texte=df["texte"].map(normalize_text))

    # 3. Dédoublonnage, insensible à la casse (clé normalisée jetable).
    df = df.assign(_cle=df["texte"].str.casefold())
    df = df.drop_duplicates(subset="_cle").drop(columns="_cle")

    return df.reset_index(drop=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute les features numériques : le texte devient des colonnes de nombres.

    Chaque feature encode une intuition métier vérifiable — pas de magie.
    """
    textes = df["texte"]
    return df.assign(
        n_caracteres=textes.str.len(),
        n_mots=textes.str.split().str.len(),
        annees_exp=(
            textes.str.extract(r"(\d+)\s+ans\b", expand=False)
            .fillna(0).astype(int)),
        nb_mots_techniques=textes.map(
            lambda t: sum(1 for w in re.split(r"[\s,:;']+", t.lower())
                          if w in TECH_WORDS)),
    )


def split_dataset(
    df: pd.DataFrame, seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Découpe 60/20/20 train/validation/test, stratifié sur le label.

    Stratifier = garder la même proportion de True/False dans chaque
    morceau. Deux appels successifs : d'abord isoler le test (20 %),
    puis séparer la validation (25 % du reste = 20 % du total).
    """
    train_val, test = train_test_split(
        df, test_size=0.2, stratify=df["label"], random_state=seed)
    train, val = train_test_split(
        train_val, test_size=0.25, stratify=train_val["label"],
        random_state=seed)
    return train, val, test


def count_leaks(train: pd.DataFrame, test: pd.DataFrame) -> int:
    """Nombre de textes présents (à la casse près) dans train ET test.

    Doit valoir 0. Chaque fuite est une ligne que le modèle a « apprise
    par cœur » et sur laquelle on prétend pourtant l'évaluer.
    """
    train_keys = set(train["texte"].map(normalize_text).str.casefold())
    test_keys = set(test["texte"].map(normalize_text).str.casefold())
    return len(train_keys & test_keys)


def demo() -> None:
    """Nettoyage complet, puis démonstration chiffrée de la fuite de données."""
    print(f"Lignes brutes           : {len(RAW_ROWS)}")
    df = clean_dataset(RAW_ROWS)
    n_supprimees = len(RAW_ROWS) - len(df)
    print(f"Après nettoyage         : {len(df)} ({n_supprimees} lignes retirées)")
    balance = df["label"].mean()
    print(f"Équilibre des classes   : {balance:.0%} de compétences")

    df = engineer_features(df)
    print("\nAperçu des features (3 premières lignes) :")
    apercu = df[["texte", "n_mots", "annees_exp", "nb_mots_techniques"]].head(3)
    print(apercu.to_string(index=False, max_colwidth=45))

    train, val, test = split_dataset(df)
    print(f"\nDécoupage stratifié     : train={len(train)}  "
          f"validation={len(val)}  test={len(test)}")
    print(f"Compétences par morceau : train={train['label'].mean():.0%}  "
          f"validation={val['label'].mean():.0%}  test={test['label'].mean():.0%}")
    print(f"Fuites train↔test       : {count_leaks(train, test)}")

    # Contre-exemple : découper le dataset SALE (avant dédoublonnage).
    df_sale = pd.DataFrame(RAW_ROWS).dropna(subset=["texte"])
    df_sale = df_sale[~df_sale["texte"].str.strip().str.lower()
                      .isin(MISSING_MARKERS)]
    train_sale, _, test_sale = split_dataset(df_sale.reset_index(drop=True))
    print(f"\n⚠️  Même découpage SANS dédoublonnage préalable : "
          f"{count_leaks(train_sale, test_sale)} fuite(s) train↔test")


if __name__ == "__main__":
    demo()

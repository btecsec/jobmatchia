"""Tests du nettoyage de données et du découpage — Chapitre 4."""

import pandas as pd
import pytest

from jobmatch.data.cleaning import (
    RAW_ROWS,
    clean_dataset,
    count_leaks,
    engineer_features,
    normalize_text,
    split_dataset,
)


@pytest.fixture(scope="module")
def df_propre():
    return clean_dataset(RAW_ROWS)


def test_normalisation():
    assert normalize_text("  Permis B,   véhicule  ") == "Permis B, véhicule"


def test_plus_de_manquants_ni_de_doublons(df_propre):
    assert df_propre["texte"].notna().all()
    assert (df_propre["texte"].str.strip() != "").all()
    assert not df_propre["texte"].str.casefold().duplicated().any()


def test_le_nettoyage_retire_les_bonnes_lignes(df_propre):
    # 40 brutes - 3 manquants (None, "   ", "N/A") - 4 doublons = 33
    assert len(RAW_ROWS) == 40
    assert len(df_propre) == 33


def test_features_numeriques(df_propre):
    df = engineer_features(df_propre)
    ligne = df[df["texte"].str.contains("Optimisation de requêtes SQL")].iloc[0]
    assert ligne["annees_exp"] == 5
    assert ligne["nb_mots_techniques"] >= 1
    assert (df["n_mots"] > 0).all()
    assert (df["annees_exp"] >= 0).all()


def test_decoupage_60_20_20(df_propre):
    train, val, test = split_dataset(df_propre)
    assert len(train) + len(val) + len(test) == len(df_propre)
    assert len(train) == pytest.approx(0.6 * len(df_propre), abs=1)
    assert len(val) == pytest.approx(0.2 * len(df_propre), abs=1)
    assert len(test) == pytest.approx(0.2 * len(df_propre), abs=1)


def test_stratification(df_propre):
    """Chaque morceau garde (à peu près) l'équilibre global des classes."""
    global_ratio = df_propre["label"].mean()
    for part in split_dataset(df_propre):
        assert part["label"].mean() == pytest.approx(global_ratio, abs=0.15)


def test_aucune_fuite_apres_nettoyage(df_propre):
    train, val, test = split_dataset(df_propre)
    assert count_leaks(train, test) == 0
    assert count_leaks(train, val) == 0
    assert count_leaks(val, test) == 0


def test_decoupage_reproductible(df_propre):
    train_a, _, test_a = split_dataset(df_propre, seed=42)
    train_b, _, test_b = split_dataset(df_propre, seed=42)
    assert train_a["texte"].tolist() == train_b["texte"].tolist()
    assert test_a["texte"].tolist() == test_b["texte"].tolist()


def test_le_contre_exemple_fuit():
    """Sans dédoublonnage préalable, des doublons finissent à cheval
    sur plusieurs morceaux : c'est la fuite de données du chapitre."""
    df_sale = pd.DataFrame(RAW_ROWS).dropna(subset=["texte"])
    df_sale = df_sale[df_sale["texte"].str.strip().str.len() > 3]
    df_sale = df_sale.reset_index(drop=True)
    fuites = 0
    for seed in range(10):
        train, val, test = split_dataset(df_sale, seed=seed)
        fuites += count_leaks(train, test) + count_leaks(train, val)
    assert fuites > 0

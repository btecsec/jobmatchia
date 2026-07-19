"""Tests du modèle de scoring CV/offre — Chapitre 5."""

import numpy as np
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from jobmatch.ml.scoring import (
    SEED,
    Offer,
    Profile,
    evaluate,
    generate_dataset,
    pair_features,
    train_scoring_model,
)


@pytest.fixture(scope="module")
def dataset():
    return generate_dataset()


@pytest.fixture(scope="module")
def splits(dataset):
    x, y = dataset
    return train_test_split(x, y, test_size=0.25, stratify=y,
                            random_state=SEED)


def test_dataset_reproductible(dataset):
    x, y = dataset
    x2, y2 = generate_dataset()
    assert np.array_equal(x, x2)
    assert np.array_equal(y, y2)


def test_desequilibre_realiste(dataset):
    """La minorité de paires compatibles émerge de la règle métier."""
    _, y = dataset
    ratio = y.mean()
    assert 0.05 <= ratio <= 0.20


def test_features_bornees():
    profile = Profile(skills=frozenset({"python", "sql"}), years=5)
    offer = Offer(required=frozenset({"python", "docker"}), min_years=3)
    coverage, missing, margin, largeur = pair_features(profile, offer)
    assert 0.0 <= coverage <= 1.0
    assert missing == 1.0        # docker manque
    assert margin == 2.0         # 5 ans - 3 requis
    assert largeur == 2.0


def test_le_dummy_a_une_bonne_accuracy_et_un_rappel_nul(splits):
    """LE piège du chapitre : accuracy flatteuse, utilité nulle."""
    x_train, x_test, y_train, y_test = splits
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(x_train, y_train)
    metrics = evaluate(dummy, x_test, y_test)
    assert metrics["accuracy"] >= 0.85
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0


def test_le_modele_bat_le_dummy_sur_le_f1(splits):
    x_train, x_test, y_train, y_test = splits
    model = train_scoring_model(x_train, y_train)
    metrics = evaluate(model, x_test, y_test)
    assert metrics["f1"] >= 0.5
    assert metrics["recall"] >= 0.8


def test_matrice_de_confusion_coherente(splits):
    x_train, x_test, y_train, y_test = splits
    model = train_scoring_model(x_train, y_train)
    tn, fp, fn, tp = confusion_matrix(y_test, model.predict(x_test)).ravel()
    assert tn + fp + fn + tp == len(y_test)
    assert tp + fn == int(y_test.sum())


def test_monter_le_seuil_augmente_la_precision(splits):
    """Le curseur précision/rappel : plus exigeant = plus précis."""
    x_train, x_test, y_train, y_test = splits
    model = train_scoring_model(x_train, y_train)
    proba = model.predict_proba(x_test)[:, 1]
    from sklearn.metrics import precision_score, recall_score
    p_bas = precision_score(y_test, proba >= 0.3, zero_division=0)
    p_haut = precision_score(y_test, proba >= 0.7, zero_division=0)
    r_bas = recall_score(y_test, proba >= 0.3, zero_division=0)
    r_haut = recall_score(y_test, proba >= 0.7, zero_division=0)
    assert p_haut > p_bas
    assert r_haut < r_bas

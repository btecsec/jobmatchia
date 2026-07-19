"""Tests du mini model registry — Chapitre 7."""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from jobmatch.ml.scoring import SEED, evaluate, generate_dataset
from jobmatch.mlops.registry import ModelRegistry, dataset_fingerprint


@pytest.fixture(scope="module")
def data():
    x, y = generate_dataset()
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, stratify=y, random_state=SEED)
    return x, y, x_train, x_test, y_train, y_test


@pytest.fixture()
def registry(tmp_path):
    return ModelRegistry(tmp_path / "registry")


@pytest.fixture(scope="module")
def trained(data):
    _, _, x_train, x_test, y_train, y_test = data
    params = {"class_weight": "balanced", "max_iter": 1000}
    model = LogisticRegression(**params).fit(x_train, y_train)
    return model, params, evaluate(model, x_test, y_test)


def test_empreinte_deterministe_et_sensible(data):
    x, y, *_ = data
    assert dataset_fingerprint(x, y) == dataset_fingerprint(x, y)
    y_altered = y.copy()
    y_altered[0] = not y_altered[0]
    assert dataset_fingerprint(x, y) != dataset_fingerprint(x, y_altered)


def test_versions_incrementales(registry, data, trained):
    x, y, *_ = data
    model, params, metrics = trained
    fingerprint = dataset_fingerprint(x, y)
    assert registry.register(model, params, metrics, fingerprint) == "v1"
    assert registry.register(model, params, metrics, fingerprint) == "v2"
    assert registry.versions() == ["v1", "v2"]


def test_manifeste_complet(registry, data, trained):
    x, y, *_ = data
    model, params, metrics = trained
    version = registry.register(model, params, metrics,
                                dataset_fingerprint(x, y))
    record = registry.manifest(version)
    assert record.params == params
    assert record.metrics == metrics
    assert record.dataset_hash == dataset_fingerprint(x, y)


def test_chargement_restitue_le_meme_modele(registry, data, trained):
    x, y, _, x_test, _, _ = data
    model, params, metrics = trained
    version = registry.register(model, params, metrics,
                                dataset_fingerprint(x, y))
    loaded = registry.load_model(version)
    assert np.array_equal(loaded.predict(x_test), model.predict(x_test))


def test_promotion_et_modele_de_production(registry, data, trained):
    x, y, _, x_test, _, _ = data
    model, params, metrics = trained
    version = registry.register(model, params, metrics,
                                dataset_fingerprint(x, y))
    registry.promote(version, "production")
    assert registry.stage_of("production") == version
    prod = registry.production_model()
    assert np.array_equal(prod.predict(x_test), model.predict(x_test))


def test_promotion_refuse_l_inconnu(registry):
    with pytest.raises(KeyError):
        registry.promote("v99", "production")


def test_stage_inconnu_refuse(registry, data, trained):
    x, y, *_ = data
    model, params, metrics = trained
    version = registry.register(model, params, metrics,
                                dataset_fingerprint(x, y))
    with pytest.raises(ValueError):
        registry.promote(version, "prod-vendredi-soir")


def test_rollback_est_une_re_promotion(registry, data, trained):
    """Revenir en arrière = redéplacer l'alias. Rien n'est détruit."""
    x, y, *_ = data
    model, params, metrics = trained
    fingerprint = dataset_fingerprint(x, y)
    v1 = registry.register(model, params, metrics, fingerprint)
    v2 = registry.register(model, params, metrics, fingerprint)
    registry.promote(v1, "production")
    registry.promote(v2, "production")
    registry.promote(v1, "production")
    assert registry.stage_of("production") == v1
    assert registry.versions() == [v1, v2]  # v2 existe toujours


def test_production_vide_leve(registry):
    with pytest.raises(LookupError):
        registry.production_model()


def test_reproductible_depuis_le_manifeste(registry, data, trained):
    """Le manifeste suffit à réentraîner un modèle bit à bit identique."""
    x, y, x_train, x_test, y_train, _ = data
    model, params, metrics = trained
    version = registry.register(model, params, metrics,
                                dataset_fingerprint(x, y))
    record = registry.manifest(version)
    assert record.dataset_hash == dataset_fingerprint(x, y)
    retrained = LogisticRegression(**record.params).fit(x_train, y_train)
    assert np.array_equal(retrained.predict(x_test), model.predict(x_test))

"""Mini model registry — Chapitre 7 : versioning de modèles.

Un model registry industriel (MLflow, par exemple) repose sur trois
idées simples, que ce module implémente en version minimale et
inspectable :

1. chaque modèle entraîné devient une VERSION immuable (v1, v2, ...),
   archivée avec son manifeste : hyperparamètres, métriques, et
   l'empreinte exacte des données d'entraînement ;
2. des ALIAS mobiles (staging, production) pointent vers une version —
   déployer ou revenir en arrière, c'est déplacer un alias, jamais
   écraser un fichier ;
3. tout est REPRODUCTIBLE : le manifeste contient ce qu'il faut pour
   réentraîner le modèle à l'identique et le prouver.

C'est aussi, en germe, l'idée de DVC : versionner une petite empreinte
(un hash) dans git, pendant que le gros fichier vit dans un cache.
"""

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np


def dataset_fingerprint(x: np.ndarray, y: np.ndarray) -> str:
    """Empreinte SHA-256 du dataset : la carte d'identité des données.

    Un seul octet change — une étiquette corrigée, une ligne dédupliquée —
    et l'empreinte change du tout au tout. C'est elle qui répond à la
    question d'audit : « ce modèle a été entraîné sur QUELLES données ? »
    """
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(x).tobytes())
    digest.update(np.ascontiguousarray(y).tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class ModelRecord:
    """Le manifeste d'une version : tout sauf le modèle lui-même."""

    version: str
    params: dict
    metrics: dict
    dataset_hash: str
    created_at: str


class ModelRegistry:
    """Registre de modèles sur disque : versions immuables, alias mobiles."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._aliases_file = self.root / "aliases.json"

    # ---- versions immuables -------------------------------------------

    def register(self, model, params: dict, metrics: dict,
                 dataset_hash: str) -> str:
        """Archive un modèle entraîné et retourne son numéro de version."""
        version = f"v{len(self.versions()) + 1}"
        folder = self.root / version
        folder.mkdir()
        joblib.dump(model, folder / "model.joblib")
        record = ModelRecord(
            version=version, params=params, metrics=metrics,
            dataset_hash=dataset_hash,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        (folder / "manifest.json").write_text(
            json.dumps(asdict(record), indent=2, ensure_ascii=False),
            encoding="utf-8")
        return version

    def versions(self) -> list[str]:
        found = [p.name for p in self.root.glob("v*") if p.is_dir()]
        return sorted(found, key=lambda name: int(name[1:]))

    def manifest(self, version: str) -> ModelRecord:
        path = self.root / version / "manifest.json"
        if not path.exists():
            raise KeyError(f"version inconnue : {version}")
        return ModelRecord(**json.loads(path.read_text(encoding="utf-8")))

    def load_model(self, version: str):
        # Sécurité : joblib repose sur pickle, qui peut exécuter du code
        # arbitraire au chargement. On ne charge ici que des artefacts
        # écrits par CE registre sur disque local — jamais un fichier
        # téléchargé ou fourni par un utilisateur (cf. Ch.17).
        self.manifest(version)  # valide l'existence
        return joblib.load(self.root / version / "model.joblib")

    # ---- alias mobiles ------------------------------------------------

    def promote(self, version: str, stage: str) -> None:
        """Fait pointer un alias (staging, production) vers une version.

        Déployer = déplacer l'alias. Revenir en arrière = le redéplacer.
        Aucun artefact n'est jamais modifié ni supprimé : l'historique
        complet reste auditable.
        """
        if stage not in ("staging", "production"):
            raise ValueError(f"stage inconnu : {stage}")
        self.manifest(version)  # refuse de promouvoir l'inexistant
        aliases = self._read_aliases()
        aliases[stage] = version
        self._aliases_file.write_text(json.dumps(aliases, indent=2),
                                      encoding="utf-8")

    def stage_of(self, stage: str) -> str | None:
        return self._read_aliases().get(stage)

    def production_model(self):
        version = self.stage_of("production")
        if version is None:
            raise LookupError("aucun modèle en production")
        return self.load_model(version)

    def _read_aliases(self) -> dict:
        if self._aliases_file.exists():
            return json.loads(self._aliases_file.read_text(encoding="utf-8"))
        return {}


def demo() -> None:
    """Deux versions, une promotion, une mauvaise surprise, un rollback."""
    import tempfile

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    from jobmatch.ml.scoring import SEED, evaluate, generate_dataset

    x, y = generate_dataset()
    data_hash = dataset_fingerprint(x, y)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, stratify=y, random_state=SEED)

    registry = ModelRegistry(Path(tempfile.mkdtemp(prefix="jobmatch_reg_")))
    print(f"Empreinte du dataset : {data_hash[:16]}…")

    # v1 : le modèle du Chapitre 5, dans les règles de l'art.
    params_v1 = {"class_weight": "balanced", "max_iter": 1000}
    m1 = LogisticRegression(**params_v1).fit(x_train, y_train)
    v1 = registry.register(m1, params_v1, evaluate(m1, x_test, y_test),
                           data_hash)
    registry.promote(v1, "production")

    # v2 : un collègue pressé retire class_weight « pour simplifier ».
    params_v2 = {"class_weight": None, "max_iter": 1000}
    m2 = LogisticRegression(**params_v2).fit(x_train, y_train)
    v2 = registry.register(m2, params_v2, evaluate(m2, x_test, y_test),
                           data_hash)
    registry.promote(v2, "production")

    print(f"\n{'VERSION':<9}{'CLASS_WEIGHT':<14}{'ACCURACY':>9}"
          f"{'RECALL':>8}{'F1':>7}   STAGE")
    for version in registry.versions():
        record = registry.manifest(version)
        stage = "production" if registry.stage_of(
            "production") == version else ""
        print(f"{version:<9}{str(record.params['class_weight']):<14}"
              f"{record.metrics['accuracy']:>9.3f}"
              f"{record.metrics['recall']:>8.3f}"
              f"{record.metrics['f1']:>7.3f}   {stage}")

    # Vendredi 18 h : le rappel s'est effondré en production. Rollback.
    print("\nRollback : production -> v1 (une ligne, zéro suppression)")
    registry.promote(v1, "production")
    print(f"Production actuelle : {registry.stage_of('production')}")

    # La preuve de reproductibilité : manifeste -> réentraînement -> bit à bit.
    record = registry.manifest(v1)
    retrained = LogisticRegression(**record.params).fit(x_train, y_train)
    identical = bool(
        (retrained.predict(x_test) == m1.predict(x_test)).all())
    assert record.dataset_hash == dataset_fingerprint(x, y)
    print(f"Réentraîné depuis le manifeste v1 : prédictions identiques "
          f"bit à bit = {identical}")


if __name__ == "__main__":
    demo()

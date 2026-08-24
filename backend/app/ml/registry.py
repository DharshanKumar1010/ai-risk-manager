"""Append-only access to ``models/registry.json``.

The registry is what makes a past prediction traceable: given a ``model_version`` from an
audit row, it names the algorithm, the training window, the exact ``feature_version`` behind
the inputs, the hyperparameters, the seed, and the held-out numbers the model was accepted on.

It is append-only, and that is enforced here rather than asked for in a comment. Every phase
from 2 to 6 writes to this file; one careless ``json.dump(new_entries)`` would erase the
provenance of every model trained before it, and the erased rows are exactly the ones an
auditor would want. :func:`append_entry` re-reads, appends and rewrites, and refuses outright
if the prior contents are not a list or if a ``model_id`` is being reused.
"""

import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path, PurePath
from typing import Any

#: Repo-root ``models/`` from ``backend/app/ml/registry.py`` — up four to the repo root.
DEFAULT_REGISTRY_PATH = Path(__file__).resolve().parents[3] / "models" / "registry.json"

#: Where trained weights are written. Gitignored; the registry is the versioned part.
DEFAULT_ARTIFACT_DIR = DEFAULT_REGISTRY_PATH.parent / "artifacts"

#: ``model_id`` shape: ``<layer>-<algorithm>-<corpus>-<utc timestamp>``. Constrained so an id
#: is always safe to use as a filename component.
MODEL_ID_PATTERN = re.compile(r"^[a-z0-9]+(?:[-_][a-z0-9]+)*$")


def utc_stamp(moment: datetime | None = None) -> str:
    """Return a compact UTC timestamp for a model id.

    Args:
        moment: Override for the current time. Passed explicitly by tests so an id is
            reproducible; production calls leave it None.
    """
    return (moment or datetime.now(UTC)).strftime("%Y%m%dT%H%M%SZ")


def build_model_id(layer: str, algorithm: str, corpus: str, moment: datetime | None = None) -> str:
    """Return a registry model id, unique per training run.

    Args:
        layer: e.g. ``tier1_anomaly``.
        algorithm: e.g. ``lightgbm``.
        corpus: The ``source_dataset`` this model was trained on. Present because no model
            in this project trains across corpora, so the id must say which one it saw.
        moment: Override for the current time.
    """
    slug = f"{layer}-{algorithm}-{corpus}-{utc_stamp(moment)}".lower().replace("_", "-")
    if not MODEL_ID_PATTERN.match(slug):
        raise ValueError(f"Model id {slug!r} contains characters that are unsafe in a filename")
    return slug


def require_bare_model_id(model_id: str) -> None:
    """Reject a model id that is not a single, relative path component.

    Raises:
        ValueError: If the id is empty, a parent reference, or carries a separator, a drive or
            a root.
    """
    if not model_id or model_id in {".", ".."}:
        raise ValueError(f"model_id must be a non-empty registry identifier, got {model_id!r}")
    if "/" in model_id or "\\" in model_id or ".." in model_id:
        raise ValueError(
            f"model_id must be a bare registry identifier, got {model_id!r}. An artefact path "
            "is built from it and a traversing id would read or write outside the model "
            "directory."
        )
    candidate = PurePath(model_id)
    if candidate.drive or candidate.root or len(candidate.parts) != 1:
        raise ValueError(
            f"model_id must be a single relative path component, got {model_id!r}. A drive "
            "letter or root makes the artefact path absolute and escapes the model directory."
        )


def artifact_path(model_id: str, directory: Path, suffix: str) -> Path:
    """Return ``directory/model_id+suffix``, proven to sit inside ``directory``.

    The single place a model id becomes a filesystem path, for every tier. Separator checks
    alone are not enough on Windows, which this project is developed on:
    ``Path("models/artifacts") / "C:secret.json"`` evaluates to ``C:secret.json``, because
    pathlib discards the base whenever a segment carries a drive. Validation and resolution
    are kept together because a guard that runs somewhere other than where the path is built
    is a guard a later caller can forget -- which is how the same hole ended up in two tiers.

    Args:
        model_id: Bare registry identifier.
        directory: The artefact directory the result must stay inside.
        suffix: Extension to append, e.g. ``".json"`` or ``".meta.json"``.

    Raises:
        ValueError: If the id is unsafe, or resolves outside ``directory``.
    """
    require_bare_model_id(model_id)
    root = directory.resolve()
    resolved = (root / f"{model_id}{suffix}").resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(
            f"model_id {model_id!r} resolves to {resolved}, outside the artefact directory "
            f"{root}. Refusing to read or write it."
        )
    return resolved


@dataclass
class RegistryEntry:
    """One trained model's permanent record.

    Attributes:
        model_id: Unique; also the ``model_version`` a Tier result reports and an audit row
            stores.
        layer: Which architecture layer this model belongs to.
        algorithm: Human-readable algorithm name.
        source_dataset: The corpus trained on.
        feature_version: Hash from the feature store identifying the exact input definition.
        training_window: ``{"start": iso, "end": iso}`` for the train split actually used.
        hyperparameters: Everything needed to retrain it.
        random_seed: Set and logged, per ml-evaluation-standards section 5.
        heldout_test: The test-split result. Never a validation number.
        baseline_comparison: Losing baselines, kept as documented comparisons rather than
            silently discarded — section 4.
        artifact: Relative path of the saved weights under ``models/``.
        notes: Anything qualifying the result.
    """

    model_id: str
    layer: str
    algorithm: str
    source_dataset: str
    feature_version: str
    training_window: dict[str, str]
    hyperparameters: dict[str, Any]
    random_seed: int
    heldout_test: dict[str, Any]
    baseline_comparison: list[dict[str, Any]] = field(default_factory=list)
    artifact: str | None = None
    notes: list[str] = field(default_factory=list)
    trained_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON shape written to the registry."""
        return {
            "model_id": self.model_id,
            "layer": self.layer,
            "algorithm": self.algorithm,
            "source_dataset": self.source_dataset,
            "trained_at": self.trained_at,
            "training_window": dict(self.training_window),
            "feature_version": self.feature_version,
            "hyperparameters": dict(self.hyperparameters),
            "random_seed": self.random_seed,
            "heldout_test": dict(self.heldout_test),
            "baseline_comparison": list(self.baseline_comparison),
            "artifact": self.artifact,
            "notes": list(self.notes),
        }


def read_registry(path: Path = DEFAULT_REGISTRY_PATH) -> list[dict[str, Any]]:
    """Return every entry currently recorded.

    Raises:
        ValueError: If the file exists but does not hold a JSON list. Appending to something
            of an unexpected shape would either lose it or corrupt it, and both are worse
            than refusing.
    """
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    payload = json.loads(text)
    if not isinstance(payload, list):
        raise ValueError(f"{path} does not contain a JSON list; refusing to append to it")
    return payload


def append_entry(entry: RegistryEntry, path: Path = DEFAULT_REGISTRY_PATH) -> list[dict[str, Any]]:
    """Append one entry, preserving every prior one.

    Raises:
        ValueError: If ``entry.model_id`` is already present. Reusing an id would leave two
            different models answering to one ``model_version``, which breaks the one property
            the registry exists to provide.

    Returns:
        The full registry contents after the append.
    """
    entries = read_registry(path)
    if any(existing.get("model_id") == entry.model_id for existing in entries):
        raise ValueError(
            f"model_id {entry.model_id!r} is already in {path}. The registry is append-only "
            "and ids must stay unique."
        )
    entries.append(entry.to_dict())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")
    return entries


def find_entry(model_id: str, path: Path = DEFAULT_REGISTRY_PATH) -> dict[str, Any] | None:
    """Return one entry by id, or None when it is not recorded."""
    for entry in read_registry(path):
        if entry.get("model_id") == model_id:
            return entry
    return None


def latest_entry(
    layer: str,
    source_dataset: str,
    path: Path = DEFAULT_REGISTRY_PATH,
) -> dict[str, Any] | None:
    """Return the most recently appended entry for one layer and corpus.

    Ordering is by position in the file, not by ``trained_at``: append order is the ground
    truth about which model was accepted last, and a clock-based sort would reorder entries
    if a machine's time were wrong.
    """
    matches = [
        entry
        for entry in read_registry(path)
        if entry.get("layer") == layer and entry.get("source_dataset") == source_dataset
    ]
    return matches[-1] if matches else None

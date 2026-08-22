"""Feature versioning — the link between a stored prediction and the features behind it.

Every scored transaction carries a ``feature_version``. Given one, this module's
:class:`FeatureDefinition` reconstructs exactly which features existed, how they were
parameterised, and which fitted encoder tables were in play. Without that link an audit row
records *that* a decision was made but not *what it was made from*, which is not an audit
trail.

The hash covers four things, and each is there for a reason:

``source_dataset``
    IEEE-CIS and PaySim produce different feature sets from the same pipeline. They must
    never collide on one version string.

``feature_names``
    Sorted, so that reordering the engineering code does not invent a new version.

``parameters``
    Window sizes, thresholds, which columns get frequency-encoded. Changing a window from
    24h to 12h changes the meaning of a feature without changing its name.

``encoder_digest``
    A digest of the *fitted* frequency tables. Two runs over different training data produce
    different encodings under identical code, and a version that ignored this would claim
    two genuinely different feature sets were the same one.

Consequently the version is stable across re-runs on unchanged inputs, and moves whenever
anything that changes a feature's value moves.
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from typing import Any

from app.data.raw_spec import SourceDataset

FEATURE_VERSION_PREFIX = "fv_"

#: 12 hex characters of SHA-256. Collision risk is negligible at the scale of "every feature
#: definition this project will ever produce", and a short string stays readable in an audit
#: row and in ``models/registry.json``.
FEATURE_VERSION_HASH_CHARS = 12


def _reject_unserialisable(value: object) -> Any:
    """Raise rather than coerce a value the hash cannot represent stably.

    ``json.dumps(default=str)`` would silently accept anything, and ``str()`` of a set or a
    dict has no ordering guarantee — two identical definitions could then hash differently.
    Failing here forces parameters to be plain JSON types.
    """
    raise TypeError(
        f"Feature parameters must be JSON-native for the hash to be stable; "
        f"got {type(value).__name__}. Convert it explicitly."
    )


def canonical_json(payload: Mapping[str, Any]) -> str:
    """Serialise a mapping deterministically, for hashing."""
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=_reject_unserialisable,
    )


def digest_encoders(encoders: Mapping[str, Mapping[str, float]]) -> str:
    """Return a stable digest of the fitted frequency-encoding tables.

    Args:
        encoders: Column name to {category: frequency}, as fitted on the train split.

    Returns:
        A short hex digest that changes whenever any fitted frequency changes.
    """
    # Frequencies are rounded before hashing so that a float re-derived from the same data
    # on a different platform does not produce a different version for the same definition.
    rounded = {
        column: {category: round(float(freq), 12) for category, freq in sorted(table.items())}
        for column, table in sorted(encoders.items())
    }
    return sha256(canonical_json(rounded).encode("utf-8")).hexdigest()[:FEATURE_VERSION_HASH_CHARS]


@dataclass(frozen=True)
class FeatureDefinition:
    """One immutable feature definition, identified by its hash.

    Attributes:
        source_dataset: Which corpus this definition applies to.
        feature_names: Every key present in the engineered feature vector.
        parameters: Engineering parameters — window sizes, encoded columns, thresholds.
        encoder_digest: Digest of the fitted encoder tables, from :func:`digest_encoders`.
    """

    source_dataset: SourceDataset
    feature_names: tuple[str, ...]
    parameters: Mapping[str, Any]
    encoder_digest: str

    def __post_init__(self) -> None:
        """Reject definitions that could not be reproduced from their own record."""
        if not self.feature_names:
            raise ValueError("A feature definition must name at least one feature")
        duplicates = sorted(
            {name for name in self.feature_names if self.feature_names.count(name) > 1}
        )
        if duplicates:
            raise ValueError(f"Duplicate feature names: {duplicates}")

    @property
    def canonical_payload(self) -> str:
        """Return the exact string that is hashed into the version."""
        return canonical_json(
            {
                "source_dataset": self.source_dataset,
                "feature_names": sorted(self.feature_names),
                "parameters": dict(self.parameters),
                "encoder_digest": self.encoder_digest,
            }
        )

    @property
    def feature_version(self) -> str:
        """Return the version string stored on every row and every prediction."""
        digest = sha256(self.canonical_payload.encode("utf-8")).hexdigest()
        return FEATURE_VERSION_PREFIX + digest[:FEATURE_VERSION_HASH_CHARS]

    def to_registry_entry(self) -> dict[str, Any]:
        """Return the record shape appended to ``models/registry.json``."""
        return {
            "feature_version": self.feature_version,
            "source_dataset": self.source_dataset,
            "feature_count": len(self.feature_names),
            "feature_names": sorted(self.feature_names),
            "parameters": dict(self.parameters),
            "encoder_digest": self.encoder_digest,
        }

    def describe(self) -> str:
        """Return a human-readable block for the data-quality report."""
        lines = [
            f"- **feature_version**: `{self.feature_version}`",
            f"- **source_dataset**: `{self.source_dataset}`",
            f"- **features** ({len(self.feature_names)}): "
            + ", ".join(f"`{name}`" for name in sorted(self.feature_names)),
            f"- **encoder_digest**: `{self.encoder_digest}`",
            "- **parameters**:",
        ]
        for key, value in sorted(self.parameters.items()):
            lines.append(f"    - `{key}` = `{value}`")
        return "\n".join(lines)

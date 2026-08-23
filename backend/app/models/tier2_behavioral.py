"""Tier-2: per-account behavioural sequence anomaly detection.

Phase 3 implements this layer: a PyTorch LSTM autoencoder trained to reconstruct normal
per-account transaction sequences, using reconstruction error as the anomaly signal and
exposing

    score(sequence: list[TransactionFeatures]) -> Tier2Result

The decision threshold is derived from this dataset's own reconstruction-error
distribution. A threshold from any other project does not transfer here.

--------------------------------------------------------------------------------------

**Why an autoencoder rather than a classifier.** Tier-1 is a supervised classifier and it
is good at what a single row can tell you. Its measured gap (BUILD_LOG Phase 2, Finding 2)
is that it catches 24.6% of fraud by count but only 14.6% by value — the expensive frauds
are the ones no single row betrays. Account takeover is expensive precisely because each
transaction in it looks ordinary; what is not ordinary is the *sequence*. So Tier-2 learns
what an account's normal rhythm looks like and measures departure from it, which is a
question a per-row classifier cannot be asked.

**Three properties that decide whether the number means anything.**

*The error is masked.* Windows are right-padded to ``W``, and every error in this module
divides by the count of real timesteps rather than by ``W``. Dividing by ``W`` would deflate
a 3-step window's error by 7/10 and turn the score into a proxy for how much history an
account has. :func:`masked_reconstruction_error` is the only place this arithmetic lives.

*The bottleneck has to bite.* An autoencoder wide enough to learn the identity map
reconstructs fraud exactly as well as it reconstructs normal behaviour, and its two error
distributions land on top of each other. With ~21 features over 10 timesteps the input is
~210 dimensions, so a 128-unit hidden state is not on its own a bottleneck; the latent
dimension is, and it is swept and selected on validation in ``train_tier2`` rather than
assumed.

*Short histories abstain.* :class:`Tier2Result` carries ``is_scoreable`` and a null
``reconstruction_error``, never a zero. Phase 1 measured 57.7% of IEEE-CIS accounts holding
a single transaction; emitting 0.0 for them would tell Phase 5's meta-learner "maximally
normal" about an account this layer has never had an opinion on. That is the same rule
``Tier1Model._require_compatible`` states as *a missing feature is an error, not a zero*.
"""

import json
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import torch
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence

from app.data.schema import TransactionFeatures
from app.models.tier2_sequences import SequenceWindows, Tier2SequenceSpec

#: p95 serving budget for one scoring call, in milliseconds. The same figure Tier-1 holds:
#: both layers sit behind the one Phase 7 endpoint, so they share a budget rather than each
#: having one. Kept as a local constant rather than imported, because CLAUDE.md's repo map is
#: explicit that the tier modules stay independent of one another.
LATENCY_BUDGET_P95_MS = 50.0

#: Sequential calls in the latency benchmark, matching Tier-1's so the two are comparable.
LATENCY_BENCHMARK_CALLS = 100

#: Rows scored per forward pass during bulk evaluation. Large enough that the per-batch
#: Python overhead disappears, small enough that a ``(batch, W, features)`` tensor stays well
#: inside cache on the 590k-row corpus.
SCORING_BATCH_SIZE = 4096

#: Why a window was not scored. Recorded on the result rather than folded into a score, so an
#: audit row says "Tier-2 had no opinion" instead of implying it had a favourable one.
ABSTAIN_TOO_SHORT = (
    "sequence shorter than the model's minimum length; Tier-2 has no behavioural baseline "
    "for this account and abstains rather than returning a score"
)


class Tier2Result(BaseModel):
    """One Tier-2 decision, as returned to the caller and recorded by the audit trail.

    ``reconstruction_error`` is nullable and that is the point. ``AuditRecord`` types the
    field as ``float | None`` for this case: an account with too little history is not a
    normal account, it is an account this layer cannot speak about, and the audit row should
    say so. Phase 5's meta-learner reads ``is_scoreable`` as an explicit missing-signal
    indicator rather than having to guess whether a 0.0 is a measurement or a placeholder.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    reconstruction_error: float | None = Field(
        default=None,
        ge=0.0,
        description="Masked mean squared reconstruction error over the window's real "
        "timesteps. Higher means the sequence departs further from learned-normal. "
        "Comparable only within one model_version. None when the model abstained.",
    )
    is_anomaly: bool = Field(
        description="reconstruction_error >= the model's operating threshold, which was "
        "chosen on the validation split by minimising estimated cost. Always False when the "
        "model abstained — an abstention is not a clearance, and callers that need to tell "
        "the two apart read is_scoreable.",
    )
    is_scoreable: bool = Field(
        description="Whether the sequence was long enough to score at all.",
    )
    sequence_length: int = Field(
        ge=0,
        description="Real transactions in the window, after truncation to the model's W.",
    )
    abstention_reason: str | None = Field(
        default=None,
        description="Plain-language reason the model had no opinion. None when it scored.",
    )
    latency_ms: float = Field(ge=0.0, description="Wall-clock time for this scoring call.")
    model_version: str = Field(
        min_length=1,
        description="Registry model_id. Resolves through models/registry.json to the exact "
        "architecture, feature definition and held-out numbers this decision rests on.",
    )


def masked_reconstruction_error(
    values: Tensor,
    reconstruction: Tensor,
    mask: Tensor,
) -> Tensor:
    """Return the per-window mean squared error over real timesteps only.

    **The single most load-bearing function in this layer.** Dividing by ``W`` instead of by
    the real-timestep count would give a length-3 window in a length-10 slot seven free
    perfectly-reconstructed steps, deflating its error by 7/10. Since 57.7% of IEEE-CIS
    accounts hold one transaction and the median holds one, that bias runs the same direction
    for most of the corpus: every short-history account would look normal and the score would
    rank accounts by how much history they have. ``tests/test_tier2.py`` pins this by
    asserting a padded window scores identically to the same window unpadded.

    Args:
        values: ``(batch, window, features)`` scaled inputs, padded positions zeroed.
        reconstruction: ``(batch, window, features)`` model output.
        mask: ``(batch, window)`` 1.0 on real timesteps, 0.0 on padding.

    Returns:
        ``(batch,)`` mean squared error per window.
    """
    squared = (values - reconstruction) ** 2
    per_step = (squared * mask.unsqueeze(-1)).sum(dim=2)
    real_steps = mask.sum(dim=1).clamp(min=1.0)
    return per_step.sum(dim=1) / (real_steps * values.shape[2])


def masked_timestep_errors(
    values: Tensor,
    reconstruction: Tensor,
    mask: Tensor,
) -> Tensor:
    """Return ``(batch, window)`` per-timestep mean squared error, zero on padding.

    The enhancement pass the phase brief asks for — "which transactions in the sequence drove
    the reconstruction error most" — falls straight out of this without an attention
    mechanism. Attention weights would be a learned approximation of contribution; the
    per-timestep error *is* the contribution, exactly, and it is what the Phase 8 dashboard
    can defend to a reviewer.
    """
    squared = (values - reconstruction) ** 2
    return squared.mean(dim=2) * mask


class LSTMAutoencoder(nn.Module):
    """Encoder-decoder LSTM over a padded, masked window of transactions.

    The encoder consumes the window through ``pack_padded_sequence``, so a padded step never
    enters the recurrence at all — a stronger guarantee than left-padding plus taking the
    final hidden state, which merely arranges for the padding to be early. Its final hidden
    state is projected to ``latent_size``; the decoder repeats that vector across the window
    and reconstructs every timestep from it.

    Repeating the latent rather than teacher-forcing the true previous step is deliberate. A
    decoder given the real ``x_{t-1}`` can reconstruct ``x_t`` well by copying it forward,
    which makes the reconstruction error small for *any* input, anomalous or not, and the
    whole layer stops discriminating. Forcing every timestep to be regenerated from one fixed
    latent is what makes the bottleneck mean something.
    """

    def __init__(
        self,
        n_features: int,
        hidden_size: int,
        latent_size: int,
        window: int,
    ) -> None:
        """Build the network.

        Args:
            n_features: Per-timestep feature count.
            hidden_size: LSTM hidden units, encoder and decoder alike.
            latent_size: The bottleneck. Must be well below ``window * n_features`` or the
                network can learn the identity map and reconstruct fraud as faithfully as
                normal behaviour.
            window: Timesteps the decoder regenerates.
        """
        super().__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.window = window

        self.encoder = nn.LSTM(n_features, hidden_size, num_layers=1, batch_first=True)
        self.to_latent = nn.Linear(hidden_size, latent_size)
        self.decoder = nn.LSTM(latent_size, hidden_size, num_layers=1, batch_first=True)
        self.to_output = nn.Linear(hidden_size, n_features)

    def encode(self, values: Tensor, lengths: Tensor) -> Tensor:
        """Return the ``(batch, latent_size)`` latent for each window."""
        packed = pack_padded_sequence(values, lengths, batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.encoder(packed)
        return cast(Tensor, self.to_latent(hidden[-1]))

    def forward(self, values: Tensor, lengths: Tensor) -> Tensor:
        """Reconstruct the window.

        Args:
            values: ``(batch, window, features)``, padded positions zeroed.
            lengths: ``(batch,)`` real length per window, on CPU as int64 — what
                ``pack_padded_sequence`` requires.

        Returns:
            ``(batch, window, features)``. Values at padded positions are unconstrained and
            are masked out of every error this module reports.
        """
        latent = self.encode(values, lengths)
        repeated = latent.unsqueeze(1).expand(-1, self.window, -1)
        decoded, _ = self.decoder(repeated)
        return cast(Tensor, self.to_output(decoded))


@dataclass
class Tier2Model:
    """A trained Tier-2 autoencoder with its input definition and operating threshold.

    Attributes:
        model_id: Registry id; also the ``model_version`` on every result.
        algorithm: ``lstm_autoencoder`` or ``dense_autoencoder`` (the non-recurrent control).
        spec: The fitted input definition — features, window, scaler.
        threshold: Reconstruction error at or above which a window is flagged. Chosen on
            validation by minimising estimated cost, never on test.
        network: The torch module, in eval mode.
        hyperparameters: Everything needed to retrain, recorded in the registry.
    """

    model_id: str
    algorithm: str
    spec: Tier2SequenceSpec
    threshold: float
    network: nn.Module
    hyperparameters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Put the network in eval mode and cache the required feature key set."""
        self.network.eval()
        self._required: frozenset[str] = frozenset(self.spec.feature_names)

    @property
    def feature_version(self) -> str:
        """Return the Tier-2 feature version this model was fitted against."""
        return self.spec.to_feature_definition().feature_version

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Return the per-timestep feature names, in matrix column order."""
        return self.spec.feature_names

    # --- Bulk scoring ----------------------------------------------------------------

    def _forward_batch(
        self,
        values: npt.NDArray[np.float32],
        mask: npt.NDArray[np.float32],
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Run one batch and return ``(values, reconstruction, mask)`` as tensors."""
        value_tensor = torch.from_numpy(values)
        mask_tensor = torch.from_numpy(mask)
        lengths = mask_tensor.sum(dim=1).to(torch.int64).clamp(min=1)
        with torch.no_grad():
            reconstruction = self.network(value_tensor, lengths)
        return value_tensor, reconstruction, mask_tensor

    def score_windows(
        self,
        windows: SequenceWindows,
        batch_size: int = SCORING_BATCH_SIZE,
    ) -> npt.NDArray[np.float64]:
        """Score every window in bulk. Used by evaluation and training, never by serving.

        Windows shorter than the spec's ``min_length`` are still scored here and returned;
        filtering them is the caller's job, because evaluation needs to report both the
        scoreable subset and the full split with abstentions counted as unflagged.
        """
        errors = np.empty(len(windows), dtype="float64")
        for start in range(0, len(windows), batch_size):
            index = np.arange(start, min(start + batch_size, len(windows)), dtype=np.intp)
            values, mask = windows.batch(index)
            value_tensor, reconstruction, mask_tensor = self._forward_batch(values, mask)
            batch_errors = masked_reconstruction_error(value_tensor, reconstruction, mask_tensor)
            errors[index] = batch_errors.numpy().astype("float64")
        return errors

    # --- Serving ----------------------------------------------------------------------

    def _require_compatible(self, sequence: Sequence[TransactionFeatures]) -> None:
        """Raise unless the window is one account's own history, in order, correctly versioned.

        Four checks, because four different callers' bugs land here and each produces a
        confidently wrong score rather than an error:

        A ``feature_version`` mismatch means the caller assembled against a different
        definition. A missing key means one assembly is incomplete — and zero-filling it
        would silently change the decision. A window mixing accounts is not a behavioural
        sequence at all. A window out of time order asks the model to read a history that
        never happened, and it will happily return a number for it.
        """
        if not sequence:
            raise ValueError("Tier-2 needs at least one transaction to score")

        for transaction in sequence:
            if transaction.feature_version != self.feature_version:
                raise ValueError(
                    f"Transaction {transaction.transaction_id!r} carries feature_version "
                    f"{transaction.feature_version!r}, but model {self.model_id} was fitted "
                    f"on {self.feature_version!r}. Refusing to score: the vectors are not the "
                    "same definition, and scoring anyway would attach a wrong decision to a "
                    "correct-looking audit row."
                )
            if not self._required.issubset(transaction.features):
                missing = sorted(self._required.difference(transaction.features))
                raise ValueError(
                    f"Transaction {transaction.transaction_id!r} is missing {len(missing)} "
                    f"Tier-2 feature(s): {missing[:10]}{'...' if len(missing) > 10 else ''}. "
                    "A missing feature is an error, not a zero — zero-filling would silently "
                    "change the decision."
                )

        accounts = {transaction.account_id for transaction in sequence}
        if len(accounts) > 1:
            raise ValueError(
                f"A Tier-2 sequence must be one account's own history; got {len(accounts)} "
                "accounts. Tier-2 measures departure from an account's established pattern, "
                "which is undefined across a mixture of accounts."
            )

        times = [transaction.event_time for transaction in sequence]
        if any(later < earlier for earlier, later in zip(times, times[1:], strict=False)):
            raise ValueError(
                "A Tier-2 sequence must be in ascending event_time order, oldest first, with "
                "the transaction being scored last. An out-of-order window describes a "
                "history that never happened."
            )

    def _matrix_from_sequence(
        self, sequence: Sequence[TransactionFeatures]
    ) -> npt.NDArray[np.float32]:
        """Arrange an assembled window into the model's scaled ``(1, W, F)`` input.

        The assembled vectors carry the **unscaled** derived features. Standardisation is
        fitted model state that ships inside ``spec``, so applying it here rather than at
        assembly keeps the caller free of model internals — and leaves the audit row holding
        an ``amount_log`` of 5.2, which a reviewer can read, instead of a z-scored -0.31,
        which they cannot.
        """
        window, n_features = self.spec.window, self.spec.n_features
        values = np.zeros((1, window, n_features), dtype=np.float32)
        means = np.asarray(self.spec.means, dtype="float64")
        stds = np.asarray(self.spec.stds, dtype="float64")

        for position, transaction in enumerate(sequence):
            row = np.array(
                [
                    np.nan if transaction.features[name] is None else transaction.features[name]
                    for name in self.spec.feature_names
                ],
                dtype="float64",
            )
            scaled = np.clip((row - means) / stds, -self.spec.clip, self.spec.clip)
            values[0, position] = np.nan_to_num(scaled, nan=0.0).astype(np.float32)
        return values

    def score(self, sequence: list[TransactionFeatures]) -> Tier2Result:
        """Score one account's trailing window.

        Args:
            sequence: The account's transactions in ascending time order, oldest first, with
                the transaction being scored **last**. Longer than the model's window is fine
                — only the most recent ``W`` are read. Each element carries the assembled
                Tier-2 feature vector and its version.

        Returns:
            The decision, its masked reconstruction error, the measured latency and the model
            version. When the window is shorter than the model's minimum length the result
            abstains: ``reconstruction_error`` is None and ``is_anomaly`` is False.

        Raises:
            ValueError: If the window is empty, spans more than one account, is out of time
                order, or carries the wrong ``feature_version`` or an incomplete vector.
        """
        started = time.perf_counter()
        self._require_compatible(sequence)

        recent = list(sequence)[-self.spec.window :]
        length = len(recent)
        if length < self.spec.min_length:
            return Tier2Result(
                reconstruction_error=None,
                is_anomaly=False,
                is_scoreable=False,
                sequence_length=length,
                abstention_reason=ABSTAIN_TOO_SHORT,
                latency_ms=(time.perf_counter() - started) * 1_000.0,
                model_version=self.model_id,
            )

        values = self._matrix_from_sequence(recent)
        mask = np.zeros((1, self.spec.window), dtype=np.float32)
        mask[0, :length] = 1.0
        value_tensor, reconstruction, mask_tensor = self._forward_batch(values, mask)
        error = float(masked_reconstruction_error(value_tensor, reconstruction, mask_tensor).item())

        return Tier2Result(
            reconstruction_error=error,
            is_anomaly=error >= self.threshold,
            is_scoreable=True,
            sequence_length=length,
            abstention_reason=None,
            latency_ms=(time.perf_counter() - started) * 1_000.0,
            model_version=self.model_id,
        )

    # --- Persistence -------------------------------------------------------------------

    def save(self, directory: Path) -> Path:
        """Write the weights and their sidecar under ``directory``.

        Only the ``state_dict`` is written — a tensor dictionary — never the module object.
        ``torch.save`` of a module pickles it, and loading a pickle executes arbitrary code;
        this load path becomes reachable from the Phase 7 scoring endpoint, which makes that
        a remote-code-execution surface rather than a style question. The matching
        :meth:`load` passes ``weights_only=True`` and reads only from the configured artefact
        directory, never from a caller-supplied path. Same reasoning as
        ``Tier1Model.save``'s choice of LightGBM's text format over a pickle.
        """
        directory.mkdir(parents=True, exist_ok=True)
        artifact = directory / f"{self.model_id}.pt"
        torch.save(self.network.state_dict(), artifact)

        sidecar = directory / f"{self.model_id}.meta.json"
        sidecar.write_text(
            json.dumps(
                {
                    "model_id": self.model_id,
                    "algorithm": self.algorithm,
                    "threshold": self.threshold,
                    "feature_version": self.feature_version,
                    "hyperparameters": dict(self.hyperparameters),
                    "spec": self.spec.to_dict(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return artifact

    @classmethod
    def load(cls, model_id: str, directory: Path) -> "Tier2Model":
        """Rebuild a model from the artefact directory.

        ``model_id`` is resolved against ``directory`` rather than accepted as a path, and it
        is validated to be a bare filename component: a caller-supplied ``../`` would
        otherwise reach outside the artefact directory, which is exactly the traversal the
        security checklist's input-handling section exists to prevent.
        """
        if "/" in model_id or "\\" in model_id or model_id in ("", ".", ".."):
            raise ValueError(
                f"model_id {model_id!r} is not a bare filename component; refusing to resolve "
                "it against the artefact directory."
            )
        sidecar = json.loads((directory / f"{model_id}.meta.json").read_text(encoding="utf-8"))
        spec = Tier2SequenceSpec.from_dict(sidecar["spec"])
        hyperparameters = dict(sidecar["hyperparameters"])
        network = build_network(
            algorithm=sidecar["algorithm"],
            spec=spec,
            hidden_size=int(hyperparameters["hidden_size"]),
            latent_size=int(hyperparameters["latent_size"]),
        )
        state = torch.load(directory / f"{model_id}.pt", weights_only=True, map_location="cpu")
        network.load_state_dict(state)
        return cls(
            model_id=sidecar["model_id"],
            algorithm=sidecar["algorithm"],
            spec=spec,
            threshold=float(sidecar["threshold"]),
            network=network,
            hyperparameters=hyperparameters,
        )


class DenseAutoencoder(nn.Module):
    """The non-recurrent control: an autoencoder over the flattened window.

    Exists to answer one question the LSTM's own metrics cannot — **is the sequence *order*
    doing any work, or only the aggregate?** It sees the same features over the same window
    with the same mask and the same bottleneck, and differs solely in having no recurrence:
    the window is flattened, so any permutation of its timesteps that preserves position
    would be a different input, but nothing in the architecture models transition.

    This is the analogue of Tier-1's matched-inputs LightGBM, which existed so that "the
    supervised model won" could not be confounded with "the supervised model got richer
    inputs". If the dense control matches the LSTM here, then Tier-2's value is the
    account-relative *features* rather than the sequence model, and that is the finding.
    """

    def __init__(
        self,
        n_features: int,
        hidden_size: int,
        latent_size: int,
        window: int,
    ) -> None:
        """Build the network with the same signature as :class:`LSTMAutoencoder`."""
        super().__init__()
        self.n_features = n_features
        self.window = window
        flat = window * n_features
        self.encoder = nn.Sequential(
            nn.Linear(flat, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, latent_size),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, flat),
        )

    def forward(self, values: Tensor, lengths: Tensor) -> Tensor:
        """Reconstruct the window. ``lengths`` is accepted and unused, to match the LSTM."""
        del lengths  # The mask is applied by the loss; the flat model has no recurrence.
        batch = values.shape[0]
        latent = self.encoder(values.reshape(batch, -1))
        return cast(Tensor, self.decoder(latent).reshape(batch, self.window, self.n_features))


def build_network(
    *,
    algorithm: str,
    spec: Tier2SequenceSpec,
    hidden_size: int,
    latent_size: int,
) -> nn.Module:
    """Construct the network for one algorithm name."""
    if algorithm == "lstm_autoencoder":
        return LSTMAutoencoder(spec.n_features, hidden_size, latent_size, spec.window)
    if algorithm == "dense_autoencoder":
        return DenseAutoencoder(spec.n_features, hidden_size, latent_size, spec.window)
    raise ValueError(f"Unknown Tier-2 algorithm {algorithm!r}")


@dataclass(frozen=True)
class TimestepContribution:
    """How much one transaction in the window drove the window's reconstruction error."""

    offset: int
    is_anchor: bool
    error: float
    share: float
    top_features: tuple[tuple[str, float], ...]

    def describe(self) -> str:
        """Return one report line."""
        where = "the scored transaction" if self.is_anchor else f"{self.offset} step(s) back"
        drivers = ", ".join(f"{name} {value:.3f}" for name, value in self.top_features)
        return f"{where}: {100 * self.share:.1f}% of the error ({drivers})"


def explain(
    model: Tier2Model,
    sequence: list[TransactionFeatures],
    top_k: int = 3,
) -> list[TimestepContribution]:
    """Return which transactions in the window drove the reconstruction error, largest first.

    The Phase 3 enhancement pass, and Phase 8's explainability panel reads it directly. No
    attention mechanism is involved: per-timestep squared error *is* the contribution, exactly
    rather than approximately, and it decomposes further to per-feature without any extra
    machinery.

    Returns an empty list when the model abstained — an abstention has no drivers.
    """
    recent = list(sequence)[-model.spec.window :]
    if len(recent) < model.spec.min_length:
        return []
    model._require_compatible(recent)

    length = len(recent)
    values = model._matrix_from_sequence(recent)
    mask = np.zeros((1, model.spec.window), dtype=np.float32)
    mask[0, :length] = 1.0
    value_tensor, reconstruction, mask_tensor = model._forward_batch(values, mask)

    per_step = masked_timestep_errors(value_tensor, reconstruction, mask_tensor)[0].numpy()
    per_feature = ((value_tensor - reconstruction) ** 2)[0].numpy()
    total = float(per_step.sum())

    contributions: list[TimestepContribution] = []
    for position in range(length):
        ranked = np.argsort(-per_feature[position])[:top_k]
        contributions.append(
            TimestepContribution(
                offset=length - 1 - position,
                is_anchor=position == length - 1,
                error=float(per_step[position]),
                share=float(per_step[position] / total) if total > 0 else 0.0,
                top_features=tuple(
                    (model.spec.feature_names[int(i)], float(per_feature[position][int(i)]))
                    for i in ranked
                ),
            )
        )
    return sorted(contributions, key=lambda item: item.error, reverse=True)


def benchmark_latency(
    model: Tier2Model,
    sequences: Sequence[list[TransactionFeatures]],
    calls: int = LATENCY_BENCHMARK_CALLS,
) -> dict[str, float]:
    """Time sequential scoring calls and return the latency percentiles.

    Sequential and one-at-a-time on purpose: it is the serving shape. A batched benchmark
    would report a throughput number dressed up as a latency one. Measures the scoring call
    only — assembling the window from Postgres is Phase 7's job and is budgeted separately.

    Returns:
        ``p50``/``p95``/``p99``/``mean``/``max`` in milliseconds, plus the call count.
    """
    if not sequences:
        raise ValueError("Latency benchmark needs at least one sequence")
    timings: list[float] = []
    for index in range(calls):
        result = model.score(sequences[index % len(sequences)])
        timings.append(result.latency_ms)
    measured = np.asarray(timings, dtype="float64")
    return {
        "calls": float(calls),
        "p50_ms": float(np.percentile(measured, 50)),
        "p95_ms": float(np.percentile(measured, 95)),
        "p99_ms": float(np.percentile(measured, 99)),
        "mean_ms": float(measured.mean()),
        "max_ms": float(measured.max()),
    }

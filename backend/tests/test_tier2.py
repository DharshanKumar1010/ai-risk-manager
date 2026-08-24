"""Tier-2 tests.

As in Tier-1, the tests that matter are not the ones checking a number comes out — they are
the ones checking the number could not have been produced dishonestly. Four carry most of
the weight:

- :func:`test_masked_error_ignores_padding` and its companion
  :func:`test_unmasked_error_would_be_deflated_by_padding`. The second is the important half:
  it computes the error the *wrong* way and asserts it is deflated, so the first test is
  demonstrably measuring something rather than passing vacuously. Get this wrong and Tier-2
  becomes a sequence-length sensor that scores every short-history account as normal —
  and on a corpus where 57.7% of accounts hold one transaction, that is most of them.
- :func:`test_windows_never_read_the_future`, paired with
  :func:`test_future_read_guard_actually_catches_one`. A guard that never fires is
  indistinguishable from a guard that cannot fire, so one test plants a window containing a
  later timestamp and asserts the check rejects it.
- :func:`test_short_sequences_abstain_rather_than_scoring_zero`. A zero here would tell
  Phase 5's meta-learner "maximally normal" about an account this layer never saw.
- :func:`test_model_selection_reads_validation_not_test`, the same guard Phase 2 needed
  after the first implementation ranked candidates on the test result.
"""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch

from app.data.schema import TransactionFeatures
from app.models.tier1_features import DENIED_COLUMNS
from app.models.tier2_behavioral import (
    Tier2Model,
    build_network,
    explain,
    masked_reconstruction_error,
    masked_timestep_errors,
)
from app.models.tier2_sequences import (
    MIN_SEQUENCE_LENGTH,
    SEQUENCE_FEATURE_NAMES,
    SOURCE_COLUMNS,
    aggregate_to_accounts,
    assemble_windows,
    coverage,
    derive_timestep_frame,
    eligible_training_rows,
    find_future_reads,
    fit_sequence_spec,
    order_full_history,
)
from app.models.train_tier2 import (
    ABSTAINED_RANK_SENTINEL,
    assemble,
    run_corpus,
    to_accounts,
)

ROWS = 900
ACCOUNTS = 45


def _synthetic_frame(rows: int = ROWS, seed: int = 11) -> pd.DataFrame:
    """Build a frame shaped like an IEEE-CIS parquet split, with a learnable sequence signal.

    Synthetic rather than sampled from ``data/processed`` so the suite runs without the
    corpora present and in seconds rather than minutes. Accounts are given uneven history
    depths on purpose — that is the condition the padding and abstention logic exists for —
    and fraud is concentrated on a minority of accounts, which is both realistic and what
    makes :func:`eligible_training_rows` have anything to exclude.
    """
    rng = np.random.default_rng(seed)
    start = datetime(2018, 1, 1, tzinfo=UTC)

    # Uneven history: a long tail of one-transaction accounts alongside a few deep ones,
    # mirroring the Phase 1 measurement (57.7% of accounts hold exactly one row).
    account_index = np.concatenate(
        [
            np.arange(ACCOUNTS),  # everyone appears once
            rng.integers(0, ACCOUNTS // 3, rows - ACCOUNTS),  # a third get repeat traffic
        ]
    )
    rng.shuffle(account_index)
    account_id = np.array([f"acct{index:03d}" for index in account_index])

    fraud_accounts = {f"acct{index:03d}" for index in range(4)}
    is_fraud = np.array(
        [name in fraud_accounts and bool(rng.random() < 0.5) for name in account_id]
    )

    amount = rng.lognormal(mean=4.0, sigma=1.0, size=rows)
    prior = pd.Series(account_id).groupby(account_id).cumcount().to_numpy()
    return pd.DataFrame(
        {
            "transaction_id": [str(index) for index in range(rows)],
            "source_dataset": "ieee_cis",
            "event_time": [start + timedelta(minutes=7 * index) for index in range(rows)],
            "amount": amount,
            "account_id": account_id,
            "counterparty_id": pd.Series([None] * rows, dtype="object"),
            "transaction_type": pd.Series(rng.choice(["W", "C", "R"], rows), dtype="object"),
            "is_fraud": is_fraud,
            "amount_log": np.log1p(amount),
            "amount_zscore_vs_own_history": np.where(
                prior >= 2, rng.normal(0.0, 1.0, rows), np.nan
            ),
            "seconds_since_prior_txn": np.where(prior > 0, rng.exponential(3600, rows), np.nan),
            "hour_of_day": rng.integers(0, 24, rows).astype("int16"),
            "day_of_week": rng.integers(0, 7, rows).astype("int16"),
            "velocity_count_1h": rng.integers(1, 5, rows).astype("float64"),
            "velocity_count_24h": rng.integers(1, 20, rows).astype("float64"),
            "velocity_sum_24h": rng.lognormal(5.0, 1.0, rows),
            "device_is_new": rng.random(rows) < 0.3,
            "device_mismatch": rng.random(rows),
            "addr_is_new": rng.random(rows) < 0.2,
            "addr_mismatch": rng.random(rows),
            "account_prior_txn_count": prior.astype("int32"),
            "freq_ProductCD": rng.random(rows),
            "freq_card4": rng.random(rows),
            "freq_card6": rng.random(rows),
            "freq_P_emaildomain": rng.random(rows),
            "feature_version": "fv_synthetic",
        }
    )


@pytest.fixture
def frame() -> pd.DataFrame:
    """Return a synthetic corpus frame."""
    return _synthetic_frame()


@pytest.fixture
def splits(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return chronological train/val/test splits of the synthetic frame."""
    first, second = int(0.7 * len(frame)), int(0.85 * len(frame))
    return {
        "train": frame.iloc[:first].copy(),
        "val": frame.iloc[first:second].copy(),
        "test": frame.iloc[second:].copy(),
    }


@pytest.fixture
def fitted(splits: dict[str, pd.DataFrame]) -> tuple[Any, Any]:
    """Return a spec fitted on the synthetic train split and the assembled corpus."""
    corpus = assemble(splits, window=5)
    return corpus.spec, corpus


# --- The masking guards, the pair that carries the most weight -------------------------


def test_masked_error_ignores_padding() -> None:
    """A padded window scores exactly as it would unpadded.

    The property the whole layer rests on. Reconstruction error divides by the count of real
    timesteps, not by W, so slot count cannot change a window's score.
    """
    torch.manual_seed(0)
    features = 4
    short_values = torch.randn(1, 3, features)
    short_recon = torch.randn(1, 3, features)
    short_mask = torch.ones(1, 3)

    padded_values = torch.zeros(1, 10, features)
    padded_recon = torch.randn(1, 10, features)
    padded_values[0, :3] = short_values[0]
    padded_recon[0, :3] = short_recon[0]
    padded_mask = torch.zeros(1, 10)
    padded_mask[0, :3] = 1.0

    unpadded = masked_reconstruction_error(short_values, short_recon, short_mask)
    padded = masked_reconstruction_error(padded_values, padded_recon, padded_mask)
    assert torch.allclose(unpadded, padded, atol=1e-6)


def test_unmasked_error_would_be_deflated_by_padding() -> None:
    """The wrong arithmetic is measurably wrong, so the test above is not vacuous.

    Averaging over W rather than over real timesteps deflates a length-3 window in a
    length-10 slot by exactly 7/10. Since the median IEEE-CIS account holds one transaction,
    that bias runs one way for most of the corpus and would make the score a proxy for how
    much history an account has.
    """
    torch.manual_seed(0)
    features = 4
    values = torch.zeros(1, 10, features)
    recon = torch.zeros(1, 10, features)
    values[0, :3] = torch.randn(3, features)
    recon[0, :3] = torch.randn(3, features)
    mask = torch.zeros(1, 10)
    mask[0, :3] = 1.0

    masked = float(masked_reconstruction_error(values, recon, mask))
    naive = float(((values - recon) ** 2).mean())
    assert naive == pytest.approx(masked * 3 / 10, rel=1e-5)
    assert naive < masked


def test_timestep_errors_are_zero_on_padding() -> None:
    """Per-timestep contributions, which the explainability panel reads, exclude padding."""
    torch.manual_seed(0)
    values, recon = torch.randn(2, 6, 3), torch.randn(2, 6, 3)
    mask = torch.zeros(2, 6)
    mask[0, :4] = 1.0
    mask[1, :2] = 1.0
    per_step = masked_timestep_errors(values, recon, mask)
    assert torch.all(per_step[0, 4:] == 0.0)
    assert torch.all(per_step[1, 2:] == 0.0)
    assert torch.all(per_step[0, :4] > 0.0)


# --- The leakage guards, likewise paired -----------------------------------------------


def test_windows_never_read_the_future(splits: dict[str, pd.DataFrame]) -> None:
    """No assembled window contains a timestep later than its own anchor."""
    history = order_full_history(splits)
    matrix = np.zeros((len(history), 1), dtype=np.float32)
    windows, order = assemble_windows(matrix, history["account_id"], window=5)
    event_time = np.asarray(history["event_time"].astype("int64").to_numpy()[order], dtype=np.int64)
    assert find_future_reads(windows, event_time) == 0


def test_future_read_guard_actually_catches_one(splits: dict[str, pd.DataFrame]) -> None:
    """Planting a forward-looking timestep makes the guard fire.

    Without this, :func:`test_windows_never_read_the_future` would pass equally well against
    a checker that always returned zero.
    """
    history = order_full_history(splits)
    matrix = np.zeros((len(history), 1), dtype=np.float32)
    windows, order = assemble_windows(matrix, history["account_id"], window=5)
    event_time = np.asarray(history["event_time"].astype("int64").to_numpy()[order], dtype=np.int64)
    # Make one window's oldest timestep the latest moment in the corpus.
    tampered = event_time.copy()
    victim = int(np.flatnonzero(windows.lengths >= 3)[0])
    tampered[windows.gather[victim, 0]] = tampered.max() + 1
    assert find_future_reads(windows, tampered) >= 1


def test_sequence_features_read_no_denied_column() -> None:
    """Tier-2's source columns contain nothing from the Tier-1 deny list.

    ``event_time``, ``TransactionDT`` and ``account_id`` are all monotone or identifying, and
    under a chronological split each separates train from test perfectly. Tier-2 reads
    *deltas* instead — ``seconds_since_prior_txn``, ``amount_zscore_vs_own_history``.
    """
    assert set(SOURCE_COLUMNS) & DENIED_COLUMNS == set()


def test_deny_list_guard_would_catch_a_planted_leak() -> None:
    """The deny-list check fires when a monotone column is planted, so it is not vacuous."""
    from app.models.tier1_features import denied_columns_present

    assert denied_columns_present((*SOURCE_COLUMNS, "TransactionDT")) == ["TransactionDT"]


# --- Window assembly --------------------------------------------------------------------


def test_window_assembly_matches_hand_computation() -> None:
    """Trailing windows grow then slide, are right-padded, and anchor on the last row."""
    accounts = pd.Series(["A", "B", "A", "A", "B", "A"])
    tag = np.arange(6, dtype=np.float32).reshape(-1, 1)
    windows, order = assemble_windows(tag, accounts, window=3)

    contents = {
        int(order[windows.anchor_row[k]]): [
            int(order[j]) for j in windows.gather[k][windows.mask[k]]
        ]
        for k in range(len(windows))
    }
    assert contents[0] == [0]
    assert contents[2] == [0, 2]
    assert contents[3] == [0, 2, 3]
    assert contents[5] == [2, 3, 5]  # slid: the oldest row dropped out at W=3
    assert contents[1] == [1]
    assert contents[4] == [1, 4]


def test_batch_zeroes_padding_and_right_aligns() -> None:
    """Materialised batches carry real data first and zeros after."""
    accounts = pd.Series(["A", "A", "A"])
    tag = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
    windows, _ = assemble_windows(tag, accounts, window=3)
    values, mask = windows.batch(np.arange(3, dtype=np.intp))
    assert values[0].ravel().tolist() == [1.0, 0.0, 0.0]
    assert values[1].ravel().tolist() == [1.0, 2.0, 0.0]
    assert values[2].ravel().tolist() == [1.0, 2.0, 3.0]
    assert mask[1].tolist() == [1.0, 1.0, 0.0]


def test_window_below_minimum_length_is_refused() -> None:
    """Assembling at a W below the minimum scoreable length is an error, not a warning."""
    with pytest.raises(ValueError, match="below MIN_SEQUENCE_LENGTH"):
        assemble_windows(np.zeros((2, 1), dtype=np.float32), pd.Series(["A", "A"]), window=2)


# --- Feature derivation ------------------------------------------------------------------


def test_cyclical_hour_encoding_keeps_midnight_adjacent(frame: pd.DataFrame) -> None:
    """Hour 23 is nearer hour 0 than hour 12 is, which an ordinal encoding gets backwards."""
    hours = frame.copy().head(3)
    hours["hour_of_day"] = pd.Series([23, 0, 12], dtype="int16").to_numpy()
    derived = derive_timestep_frame(hours)
    point = derived[["hour_sin", "hour_cos"]].to_numpy()
    midnight_gap = float(np.linalg.norm(point[0] - point[1]))
    midday_gap = float(np.linalg.norm(point[0] - point[2]))
    assert midnight_gap < midday_gap


def test_missingness_indicators_track_structural_nulls(frame: pd.DataFrame) -> None:
    """``has_prior`` and ``has_zscore`` record what the mean-imputation would otherwise erase."""
    derived = derive_timestep_frame(frame)
    first_of_account = frame["account_prior_txn_count"].to_numpy() == 0
    assert np.all(derived["has_prior"].to_numpy()[first_of_account] == 0.0)
    assert np.all(derived["has_prior"].to_numpy()[~first_of_account] == 1.0)
    zscore_present = frame["amount_zscore_vs_own_history"].notna().to_numpy()
    assert np.array_equal(derived["has_zscore"].to_numpy() == 1.0, zscore_present)


def test_derive_rejects_a_frame_missing_a_source_column(frame: pd.DataFrame) -> None:
    """A missing engineered column fails loudly rather than producing a silent zero."""
    with pytest.raises(ValueError, match="absent from the frame"):
        derive_timestep_frame(frame.drop(columns=["velocity_sum_24h"]))


# --- Training-set eligibility ------------------------------------------------------------


def test_training_set_excludes_whole_fraud_bearing_accounts(
    splits: dict[str, pd.DataFrame],
) -> None:
    """An account with any train fraud is excluded entirely, not just on its fraud rows.

    IEEE-CIS propagates a chargeback forward across an account's later transactions, so a
    fraud-bearing account is a compromised account and its clean-looking earlier rows are
    ambiguous. Training on them would teach the autoencoder that takeover looks normal.
    """
    train = splits["train"]
    eligible = eligible_training_rows(train)
    fraud_accounts = set(train.loc[train["is_fraud"], "account_id"])

    assert not eligible[train["is_fraud"].to_numpy()].any()
    # And the stronger claim: clean rows of a fraud-bearing account are excluded too.
    clean_rows_of_dirty_accounts = (
        ~train["is_fraud"].to_numpy() & train["account_id"].isin(fraud_accounts).to_numpy()
    )
    assert clean_rows_of_dirty_accounts.any(), "fixture must contain this case to test it"
    assert not eligible[clean_rows_of_dirty_accounts].any()


def test_scaler_is_fitted_on_eligible_rows_only(splits: dict[str, pd.DataFrame]) -> None:
    """Standardising against a mean that includes fraud would centre on the wrong behaviour."""
    train = splits["train"]
    eligible = eligible_training_rows(train)
    on_eligible = fit_sequence_spec(train, eligible, "ieee_cis", window=5)
    on_everything = fit_sequence_spec(train, np.ones(len(train), dtype=bool), "ieee_cis", window=5)
    assert on_eligible.means != on_everything.means


def test_fit_refuses_when_no_row_is_eligible(splits: dict[str, pd.DataFrame]) -> None:
    """No eligible rows is an error: the autoencoder would have nothing to learn normal from."""
    train = splits["train"]
    with pytest.raises(ValueError, match="No eligible train rows"):
        fit_sequence_spec(train, np.zeros(len(train), dtype=bool), "ieee_cis", window=5)


def test_feature_version_moves_with_the_scaler(splits: dict[str, pd.DataFrame]) -> None:
    """Two different fitted scalers are two different feature definitions, and hash apart."""
    train = splits["train"]
    eligible = eligible_training_rows(train)
    first = fit_sequence_spec(train, eligible, "ieee_cis", window=5)
    second = fit_sequence_spec(
        train.head(len(train) // 2), eligible[: len(train) // 2], "ieee_cis", window=5
    )
    assert first.to_feature_definition().feature_version != (
        second.to_feature_definition().feature_version
    )


def test_feature_version_is_stable_across_identical_fits(splits: dict[str, pd.DataFrame]) -> None:
    """The same data under the same code produces the same version, or provenance is noise."""
    train = splits["train"]
    eligible = eligible_training_rows(train)
    left = fit_sequence_spec(train, eligible, "ieee_cis", window=5)
    right = fit_sequence_spec(train, eligible, "ieee_cis", window=5)
    assert left.to_feature_definition().feature_version == (
        right.to_feature_definition().feature_version
    )


# --- The serving contract ----------------------------------------------------------------


def _model(spec: Any, threshold: float = 0.0) -> Tier2Model:
    """Return an untrained model wired to ``spec``, for contract tests."""
    torch.manual_seed(3)
    return Tier2Model(
        model_id="tier2-behavioral-lstm-test",
        algorithm="lstm_autoencoder",
        spec=spec,
        threshold=threshold,
        network=build_network(
            algorithm="lstm_autoencoder", spec=spec, hidden_size=8, latent_size=4
        ),
        hyperparameters={"hidden_size": 8, "latent_size": 4},
    )


def _transaction(
    model: Tier2Model,
    index: int,
    *,
    account: str = "acct000",
    minutes: int = 0,
    feature_version: str | None = None,
    drop: str | None = None,
) -> TransactionFeatures:
    """Return one assembled Tier-2 scoring contract."""
    features: dict[str, Any] = {name: 0.25 for name in model.feature_names}
    if drop is not None:
        del features[drop]
    return TransactionFeatures(
        transaction_id=f"t{index}",
        source_dataset="ieee_cis",
        event_time=datetime(2018, 6, 1, tzinfo=UTC) + timedelta(minutes=minutes),
        amount=Decimal("12.3400"),
        account_id=account,
        counterparty_id=None,
        transaction_type="W",
        feature_version=feature_version or model.feature_version,
        features=features,
    )


def test_short_sequences_abstain_rather_than_scoring_zero(fitted: tuple[Any, Any]) -> None:
    """Below the minimum length the model has no opinion, and says so.

    A 0.0 would read to Phase 5's meta-learner as "maximally normal" about an account this
    layer has never seen. ``AuditRecord.tier2_reconstruction_error`` is nullable for exactly
    this case.
    """
    spec, _ = fitted
    model = _model(spec)
    short = [_transaction(model, i, minutes=i) for i in range(spec.min_length - 1)]
    result = model.score(short)

    assert result.is_scoreable is False
    assert result.reconstruction_error is None
    assert result.is_anomaly is False
    assert result.abstention_reason is not None
    assert result.sequence_length == spec.min_length - 1


def test_long_enough_sequences_score(fitted: tuple[Any, Any]) -> None:
    """At or above the minimum length the model returns a real, non-negative error."""
    spec, _ = fitted
    model = _model(spec)
    window = [_transaction(model, i, minutes=i) for i in range(spec.min_length)]
    result = model.score(window)

    assert result.is_scoreable is True
    assert result.reconstruction_error is not None
    assert result.reconstruction_error >= 0.0
    assert result.model_version == model.model_id


def test_score_truncates_to_the_model_window(fitted: tuple[Any, Any]) -> None:
    """A caller may pass more history than W; only the most recent W is read."""
    spec, _ = fitted
    model = _model(spec)
    long_window = [_transaction(model, i, minutes=i) for i in range(spec.window + 4)]
    assert model.score(long_window).sequence_length == spec.window


def test_score_rejects_feature_version_mismatch(fitted: tuple[Any, Any]) -> None:
    """A vector built against another definition is refused, not scored."""
    spec, _ = fitted
    model = _model(spec)
    window = [_transaction(model, i, minutes=i) for i in range(spec.min_length)]
    window[1] = _transaction(model, 1, minutes=1, feature_version="fv_something_else")
    with pytest.raises(ValueError, match="not the same definition"):
        model.score(window)


def test_score_rejects_an_incomplete_vector(fitted: tuple[Any, Any]) -> None:
    """A missing feature is an error, not a zero."""
    spec, _ = fitted
    model = _model(spec)
    window = [_transaction(model, i, minutes=i) for i in range(spec.min_length)]
    window[0] = _transaction(model, 0, minutes=0, drop=SEQUENCE_FEATURE_NAMES[0])
    with pytest.raises(ValueError, match="missing 1 Tier-2 feature"):
        model.score(window)


def test_score_rejects_a_window_spanning_two_accounts(fitted: tuple[Any, Any]) -> None:
    """Departure from "this account's pattern" is undefined across a mixture of accounts."""
    spec, _ = fitted
    model = _model(spec)
    window = [_transaction(model, i, minutes=i) for i in range(spec.min_length)]
    window[0] = _transaction(model, 0, account="acct999", minutes=0)
    with pytest.raises(ValueError, match="one account's own history"):
        model.score(window)


def test_score_rejects_an_out_of_order_window(fitted: tuple[Any, Any]) -> None:
    """An unsorted window describes a history that never happened."""
    spec, _ = fitted
    model = _model(spec)
    window = [_transaction(model, i, minutes=i) for i in range(spec.min_length)]
    window.reverse()
    with pytest.raises(ValueError, match="ascending event_time order"):
        model.score(window)


def test_score_rejects_an_empty_window(fitted: tuple[Any, Any]) -> None:
    """Nothing to score is an error rather than an abstention with a fabricated length."""
    spec, _ = fitted
    model = _model(spec)
    with pytest.raises(ValueError, match="at least one transaction"):
        model.score([])


def test_explain_attributes_the_error_across_timesteps(fitted: tuple[Any, Any]) -> None:
    """Contributions cover every real timestep, sum to one, and name the anchor."""
    spec, _ = fitted
    model = _model(spec)
    window = [_transaction(model, i, minutes=i) for i in range(spec.min_length)]
    contributions = explain(model, window)

    assert len(contributions) == spec.min_length
    assert sum(item.share for item in contributions) == pytest.approx(1.0, rel=1e-5)
    assert sum(1 for item in contributions if item.is_anchor) == 1
    assert all(len(item.top_features) == 3 for item in contributions)


def test_explain_returns_nothing_for_an_abstention(fitted: tuple[Any, Any]) -> None:
    """An abstention has no drivers to report."""
    spec, _ = fitted
    model = _model(spec)
    short = [_transaction(model, i, minutes=i) for i in range(spec.min_length - 1)]
    assert explain(model, short) == []


# --- Persistence -------------------------------------------------------------------------


def test_save_load_roundtrip_reproduces_scores(fitted: tuple[Any, Any], tmp_path: Path) -> None:
    """A reloaded model returns the same error, through the weights-only load path."""
    spec, _ = fitted
    model = _model(spec, threshold=0.5)
    model.save(tmp_path)

    restored = Tier2Model.load(model.model_id, tmp_path)
    window = [_transaction(model, i, minutes=i) for i in range(spec.min_length)]
    original = model.score(window).reconstruction_error
    reloaded = restored.score(window).reconstruction_error

    assert original is not None and reloaded is not None
    assert reloaded == pytest.approx(original, rel=1e-6)
    assert restored.threshold == model.threshold
    assert restored.feature_version == model.feature_version


def test_saved_artifact_is_a_tensor_dict_not_a_pickled_module(
    fitted: tuple[Any, Any], tmp_path: Path
) -> None:
    """The artefact loads under ``weights_only=True``.

    This load path becomes reachable from the Phase 7 scoring endpoint. A pickled module
    would execute arbitrary code on load; a state dict cannot.
    """
    spec, _ = fitted
    model = _model(spec)
    artifact = model.save(tmp_path)
    state = torch.load(artifact, weights_only=True, map_location="cpu")
    assert isinstance(state, dict)
    assert all(isinstance(value, torch.Tensor) for value in state.values())


@pytest.mark.parametrize("model_id", ["../escape", "sub/dir", "..", ""])
def test_load_refuses_a_path_traversing_model_id(tmp_path: Path, model_id: str) -> None:
    """``model_id`` is resolved against the artefact directory, never used as a path."""
    with pytest.raises(ValueError, match="registry identifier|relative path component"):
        Tier2Model.load(model_id, tmp_path)


# --- Account-level evaluation --------------------------------------------------------------


def test_aggregate_takes_the_max_score_and_any_label() -> None:
    """An account is as suspicious as its most suspicious window, and fraudulent if any row is."""
    scores, labels, amounts, count = aggregate_to_accounts(
        ["a", "a", "b", "b"],
        np.array([0.1, 0.9, 0.2, 0.3]),
        np.array([False, True, False, False]),
        np.array([10.0, 50.0, 5.0, 7.0]),
    )
    assert count == 2
    assert scores.tolist() == [0.9, 0.3]
    assert labels.tolist() == [True, False]
    # A missed account costs the value of its fraudulent rows, not of all its rows.
    assert amounts.tolist() == [50.0, 0.0]


def test_aggregate_skips_abstained_windows() -> None:
    """An account scores on the windows Tier-2 could read, and abstains only if none were."""
    scores, _, _, _ = aggregate_to_accounts(
        ["a", "a", "b"],
        np.array([np.nan, 0.4, np.nan]),
        np.array([False, False, False]),
        np.array([1.0, 1.0, 1.0]),
    )
    assert scores[0] == pytest.approx(0.4)
    assert np.isnan(scores[1])


def test_abstained_accounts_rank_last_and_never_flag(fitted: tuple[Any, Any]) -> None:
    """In the system-level view an abstention is "never flagged", not "judged normal"."""
    _, corpus = fitted
    split = corpus.per_split["test"]
    scores = np.full(len(split.windows), 0.5, dtype="float64")

    everything = to_accounts(split, scores, min_length=99, scoreable_only=False)
    assert everything.n_abstained == everything.n_accounts
    assert np.all(everything.scores == ABSTAINED_RANK_SENTINEL)
    # The sentinel sits below every real score, so no achievable threshold flags it.
    assert ABSTAINED_RANK_SENTINEL < 0.0

    scoreable = to_accounts(split, scores, min_length=99, scoreable_only=True)
    assert scoreable.n_accounts == 0


def test_coverage_reports_fraud_separately() -> None:
    """Fraud coverage is computed, not inferred from row coverage."""
    lengths = np.array([1, 5, 5, 2], dtype=np.int32)
    labels = np.array([True, True, False, True])
    measured = coverage(lengths, labels, min_length=3)
    assert measured["row_coverage"] == pytest.approx(0.5)
    assert measured["fraud_total"] == 3
    assert measured["fraud_scoreable"] == 1
    assert measured["fraud_coverage"] == pytest.approx(1 / 3)


# --- The split discipline ------------------------------------------------------------------


def test_model_selection_reads_validation_not_test(
    splits: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    """The shipped model is chosen on validation account PR-AUC, never on the test result.

    The same guard Phase 2 needed: its first implementation ranked candidates by
    ``candidate.result.pr_auc``, which is the *test* result — the contamination
    ml-evaluation-standards section 1 forbids, since the winner would then have been chosen
    using the split its headline is quoted from.
    """
    report = run_corpus(
        splits,
        datetime(2026, 1, 1, tzinfo=UTC),
        epochs=1,
        artifact_dir=tmp_path / "artifacts",
        registry_path=tmp_path / "registry.json",
        reports_dir=tmp_path / "reports",
    )
    trained = [candidate for candidate in report.candidates if candidate.model is not None]
    assert report.winner.validation_pr_auc >= report.runner_up.validation_pr_auc
    # The winner's validation score is the best among trained candidates, up to the
    # re-measurement at the chosen abstention threshold N.
    assert report.winner in trained


def test_run_reports_coverage_and_the_length_diagnostic(
    splits: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    """A run produces the numbers the phase is gated on, not just a PR-AUC."""
    report = run_corpus(
        splits,
        datetime(2026, 1, 1, tzinfo=UTC),
        epochs=1,
        artifact_dir=tmp_path / "artifacts",
        registry_path=tmp_path / "registry.json",
        reports_dir=tmp_path / "reports",
    )
    assert 0.0 <= report.coverage["fraud_coverage"] <= 1.0
    assert 0.0 <= report.coverage["account_coverage"] <= 1.0
    assert -1.0 <= report.error_length_correlation <= 1.0
    assert report.plots["error_distribution"].exists()
    assert report.plots["loss_curve"].exists()
    # Test is scored once, so every candidate carries a test result on the test split.
    assert all(candidate.result.split == "test" for candidate in report.candidates)
    # The deployed-system number covers every test account; the model-only number covers
    # fewer, and is the higher of the two by construction.
    assert report.scoreable_only.rows <= report.winner.result.rows


def test_windows_reaching_back_are_counted_not_hidden(
    splits: dict[str, pd.DataFrame],
) -> None:
    """Test windows may read train-period history, and the amount is measured."""
    corpus = assemble(splits, window=5)
    assert corpus.per_split["train"].reaches_earlier_split == 0
    assert corpus.per_split["test"].reaches_earlier_split >= 0
    # Whatever the count, the one-directional guarantee still holds.
    event_time = np.asarray(
        corpus.history["event_time"].astype("int64").to_numpy()[corpus.order], dtype=np.int64
    )
    assert find_future_reads(corpus.all_windows, event_time) == 0


def test_early_stopping_set_does_not_read_test_labels(splits: dict[str, pd.DataFrame]) -> None:
    """Flipping a test label must not change which windows early stopping averages over.

    The regression guard for a real bug: the first implementation built the clean-validation
    set from ``is_fraud`` across *all* splits, so a test-split label helped decide when
    training stopped. That is contamination even though no test score was read — the model
    that ships would have been shaped by the split its headline is quoted from.
    """
    baseline = assemble(splits, window=5)

    tampered = {name: frame.copy() for name, frame in splits.items()}
    # Flip every test label; the validation set the loss is averaged over must be unmoved.
    tampered["test"]["is_fraud"] = ~tampered["test"]["is_fraud"].to_numpy()
    flipped = assemble(tampered, window=5)

    assert np.array_equal(baseline.clean_val, flipped.clean_val)
    assert np.array_equal(baseline.train_eligible, flipped.train_eligible)


def test_training_windows_are_train_anchored_and_long_enough(
    splits: dict[str, pd.DataFrame],
) -> None:
    """The autoencoder never fits on a validation or test anchor, nor on a stub window."""
    corpus = assemble(splits, window=5)
    anchors = corpus.all_windows.anchor_row[corpus.train_eligible]
    anchor_splits = corpus.history["split"].to_numpy()[corpus.order][anchors]
    assert set(anchor_splits) == {"train"}
    assert np.all(corpus.all_windows.lengths[corpus.train_eligible] >= MIN_SEQUENCE_LENGTH)

"""Tier-2: per-account behavioural sequence anomaly detection.

Phase 3 implements this layer: a PyTorch LSTM autoencoder trained to reconstruct normal
per-account transaction sequences, using reconstruction error as the anomaly signal and
exposing

    score(sequence: list[TransactionFeatures]) -> Tier2Result

The decision threshold is derived from this dataset's own reconstruction-error
distribution. A threshold from any other project does not transfer here.
"""

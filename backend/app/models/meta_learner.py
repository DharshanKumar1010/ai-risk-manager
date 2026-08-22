"""Meta-learner: fuses the Tier-1/2/3 signals into one calibrated risk score.

Phase 5 implements this layer: an XGBoost classifier over the three tier signals plus the
original engineered features, probability-calibrated with CalibratedClassifierCV, with a
SHAP TreeExplainer supplying per-prediction attribution, exposing

    predict(transaction, tier1, tier2, tier3) -> MetaResult

Raw XGBoost margins are not probabilities. Calibration is required, not optional — an
uncalibrated score undermines the honest-metrics claim this project rests on.
"""

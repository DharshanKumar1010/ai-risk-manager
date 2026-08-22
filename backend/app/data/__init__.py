"""The data layer: raw-file validation, source adapters, feature engineering, versioning.

Phase 1 builds this package in the order the pipeline runs:

``raw_spec``      what each raw dataset file is expected to contain
``validate_raw``  verifies a download against that expectation before anything reads it
``schema``        the canonical ``TransactionFeatures`` contract both sources map into
``adapters/``     per-source translation into that contract (and, in Phase 9, Razorpay)
``pipeline``      cleaning, feature engineering, the chronological split, persistence
``feature_store`` the ``feature_version`` hash tying a prediction to a feature definition

The reconciliation rule that shapes all of it: IEEE-CIS and PaySim share a schema but
never share a timeline. Their time bases have no common origin, so splits are computed
per source and no model trains across both. See ``app/data/schema.py``.
"""

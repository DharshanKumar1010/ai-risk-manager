"""Shared model-evaluation machinery.

Phases 2-6 each train a model and each must report it to the same bar — PR-AUC on the
held-out test split, the full confusion matrix, a stated threshold, a base rate and a
false-positive cost estimate. That bar lives in
``.claude/skills/ml-evaluation-standards/SKILL.md``; this package is the code that meets it,
written once so no tier can quietly report to a lower standard than another.
"""

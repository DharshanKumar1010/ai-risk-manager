# Notebooks

Notebooks are versioned; their **outputs are not**. The `nbstripout` pre-commit hook
strips outputs before a commit lands, so diffs stay readable and no dataset rows leak
into git history through a cached cell output.

Phase 1 adds `eda_report.ipynb` — class balance, missing-value rates, and feature
distributions before and after engineering.

Anything a notebook proves must be reproducible from code under `backend/app/`.
A number that exists only in a notebook cell is not a result.

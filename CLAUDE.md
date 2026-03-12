# Project

HN upvote prediction pipeline. Python 3.10+, PyTorch-first. All core models built from scratch.

## Commands

```bash
# Lint and format
ruff check . --fix
ruff format .

# Test
pytest tests/ -v

# Install
pip install -r requirements.txt
```

## Code standards

- Line length: 100 chars (configured in pyproject.toml)
- Ruff rules: E, F, I, W — do not disable or add `noqa` without justification
- Imports: sorted by ruff (isort-compatible). stdlib → third-party → local, one import per line
- Use `pathlib.Path` over `os.path`
- Type hints on all function signatures (args + return)
- No wildcard imports (`from x import *`)

## Project structure

```
src/           — all source code
src/data/      — data loading, preprocessing, splits, content extraction
tests/         — mirrors src/ layout, prefix files with test_
notebooks/     — exploration only, no imports from notebooks into src/
```

- New modules go under `src/`. Every directory needs `__init__.py`
- Tests go in `tests/` matching the module path: `src/data/preprocess.py` → `tests/data/test_preprocess.py`
- Do not put logic in `__init__.py` beyond re-exports

## Data

- Dataset: `hn.csv` (~3.9M rows). Never committed to git (in .gitignore)
- Parquet outputs go to `data/`. Also gitignored
- Use pandas with pyarrow backend for all tabular data
- Always use temporal splits: train ≤2018, val 2019–2020, test 2021–2023. Never shuffle across years

## Git workflow

- Branch off `master` for features: `feature/<name>` or `fix/<name>`
- PR into `master`. CI runs ruff + pytest on all PRs
- Pre-commit hook auto-formats staged files. One commit is enough
- Commit messages: imperative mood, concise. e.g. "Add content extraction pipeline"

## Rules

- Do NOT add docstrings, comments, or type annotations to code you didn't change
- Do NOT create `.md` files other than README.md — all others are gitignored
- Do NOT install new dependencies without asking. Keep requirements.txt minimal
- Do NOT use gensim, HuggingFace tokenizers, or pre-trained embeddings unless explicitly asked — core models are from-scratch PyTorch
- Do NOT put data file paths in source code. Use CLI args or config
- Prefer simple loops and list comprehensions over pandas `.apply()` with lambdas
- When reading CSV/parquet, always specify `dtype` or `columns` to avoid loading unnecessary data
- Test files must be runnable with `pytest` alone — no special fixtures or setup scripts

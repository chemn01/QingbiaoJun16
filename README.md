# Qingbiao Tender Optimization

This repository is a reproducible research and code project for optimizing tender bids under the stochastic multi-stage rule in `qingbiao.md`.

## Source of Truth

- `qingbiao.md` contains the official rule statement in Chinese.
- Do not edit `qingbiao.md` unless the user explicitly requests it.
- `PROJECT_MEMORY.md`, `DECISIONS.md`, `EXPERIMENT_LOG.md`, and `TODO.md` preserve project state across chats.

## Current Objective

The current target bidder is Unit 5 (`X5`). The main objective is to maximize `X5`'s winning probability.

All bid variables `X_i` are discount rates in `[10, 30]`. A larger discount rate means a lower quoted price.

Adjustable variables:

```text
X3, X5, X6, X7, X9, X10, X11, X13, X16, X17, X19, X20
```

## Python Environment

This project is initialized with `uv` and targets Python 3.12.

On each machine, install `uv` once, then from the repository root run:

```bash
uv python install 3.12
uv sync
uv run python --version
```

Use the same commands on the Windows CPU server and the Linux GPU server. Keep common dependencies in `pyproject.toml` with `uv add <package>`. Add GPU-specific packages only on the Linux GPU server after confirming its CUDA/PyTorch compatibility.

## Quality Checks

All code changes should include pytest unit tests and pass:

```bash
uv run ruff check .
uv run mypy
uv run pytest
```

## Neural Surrogate

`neural_surrogate.py` trains a Residual MLP to approximate the current Differential Evolution soft-loss calculation. The input is the full bid vector `X1..X20`, sampled from `[10, 30]`; the label is the average DE soft loss over the 108 discrete scenario combinations.

Local smoke data generation:

```bash
uv run python neural_surrogate.py generate --samples 1024 --seed 42 --output surrogate_runs/smoke/data.npz --workers 1
```

Local smoke training, after PyTorch dependencies are installed:

```bash
uv run python neural_surrogate.py train --data surrogate_runs/smoke/data.npz --output-dir surrogate_runs/smoke/model --epochs 2 --batch-size 128 --device auto
```

Linux GPU server training example:

```bash
uv sync
uv run python neural_surrogate.py generate --samples 131072 --seed 42 --workers -1 --output surrogate_runs/full_131072/data.npz
uv run python neural_surrogate.py train --data surrogate_runs/full_131072/data.npz --output-dir surrogate_runs/full_131072/model --epochs 200 --batch-size 4096 --device auto
```

Prediction expects a JSON file containing either an `X1` through `X20` object, one 20-value list, or a list of 20-value lists:

```bash
uv run python neural_surrogate.py predict --model-dir surrogate_runs/full_131072/model --bids-json bids.json --device auto --compare-exact
```

## Planned Workflow

1. Implement and test the exact rule engine.
2. Enumerate the discrete random choices to compute exact winning probability for a fixed bid vector.
3. Optimize adjustable variables with reproducible Differential Evolution settings.
4. Optionally train a surrogate model and verify final candidates against the exact evaluator.

## Rule-Implementation Priorities

- Keep discount rates and quoted prices conceptually separate.
- Use round-half-up for rule-critical rounding unless a different convention is confirmed.
- Test tie-breaking explicitly.
- Prefer exact enumeration over Monte Carlo when feasible.
- In the final winner stage, first choose the closest bidder among discount rates `X_i > K`; if none exist, choose the bidder with the smallest `|X_i - K|`.

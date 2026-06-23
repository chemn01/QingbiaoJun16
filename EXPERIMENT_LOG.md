# Experiment Log

This file records optimization runs, surrogate-model runs, evaluator checks, parameters, seeds, outputs, and conclusions.

## 2026-06-18

Initial setup notes:

- Target bidder: Unit 5 (`X5`).
- Planned evaluator: exact rule engine by enumeration over the discrete random choices in `qingbiao.md`.
- Planned optimizer route: Differential Evolution over adjustable discount rates in `[10, 30]`.
- Planned surrogate route: train and validate a neural-network surrogate against exact evaluator outputs before using it for candidate generation.

Implementation update:

- Added `de_softmax_optimizer.py`, a Differential Evolution optimizer for `X5`.
- Added `numpy` and `scipy` dependencies and created a local `.venv` with `uv --cache-dir .uv-cache sync`.
- Optimizer defaults: `samples=8192`, `maxiter=800`, `popsize=90`, `workers=-1`, `seed=42`, `checkpoint_every=50`.
- Optimizer output directory default: `de_softmax_results/`.
- Commit pushed to GitHub: `48f0953 Add differential evolution softmax optimizer`.

Validation implementation update:

- Added `validate_de_results.py`, a high-precision validator for DE checkpoints/results.
- Validator default flow: read `best_result.json` and `checkpoint_iter_*.json`; choose the top 20 by surrogate objective; validate with 32768 Sobol samples over the 8 non-adjustable units; exactly enumerate all 324 discrete rule scenarios per sample; refine the top 5 true-probability candidates with 65536 samples.
- Validator outputs: `validation_summary.csv`, `validation_results.json`, and `validation_report.txt`.
- Added `tests/test_validate_de_results.py` for candidate loading, TopN behavior, bid mapping, evaluator consistency with `forward_bidding_calculator.py`, count helpers, and refinement selection.

Smoke checks:

- Ran `uv --cache-dir .uv-cache run python -m unittest discover -s tests -v`.
- Result after optimizer implementation: 3 tests passed.
- Ran a tiny optimizer smoke command with `samples=16`, `maxiter=1`, `popsize=3`, `workers=1`, `checkpoint_every=1`, `--no-polish`, `--quiet`.
- Result: completed successfully and wrote checkpoint/final result files.
- Ran `uv --cache-dir .uv-cache run python -m unittest discover -s tests -v` after adding the validator and increasing optimizer default samples.
- Result: 7 tests passed.
- Ran a tiny validator smoke command against `de_softmax_smoke_results` with `samples=4`, `top=1`, `refine-top=1`, `refine-samples=8`, `workers=1`, and `chunk-size=2`.
- Result: completed successfully and wrote CSV/JSON/TXT outputs under `/private/tmp/qingbiao_validate_smoke`.

Notes:

- No full high-precision optimization run has been recorded yet.
- No full high-precision validation run on a real optimizer output directory has been recorded yet.

## 2026-06-22

Rule update implementation:

- Updated code to match the revised `qingbiao.md` final winner rule: after `K = K1 + K2`, prioritize finalists with `X_i > K`; if none exist, select the finalist with the smallest `|X_i-K|`.
- Updated the forward calculator, DE validation evaluator, DE surrogate objective, and tests.
- Existing generated optimizer and validation outputs were produced under the previous final-stage interpretation and should be regenerated before drawing conclusions from their probabilities.

## 2026-06-23

Neural surrogate implementation:

- Added `neural_surrogate.py` with `generate`, `train`, and `predict` CLI commands.
- Implemented the first label definition: complete `X1..X20` bid vector to average DE soft loss over 108 discrete scenario combinations, reusing `de_softmax_optimizer.evaluate_scenario_loss`.
- Implemented a PyTorch Residual MLP training path with input normalization, standardized labels, Huber loss, AdamW, automatic `cuda > mps > cpu` device selection, and saved model/config/normalization/metrics artifacts.
- Added `tests/test_neural_surrogate.py` for bid splitting, deterministic labels, dataset generation, prediction payload parsing, and training smoke behavior when PyTorch is installed.
- Ran `uv run --no-sync python neural_surrogate.py generate --samples 16 --seed 42 --output surrogate_runs/smoke/data.npz --workers 1`; result: completed in about 0.07 seconds, label range roughly `[0.0247, 3.9989]`.
- Ran `uv run --no-sync ruff check .`, `uv run --no-sync mypy`, and `uv run --no-sync pytest -q`; result: all passed, with the PyTorch training smoke test skipped locally because PyTorch was not installed.
- Attempted local `uv sync`; it repeatedly stalled while downloading the 83.9 MiB torch wheel, so full training smoke remains pending until dependency sync succeeds locally or on the Linux GPU server.

Full surrogate run artifact analysis:

- Dataset command used on the Linux server: `uv run python neural_surrogate.py generate --samples 131072 --seed 42 --workers 60 --output surrogate_runs/full_131072/data.npz`.
- Full artifact path synced locally: `surrogate_runs/full_131072/`.
- Dataset labels: range `[0, 4]`, mean about `2.5452`, std about `0.7801`.
- Training metrics: best epoch 12, best validation RMSE `0.3514548838`, validation MAE `0.2388104796`, validation max error `2.3319456577`; final epoch 200 overfit to validation RMSE about `0.4135`.
- Interpretation: the model is useful as a soft-loss candidate generator, but final candidate trust still requires exact validation with `validate_de_results.py`.

Gradient surrogate optimizer implementation:

- Added `surrogate_bid_optimizer.py` to optimize the 12 adjustable units with multi-start Adam over the trained neural surrogate.
- Objective: fixed Sobol competitor environments, minimize mean predicted surrogate soft-loss.
- Output format: `best_result.json`, `candidate_rank_*.json`, `best_result.txt`, and `optimizer_summary.json`.
- Updated `validate_de_results.py` to read and deduplicate `candidate_rank_*.json` files.
- Validation command convention for server runs uses `--workers 60` rather than `--workers -1`.
- Local checks after implementation:
  - `python3 -m py_compile surrogate_bid_optimizer.py validate_de_results.py`
  - `uv run --no-sync ruff check surrogate_bid_optimizer.py validate_de_results.py tests/test_surrogate_bid_optimizer.py tests/test_validate_de_results.py`
  - `uv run --no-sync pytest -q`
  - Result: all passed; two optional PyTorch smoke tests skipped locally because torch is not installed.
- Server first run exposed a PyTorch autograd issue in `surrogate_bid_optimizer.py`: each optimization step computed adjustable bids once and then called `backward()` once per environment chunk, reusing a freed graph on the second chunk.
- Fixed the optimizer to recompute the sigmoid-parameterized adjustable bids inside each environment chunk before `backward()`, preserving gradient accumulation without `retain_graph=True`.

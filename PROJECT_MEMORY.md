# Project Memory

Last updated: 2026-06-23

## Source of Truth

- `qingbiao.md` is the official rule statement.
- Do not modify `qingbiao.md` unless the user explicitly requests it.
- Use `PROJECT_BOOTSTRAP_PROMPT.md` as the new-chat startup prompt.

## Current Objective

- Current target bidder: Unit 5, represented by `X5`.
- Goal: maximize the winning probability of `X5`.
- The official rules note that direct winning probability may be difficult to optimize, so smoother surrogate objectives may be useful.

## Decision Variables

- All discount rates satisfy `X_i in [10, 30]`.
- Adjustable variables:
  `X3, X5, X6, X7, X9, X10, X11, X13, X16, X17, X19, X20`.
- Higher `X_i` means a lower quoted price.
- Lower `X_i` means a higher quoted price.

## Planned Routes

- Route 1: Differential Evolution optimization with a soft objective, followed by separate exact/probability validation.
- Route 2: neural-network simulation or surrogate modeling, verified against the rule engine.

## Current Architecture

- `forward_bidding_calculator.py` implements a logged forward calculator for the five official stages and exact enumeration for a fixed full bid vector.
- `de_softmax_optimizer.py` implements the first Differential Evolution optimizer for `X5`.
  - Optimizes the 12 adjustable variables only.
  - Treats non-adjustable units as a Sobol/CRN environment sampled from their known ranges.
  - Default CRN sample count is now 8192 for formal candidate generation.
  - Uses hard stage flow to locate where `X5` fails, then applies a soft failure-margin loss.
  - Uses a final-stage softmax matching the current rule: prioritize finalists with `X_i > K`, then fall back to the smallest `|X_i-K|` if none exist.
  - Saves checkpoints every `checkpoint_every` iterations; default is every 50 iterations.
- `validate_de_results.py` validates optimizer checkpoints/results against the true current rules.
  - It reads `best_result.json`, `checkpoint_iter_*.json`, and neural surrogate `candidate_rank_*.json` files from an optimizer result directory.
  - It deduplicates candidates by rounded adjustable bids before validation.
  - It samples the 8 non-adjustable units with Sobol and exactly enumerates the 324 discrete rule scenarios per environment sample.
  - Defaults: first pass `samples=32768`, `top=20`; automatic refinement `refine_top=5`, `refine_samples=65536`.
  - Outputs `validation_summary.csv`, `validation_results.json`, and `validation_report.txt`.
- `neural_surrogate.py` trains and uses a Residual MLP surrogate for the DE soft-loss objective.
  - Input is the full bid vector `X1..X20`.
  - Label is the average DE soft loss over 108 combinations of `Q`, `B2`, excluded-lowest-count, and `target_n`; `K2` is averaged inside `evaluate_scenario_loss`.
  - Full run artifact currently expected at `surrogate_runs/full_131072/model`.
- `surrogate_bid_optimizer.py` uses a trained neural surrogate for multi-start Adam gradient optimization of the 12 adjustable units.
  - Competitor/non-adjustable bids are Sobol-sampled over configured ranges.
  - Objective is mean predicted surrogate soft-loss, not true winning probability.
  - Outputs `best_result.json`, `candidate_rank_*.json`, `best_result.txt`, and `optimizer_summary.json` for downstream exact validation.
- `tests/test_de_softmax_optimizer.py` contains smoke and mapping tests for the optimizer.
- `tests/test_validate_de_results.py` contains candidate loading, mapping, count, refinement-selection, and evaluator-consistency tests for the validator.
- `tests/test_surrogate_bid_optimizer.py` contains range validation, Sobol sampling, bid composition, payload, and optional PyTorch smoke tests for the gradient surrogate optimizer.
- Local `uv` environment exists at `.venv`; use `uv --cache-dir .uv-cache ...` locally because the sandbox blocks `~/.cache/uv`.
- Quality gate for all code changes: add/maintain pytest unit tests and pass `uv run ruff check .`, `uv run mypy`, and `uv run pytest`.
- Long-term project files now include:
  - `PROJECT_BOOTSTRAP_PROMPT.md` for new-chat setup.
  - `PROJECT_MEMORY.md` for stable project state.
  - `DECISIONS.md` for durable modeling and engineering decisions.
  - `EXPERIMENT_LOG.md` for optimization and surrogate runs.
  - `TODO.md` for current next actions.
  - `README.md` for a short repository overview.

## Implementation Notes

- The random choices in the current rules are discrete, so exact enumeration is used inside each sampled continuous environment.
- Non-adjustable units are continuous uncertain variables; validation uses Sobol sampling over their 8-dimensional range and reports approximate standard error across environment samples.
- Rule-critical rounding should use ordinary round-half-up behavior unless later confirmed otherwise.
- Tie-breaking and the distinction between quoted price and discount rate are high-risk parts of the implementation and need tests.
- The optimizer's objective is not true winning probability. Low objective values should be treated as candidate-generation signals, then checked by `validate_de_results.py`.
- The neural surrogate gradient optimizer should be treated as a faster candidate generator only; final trust still comes from exact validation.
- The optimizer was committed and pushed to GitHub as `48f0953 Add differential evolution softmax optimizer`.
- On 2026-06-22, `qingbiao.md` changed the final winner stage so that a winner is still selected when no finalist has `X_i > K`; existing generated DE/validation artifacts from the older final-stage rule are stale until rerun.
- Current full neural surrogate training metrics from `surrogate_runs/full_131072/model/metrics.json`: best epoch 12, best validation RMSE about 0.3515, validation MAE about 0.2388, label range `[0, 4]`, and rough validation `R^2` about 0.794. Later epochs overfit, but `model.pt` stores the best checkpoint.

## Open Questions

- Run `surrogate_bid_optimizer.py` on the Linux GPU/server artifact and record the Top K surrogate candidates.
- Run high-precision validation on `surrogate_runs/full_131072/optimizer` using `--workers 60` and compare surrogate ranking against true `X5` winning probability.
- Rerun DE optimization with `samples=8192` under the 2026-06-22 final-stage fallback rule only if a DE baseline is still needed.
- Add broader tests for the official rule engine and tie-breaking before relying on long optimization runs.

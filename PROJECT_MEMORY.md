# Project Memory

Last updated: 2026-06-18

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
  - Uses a one-sided softmax in the final winning stage that respects the rule `X_i > K`.
  - Saves checkpoints every `checkpoint_every` iterations; default is every 50 iterations.
- `validate_de_results.py` validates optimizer checkpoints/results against the true current rules.
  - It reads `best_result.json` and `checkpoint_iter_*.json` from an optimizer result directory.
  - It samples the 8 non-adjustable units with Sobol and exactly enumerates the 324 discrete rule scenarios per environment sample.
  - Defaults: first pass `samples=32768`, `top=20`; automatic refinement `refine_top=5`, `refine_samples=65536`.
  - Outputs `validation_summary.csv`, `validation_results.json`, and `validation_report.txt`.
- `tests/test_de_softmax_optimizer.py` contains smoke and mapping tests for the optimizer.
- `tests/test_validate_de_results.py` contains candidate loading, mapping, count, refinement-selection, and evaluator-consistency tests for the validator.
- Local `uv` environment exists at `.venv`; use `uv --cache-dir .uv-cache ...` locally because the sandbox blocks `~/.cache/uv`.
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
- The optimizer was committed and pushed to GitHub as `48f0953 Add differential evolution softmax optimizer`.

## Open Questions

- Run a first full DE optimization with `samples=8192` and record results.
- Run high-precision validation on the full optimizer output directory and compare surrogate objective ranking against true `X5` winning probability.
- Add broader tests for the official rule engine and tie-breaking before relying on long optimization runs.

# Decisions

This file records durable modeling and engineering decisions. Keep entries concise and append-only unless a later decision supersedes an earlier one.

## 2026-06-18

- Treat `qingbiao.md` as the immutable source of truth for the tender-cleaning and winner-selection rules.
- Treat Unit 5 (`X5`) as the target bidder unless the user explicitly changes the objective.
- Interpret `X_i` as discount rates, not quoted prices. Larger `X_i` means a lower quoted price.
- Use ordinary round-half-up behavior for rule-critical rounding unless a different convention is confirmed.
- Prefer exact enumeration over Monte Carlo for fixed bid-vector evaluation while all random events remain discrete and small.
- Keep non-adjustable bidder discounts fixed during optimization only after documenting the chosen baseline values.
- Write rule-engine tests before trusting optimization or surrogate-model results.
- First Differential Evolution optimizer optimizes only `X3, X5, X6, X7, X9, X10, X11, X13, X16, X17, X19, X20`.
- During optimizer candidate generation, non-adjustable bidders are modeled as Sobol/CRN random environments drawn from their known ranges.
- Superseded on 2026-06-22: optimizer objective is a surrogate where hard official stages locate the first `X5` failure, then a soft failure-margin loss is applied; the original final-stage softmax only respected `X_i > K`.
- Optimizer checkpoints are saved periodically because surrogate objective value is not guaranteed to rank true winning probability perfectly.
- True winning-probability validation should be implemented separately and used to compare optimizer checkpoints/results.
- Local development environment uses `uv` with project-local cache: `uv --cache-dir .uv-cache ...`.
- Increase the optimizer default CRN sample count from 4096 to 8192 for more stable candidate generation over the 8 non-adjustable continuous environment variables.
- Validate optimizer outputs with a two-stage Sobol scheme: first pass 32768 samples over the 8 non-adjustable units for the top 20 surrogate candidates, then refine the top 5 true-probability candidates with 65536 samples.
- For validation, exactly enumerate all 324 discrete rule scenarios per continuous environment sample and estimate uncertainty across continuous environment samples.

## 2026-06-22

- Supersede the earlier final-stage one-sided assumption: after computing `K = K1 + K2`, first choose the closest bidder among finalists with `X_i > K`; if no finalist has `X_i > K`, choose the finalist with the smallest `|X_i - K|`.
- The exact forward calculator, DE validator, and DE surrogate objective must all use this fallback rule. Existing generated optimizer/validation artifacts from before this date should be treated as stale until regenerated.

## 2026-06-23

- First neural surrogate targets the current DE soft-loss function, not true winning probability.
- Neural surrogate input is the complete 20-dimensional bid vector `X1..X20`, sampled over the broad domain `[10, 30]` for every unit.
- A fixed full bid vector's label is the average `evaluate_scenario_loss` value over the 108 combinations of `Q`, `B2`, excluded-lowest-count, and `target_n`; `K2` remains averaged inside the existing DE soft-loss evaluator.
- Use a Residual MLP as the default neural surrogate architecture because it is lightweight for 20-dimensional tabular regression while supporting deeper nonlinear interactions and future gradient-based bid search.
- Local development only needs smoke-scale runs; full training is expected to run on the Linux GPU server after syncing PyTorch dependencies.
- Use multi-start Adam gradient optimization as the first neural-surrogate candidate generator, because the trained PyTorch surrogate is differentiable and should be faster than Differential Evolution for this objective.
- The neural-surrogate optimizer outputs candidates only; exact `X5` winning-probability validation remains a separate step through `validate_de_results.py`.
- Use explicit `--workers 60` for server validation commands rather than `--workers -1`.

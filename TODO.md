# TODO

## Next

- Run a first real high-precision DE optimization with default `samples=8192` and record checkpoint/result conclusions in `EXPERIMENT_LOG.md`.
- Run `validate_de_results.py` on the full optimizer output directory and record the true-probability ranking.
- Compare surrogate objective ranking against validated `X5` winning probability; decide whether objective tuning is needed.
- Add tests for discount-rate versus quoted-price ordering.
- Add tests for round-half-up behavior and tie-breaking.
- Add broader tests comparing the optimizer's fast evaluator against `forward_bidding_calculator.py` on selected fixed scenarios.

## Later

- Add experiment logging helpers for optimizer runs.
- Design and validate a surrogate objective or neural-network surrogate.
- Compare surrogate-generated candidates against exact winning probability.

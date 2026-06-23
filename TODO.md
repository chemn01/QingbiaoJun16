# TODO

## Next

- Pull the latest commit on the Linux server and run `surrogate_bid_optimizer.py` against `surrogate_runs/full_131072/model`.
- Validate the generated surrogate candidates with `validate_de_results.py` using `--workers 60`, then record the true-probability ranking.
- Rerun a real high-precision DE optimization with default `samples=8192` under the 2026-06-22 final-stage fallback rule, then record checkpoint/result conclusions in `EXPERIMENT_LOG.md`.
- Compare neural surrogate objective ranking against validated `X5` winning probability; decide whether objective tuning or a conservative objective is needed.
- Add tests for discount-rate versus quoted-price ordering.
- Add tests for round-half-up behavior and tie-breaking.
- Add broader tests comparing the optimizer's fast evaluator against `forward_bidding_calculator.py` on selected fixed scenarios.

## Later

- Add experiment logging helpers for optimizer runs.
- Compare surrogate-generated candidates against exact winning probability.

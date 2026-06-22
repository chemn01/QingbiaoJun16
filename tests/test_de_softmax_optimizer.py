from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pytest

import de_softmax_optimizer as optimizer


def test_resolve_worker_count_caps_default_workers_on_windows(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(optimizer.os, "name", "nt")
    monkeypatch.setattr(optimizer.mp, "cpu_count", lambda: 128)

    actual_workers = optimizer.resolve_worker_count(-1)

    assert actual_workers == optimizer.WINDOWS_MAX_POOL_WORKERS
    assert "已从 90 降为 61" in capsys.readouterr().out


def test_resolve_worker_count_caps_explicit_workers_on_windows(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(optimizer.os, "name", "nt")

    actual_workers = optimizer.resolve_worker_count(90)

    assert actual_workers == optimizer.WINDOWS_MAX_POOL_WORKERS
    assert "已从 90 降为 61" in capsys.readouterr().out


def test_resolve_worker_count_keeps_default_cap_on_non_windows(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(optimizer.os, "name", "posix")
    monkeypatch.setattr(optimizer.mp, "cpu_count", lambda: 128)

    actual_workers = optimizer.resolve_worker_count(-1)

    assert actual_workers == optimizer.DEFAULT_MAX_WORKERS
    assert capsys.readouterr().out == ""


def test_resolve_worker_count_rejects_non_positive_workers() -> None:
    with pytest.raises(ValueError, match="--workers"):
        optimizer.resolve_worker_count(0)


class DeSoftmaxOptimizerTests(unittest.TestCase):
    def test_adjustable_mapping_writes_expected_units(self) -> None:
        decision_vars = np.arange(len(optimizer.ADJUSTABLE_UNITS), dtype=float) + 100.0
        env_vars = np.arange(len(optimizer.NON_ADJUSTABLE_UNITS), dtype=float) + 200.0

        bids = optimizer.build_bid_vector(decision_vars, env_vars)

        for index, unit in enumerate(optimizer.ADJUSTABLE_UNITS):
            self.assertEqual(bids[unit], decision_vars[index])
        for index, unit in enumerate(optimizer.NON_ADJUSTABLE_UNITS):
            self.assertEqual(bids[unit], env_vars[index])

    def test_bid_summaries_round_to_two_decimals(self) -> None:
        decision_vars = np.full(len(optimizer.ADJUSTABLE_UNITS), 18.766)

        summary = optimizer.optimized_bid_summary(decision_vars)

        self.assertEqual(summary["X3"], 18.77)

    def test_final_stage_softmax_penalizes_units_not_above_k_when_priority_exists(self) -> None:
        bids = np.zeros(optimizer.NUM_UNITS + 1, dtype=float)
        bids[5] = 18.9
        bids[6] = 19.1
        bids[7] = 19.4
        finalists = [5, 6, 7]

        p_not_eligible = optimizer.final_stage_softmax_probability(
            finalists,
            bids,
            final_k=19.0,
            temperature=0.1,
            invalid_cost=10.0,
        )
        self.assertLess(p_not_eligible, 1e-6)

        bids[5] = 19.05
        p_eligible = optimizer.final_stage_softmax_probability(
            finalists,
            bids,
            final_k=19.0,
            temperature=0.1,
            invalid_cost=10.0,
        )
        self.assertGreater(p_eligible, 0.5)

    def test_final_stage_falls_back_to_absolute_distance_when_no_unit_above_k(self) -> None:
        bids = np.zeros(optimizer.NUM_UNITS + 1, dtype=float)
        bids[5] = 18.9
        bids[6] = 18.7
        bids[7] = 18.1
        finalists = [5, 6, 7]

        self.assertEqual(optimizer.hard_winner(finalists, bids, final_k=19.0), 5)

        p_fallback = optimizer.final_stage_softmax_probability(
            finalists,
            bids,
            final_k=19.0,
            temperature=0.1,
            invalid_cost=10.0,
        )
        self.assertGreater(p_fallback, 0.8)

    def test_smoke_run_saves_checkpoint_and_final_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            payload = optimizer.optimize_with_differential_evolution(
                n_samples=16,
                maxiter=1,
                popsize=3,
                workers=1,
                seed=7,
                checkpoint_every=1,
                output_dir=Path(tmpdir),
                polish=False,
                disp=False,
            )

            checkpoint_path = Path(tmpdir) / "checkpoint_iter_0001.json"
            best_path = Path(tmpdir) / "best_result.json"

            self.assertTrue(checkpoint_path.exists())
            self.assertTrue(best_path.exists())
            self.assertIn("best_adjustable_bids", payload)

            with best_path.open("r", encoding="utf-8") as file:
                saved = json.load(file)
            self.assertEqual(saved["target_unit"], "X5")
            self.assertEqual(len(saved["best_adjustable_bids"]), len(optimizer.ADJUSTABLE_UNITS))


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import de_softmax_optimizer as optimizer


class DeSoftmaxOptimizerTests(unittest.TestCase):
    def test_adjustable_mapping_writes_expected_units(self) -> None:
        decision_vars = np.arange(len(optimizer.ADJUSTABLE_UNITS), dtype=float) + 100.0
        env_vars = np.arange(len(optimizer.NON_ADJUSTABLE_UNITS), dtype=float) + 200.0

        bids = optimizer.build_bid_vector(decision_vars, env_vars)

        for index, unit in enumerate(optimizer.ADJUSTABLE_UNITS):
            self.assertEqual(bids[unit], decision_vars[index])
        for index, unit in enumerate(optimizer.NON_ADJUSTABLE_UNITS):
            self.assertEqual(bids[unit], env_vars[index])

    def test_one_sided_softmax_penalizes_units_not_above_k(self) -> None:
        bids = np.zeros(optimizer.NUM_UNITS + 1, dtype=float)
        bids[5] = 18.9
        bids[6] = 19.1
        bids[7] = 19.4
        finalists = [5, 6, 7]

        p_not_eligible = optimizer.one_sided_softmax_probability(
            finalists,
            bids,
            final_k=19.0,
            temperature=0.1,
            invalid_cost=10.0,
        )
        self.assertLess(p_not_eligible, 1e-6)

        bids[5] = 19.05
        p_eligible = optimizer.one_sided_softmax_probability(
            finalists,
            bids,
            final_k=19.0,
            temperature=0.1,
            invalid_cost=10.0,
        )
        self.assertGreater(p_eligible, 0.5)

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

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import de_softmax_optimizer as optimizer
import forward_bidding_calculator as forward
import validate_de_results as validator


def adjustable_payload(objective: float, offset: float = 0.0) -> dict[str, object]:
    bids = {}
    for unit in optimizer.ADJUSTABLE_UNITS:
        lower, upper = optimizer.BID_BOUNDS[unit]
        bids[f"X{unit}"] = (lower + upper) / 2 + offset
    return {
        "iteration": int(objective * 1000),
        "objective_value": objective,
        "best_adjustable_bids": bids,
    }


class ValidateDeResultsTests(unittest.TestCase):
    def test_load_candidates_and_top_selection(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            result_dir = Path(tmpdir)
            (result_dir / "best_result.json").write_text(
                json.dumps(adjustable_payload(0.30)),
                encoding="utf-8",
            )
            (result_dir / "checkpoint_iter_0001.json").write_text(
                json.dumps(adjustable_payload(0.10, offset=0.01)),
                encoding="utf-8",
            )
            (result_dir / "checkpoint_iter_0002.json").write_text(
                json.dumps(adjustable_payload(0.20, offset=0.02)),
                encoding="utf-8",
            )

            candidates = validator.load_candidates(result_dir)
            self.assertEqual([candidate.objective_value for candidate in candidates], [0.10, 0.20, 0.30])

            top_one = validator.select_top_candidates(candidates, 1)
            self.assertEqual(len(top_one), 1)
            self.assertEqual(top_one[0].objective_value, 0.10)

            all_candidates = validator.select_top_candidates(candidates, 0)
            self.assertEqual(len(all_candidates), 3)

    def test_build_bid_vector_mapping(self) -> None:
        decision_vars = np.arange(len(optimizer.ADJUSTABLE_UNITS), dtype=float) + 100.0
        env_vars = np.arange(len(optimizer.NON_ADJUSTABLE_UNITS), dtype=float) + 200.0
        bids = validator.build_bid_vector(decision_vars, env_vars)

        for index, unit in enumerate(optimizer.ADJUSTABLE_UNITS):
            self.assertEqual(bids[unit], decision_vars[index])
        for index, unit in enumerate(optimizer.NON_ADJUSTABLE_UNITS):
            self.assertEqual(bids[unit], env_vars[index])

    def test_fast_evaluator_matches_logged_forward_calculator(self) -> None:
        raw_bids = [
            (lower + upper) / 2
            for unit, (lower, upper) in optimizer.BID_BOUNDS.items()
        ]
        forward_bids = forward.validate_bids(raw_bids)
        fast_bids = np.asarray([0.0] + raw_bids, dtype=float)

        scenarios = [
            validator.RuleScenario(q=15, b2=0.5, exclude_lowest_price_count=1, target_n=3, k2=0.0),
            validator.RuleScenario(q=18, b2=1.0, exclude_lowest_price_count=2, target_n=4, k2=0.25),
            validator.RuleScenario(q=20, b2=1.5, exclude_lowest_price_count=1, target_n=5, k2=0.5),
        ]

        for scenario in scenarios:
            fast = validator.evaluate_rule_scenario(fast_bids, scenario)
            logged = forward.simulate_and_log(
                forward_bids,
                forward.Scenario(
                    q=scenario.q,
                    b2=scenario.b2,
                    exclude_lowest_price_count=scenario.exclude_lowest_price_count,
                    target_n=scenario.target_n,
                    k2=scenario.k2,
                ),
            )
            self.assertEqual(fast.winner, logged.winner)
            self.assertEqual(fast.target_status, logged.target_status)

    def test_count_helpers_and_refine_selection(self) -> None:
        self.assertEqual(validator.DISCRETE_SCENARIO_COUNT, 324)
        self.assertEqual(
            validator.total_scenarios_for_samples(32768),
            32768 * 324,
        )
        self.assertEqual(
            validator.total_scenarios_for_samples(65536),
            65536 * 324,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            candidates = [
                validator.Candidate(
                    name=f"c{index}",
                    source_path=Path(tmpdir) / f"c{index}.json",
                    iteration=index,
                    objective_value=float(index),
                    decision_vars=np.zeros(len(optimizer.ADJUSTABLE_UNITS)),
                    adjustable_bids={},
                    payload={},
                )
                for index in range(3)
            ]
            results = [
                validator.ValidationResult(
                    phase="initial",
                    candidate_name="c0",
                    source_path=str(candidates[0].source_path),
                    iteration=0,
                    objective_value=0.0,
                    samples=10,
                    total_scenarios=3240,
                    target_wins=10,
                    target_win_probability=0.1,
                    standard_error=0.01,
                    ci95_half_width=0.0196,
                    winner_distribution={},
                    status_distribution={},
                    marginal_probabilities={},
                    adjustable_bids={},
                ),
                validator.ValidationResult(
                    phase="initial",
                    candidate_name="c1",
                    source_path=str(candidates[1].source_path),
                    iteration=1,
                    objective_value=1.0,
                    samples=10,
                    total_scenarios=3240,
                    target_wins=20,
                    target_win_probability=0.2,
                    standard_error=0.01,
                    ci95_half_width=0.0196,
                    winner_distribution={},
                    status_distribution={},
                    marginal_probabilities={},
                    adjustable_bids={},
                ),
            ]

            self.assertEqual(
                [candidate.name for candidate in validator.choose_refinement_candidates(candidates, results, 1)],
                ["c1"],
            )
            self.assertEqual(
                validator.choose_refinement_candidates(candidates, results, 0),
                [],
            )


if __name__ == "__main__":
    unittest.main()

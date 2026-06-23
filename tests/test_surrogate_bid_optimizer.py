from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import de_softmax_optimizer as optimizer
import neural_surrogate
import surrogate_bid_optimizer as bid_optimizer


def test_resolve_bid_ranges_uses_defaults_and_json_overrides(tmp_path: Path) -> None:
    ranges_path = tmp_path / "ranges.json"
    ranges_path.write_text(
        json.dumps(
            {
                "X1": [19.25, 19.75],
                "X5": [14.0, 18.0],
            }
        ),
        encoding="utf-8",
    )

    ranges, overrides = bid_optimizer.resolve_bid_ranges(ranges_path)

    assert ranges[1] == (19.25, 19.75)
    assert ranges[5] == (14.0, 18.0)
    assert ranges[2] == optimizer.BID_BOUNDS[2]
    assert overrides == {1: (19.25, 19.75), 5: (14.0, 18.0)}


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        ({"Y1": [10, 11]}, "Unknown unit"),
        ({"X21": [10, 11]}, "Unknown unit"),
        ({"X1": [20, 19]}, "lower bound"),
        ({"X1": [9, 19]}, "outside surrogate training domain"),
        ({"X1": [19, float("inf")]}, "finite"),
    ],
)
def test_resolve_bid_ranges_rejects_invalid_overrides(
    tmp_path: Path,
    payload: dict[str, object],
    match: str,
) -> None:
    ranges_path = tmp_path / "ranges.json"
    ranges_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        bid_optimizer.resolve_bid_ranges(ranges_path)


def test_generate_env_matrix_is_deterministic_and_in_range() -> None:
    ranges, _ = bid_optimizer.resolve_bid_ranges()

    first = bid_optimizer.generate_env_matrix(samples=8, seed=7, ranges=ranges)
    second = bid_optimizer.generate_env_matrix(samples=8, seed=7, ranges=ranges)

    assert first.shape == (8, len(optimizer.NON_ADJUSTABLE_UNITS))
    np.testing.assert_array_equal(first, second)
    for col, unit in enumerate(optimizer.NON_ADJUSTABLE_UNITS):
        lower, upper = ranges[unit]
        assert float(first[:, col].min()) >= lower
        assert float(first[:, col].max()) <= upper


def test_compose_full_bid_matrix_uses_optimizer_unit_order() -> None:
    adjustable = np.arange(len(optimizer.ADJUSTABLE_UNITS), dtype=np.float32) + 100.0
    env = np.arange(len(optimizer.NON_ADJUSTABLE_UNITS), dtype=np.float32) + 200.0

    full = bid_optimizer.compose_full_bid_matrix(adjustable, env)

    assert full.shape == (optimizer.NUM_UNITS,)
    for index, unit in enumerate(optimizer.ADJUSTABLE_UNITS):
        assert full[unit - 1] == adjustable[index]
    for index, unit in enumerate(optimizer.NON_ADJUSTABLE_UNITS):
        assert full[unit - 1] == env[index]


def test_build_candidate_payload_has_validator_required_fields() -> None:
    ranges, overrides = bid_optimizer.resolve_bid_ranges()
    env_matrix = bid_optimizer.generate_env_matrix(samples=4, seed=5, ranges=ranges)
    adjustable, _ = bid_optimizer.bounds_for_units(optimizer.ADJUSTABLE_UNITS, ranges)
    candidate = bid_optimizer.RankedCandidate(
        rank=1,
        objective_value=0.123,
        adjustable_bids=adjustable,
    )

    payload = bid_optimizer.build_candidate_payload(
        candidate=candidate,
        env_matrix=env_matrix,
        ranges=ranges,
        overrides=overrides,
        config={"samples": 4},
        elapsed_seconds=1.5,
        model_dir=Path("model"),
    )

    assert payload["objective_value"] == pytest.approx(0.123)
    assert payload["target_unit"] == "X5"
    assert set(payload["best_adjustable_bids"]) == {f"X{unit}" for unit in optimizer.ADJUSTABLE_UNITS}


def test_optimize_with_surrogate_smoke_saves_candidate_files(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    data_path = tmp_path / "data.npz"
    model_dir = tmp_path / "model"
    output_dir = tmp_path / "optimizer"

    neural_surrogate.generate_dataset(
        samples=32,
        seed=11,
        output_path=data_path,
        workers=1,
        use_sobol=True,
    )
    neural_surrogate.train_model(
        data_path=data_path,
        output_dir=model_dir,
        epochs=1,
        batch_size=8,
        width=32,
        blocks=1,
        head_width=16,
        device="cpu",
    )

    payloads = bid_optimizer.optimize_with_surrogate(
        model_dir=model_dir,
        output_dir=output_dir,
        samples=4,
        starts=4,
        steps=2,
        top_k=2,
        seed=3,
        device="cpu",
        start_batch_size=2,
        env_batch_size=2,
        learning_rate=0.01,
    )

    assert payloads
    assert (output_dir / "best_result.json").exists()
    assert (output_dir / "candidate_rank_001.json").exists()
    assert (output_dir / "optimizer_summary.json").exists()

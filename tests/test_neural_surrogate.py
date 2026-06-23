from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

import de_softmax_optimizer as optimizer
import neural_surrogate as surrogate


def test_split_full_bids_uses_optimizer_unit_order() -> None:
    full_bids = np.arange(1, optimizer.NUM_UNITS + 1, dtype=float)

    decision_vars, env_vars = surrogate.split_full_bids(full_bids)

    expected_decision = np.asarray(optimizer.ADJUSTABLE_UNITS, dtype=float)
    expected_env = np.asarray(optimizer.NON_ADJUSTABLE_UNITS, dtype=float)
    np.testing.assert_array_equal(decision_vars, expected_decision)
    np.testing.assert_array_equal(env_vars, expected_env)


def test_split_full_bids_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match="shape"):
        surrogate.split_full_bids(np.ones(19, dtype=float))


def test_full_bid_soft_loss_is_deterministic_and_finite() -> None:
    full_bids = np.linspace(10.5, 29.5, optimizer.NUM_UNITS, dtype=float)

    first = surrogate.full_bid_soft_loss(full_bids)
    second = surrogate.full_bid_soft_loss(full_bids)

    assert math.isfinite(first)
    assert first == pytest.approx(second)


def test_generate_dataset_writes_expected_npz(tmp_path: Path) -> None:
    output_path = tmp_path / "data.npz"

    metadata = surrogate.generate_dataset(
        samples=8,
        seed=7,
        output_path=output_path,
        workers=1,
        use_sobol=True,
    )
    bids, labels, loaded_metadata = surrogate.load_dataset(output_path)

    assert output_path.exists()
    assert bids.shape == (8, optimizer.NUM_UNITS)
    assert labels.shape == (8,)
    assert float(bids.min()) >= surrogate.FULL_BID_LOWER
    assert float(bids.max()) <= surrogate.FULL_BID_UPPER
    assert loaded_metadata["samples"] == 8
    assert loaded_metadata["scenario_count"] == len(surrogate.SCENARIOS)
    assert metadata["label_summary"]["min"] == pytest.approx(float(labels.min()))


def test_parse_bids_payload_accepts_dict_and_matrix() -> None:
    dict_payload = {f"X{unit}": float(unit) for unit in range(1, optimizer.NUM_UNITS + 1)}
    matrix_payload = [[float(unit) for unit in range(1, optimizer.NUM_UNITS + 1)]]

    dict_bids = surrogate.parse_bids_payload(dict_payload)
    matrix_bids = surrogate.parse_bids_payload(matrix_payload)

    assert dict_bids.shape == (1, optimizer.NUM_UNITS)
    np.testing.assert_array_equal(dict_bids, matrix_bids)


def test_train_model_smoke_saves_artifacts(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    data_path = tmp_path / "data.npz"
    output_dir = tmp_path / "model"
    surrogate.generate_dataset(
        samples=32,
        seed=11,
        output_path=data_path,
        workers=1,
        use_sobol=True,
    )

    metrics = surrogate.train_model(
        data_path=data_path,
        output_dir=output_dir,
        epochs=1,
        batch_size=8,
        width=32,
        blocks=1,
        head_width=16,
        device="cpu",
    )

    assert (output_dir / "model.pt").exists()
    assert (output_dir / "metrics.json").exists()
    assert (output_dir / "normalization.json").exists()
    assert metrics["best_epoch"] == 1

    loaded_metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert loaded_metrics["validation_samples"] > 0

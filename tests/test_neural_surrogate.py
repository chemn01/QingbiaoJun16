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


def test_stratified_quotas_match_planned_262144_counts() -> None:
    quotas = surrogate.stratified_quotas(262144)

    assert quotas == {
        surrogate.STRATUM_GLOBAL_UNIFORM: 183502,
        surrogate.STRATUM_ELITE_LOSS: 26214,
        surrogate.STRATUM_VERY_LOW_LOSS: 26214,
        surrogate.STRATUM_LOW_LOSS: 26214,
    }


def test_soft_loss_stratum_uses_configurable_thresholds() -> None:
    assert surrogate.soft_loss_stratum(0.75, 0.75, 1.25, 1.75) == surrogate.STRATUM_ELITE_LOSS
    assert surrogate.soft_loss_stratum(1.25, 0.75, 1.25, 1.75) == surrogate.STRATUM_VERY_LOW_LOSS
    assert surrogate.soft_loss_stratum(1.75, 0.75, 1.25, 1.75) == surrogate.STRATUM_LOW_LOSS
    assert surrogate.soft_loss_stratum(1.76, 0.75, 1.25, 1.75) is None

    with pytest.raises(ValueError, match="elite-threshold"):
        surrogate.validate_stratified_thresholds(1.25, 0.75, 1.75)


def test_generate_stratified_dataset_writes_extra_stratum_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_path = tmp_path / "stratified.npz"

    def fake_generate_labels(
        bids: np.ndarray,
        options: optimizer.ObjectiveOptions,
        workers: int = 1,
        chunksize: int = 64,
    ) -> np.ndarray:
        del options, workers, chunksize
        pattern = np.asarray([0.5, 1.5, 2.5, 3.5], dtype=np.float32)
        return np.resize(pattern, bids.shape[0]).astype(np.float32)

    monkeypatch.setattr(surrogate, "generate_labels", fake_generate_labels)

    metadata = surrogate.generate_stratified_dataset(
        samples=40,
        seed=9,
        output_path=output_path,
        workers=1,
        pool_batch_size=8,
        max_candidate_multiplier=3,
        elite_threshold=1.0,
        very_low_threshold=2.0,
        low_threshold=3.0,
    )
    bids, labels, loaded_metadata = surrogate.load_dataset(output_path)

    with np.load(output_path, allow_pickle=False) as data:
        stratum_ids = np.asarray(data["stratum_id"]).astype(str)

    assert bids.shape == (40, optimizer.NUM_UNITS)
    assert labels.shape == (40,)
    assert stratum_ids.shape == (40,)
    assert float(bids.min()) >= surrogate.FULL_BID_LOWER
    assert float(bids.max()) <= surrogate.FULL_BID_UPPER
    assert loaded_metadata["dataset_type"] == "global_softloss_stratified"
    assert metadata["stratum_counts"] == {
        surrogate.STRATUM_ELITE_LOSS: 4,
        surrogate.STRATUM_GLOBAL_UNIFORM: 28,
        surrogate.STRATUM_LOW_LOSS: 4,
        surrogate.STRATUM_VERY_LOW_LOSS: 4,
    }
    assert dict(zip(*np.unique(stratum_ids, return_counts=True))) == metadata["stratum_counts"]


def test_generate_stratified_dataset_is_seed_stable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_generate_labels(
        bids: np.ndarray,
        options: optimizer.ObjectiveOptions,
        workers: int = 1,
        chunksize: int = 64,
    ) -> np.ndarray:
        del options, workers, chunksize
        pattern = np.asarray([0.5, 1.5, 2.5, 3.5], dtype=np.float32)
        return np.resize(pattern, bids.shape[0]).astype(np.float32)

    monkeypatch.setattr(surrogate, "generate_labels", fake_generate_labels)
    first_path = tmp_path / "first.npz"
    second_path = tmp_path / "second.npz"

    first_metadata = surrogate.generate_stratified_dataset(
        samples=40,
        seed=13,
        output_path=first_path,
        workers=1,
        pool_batch_size=8,
        max_candidate_multiplier=3,
        elite_threshold=1.0,
        very_low_threshold=2.0,
        low_threshold=3.0,
    )
    second_metadata = surrogate.generate_stratified_dataset(
        samples=40,
        seed=13,
        output_path=second_path,
        workers=1,
        pool_batch_size=8,
        max_candidate_multiplier=3,
        elite_threshold=1.0,
        very_low_threshold=2.0,
        low_threshold=3.0,
    )

    with np.load(first_path, allow_pickle=False) as first, np.load(second_path, allow_pickle=False) as second:
        np.testing.assert_array_equal(first["bids"], second["bids"])
        np.testing.assert_array_equal(first["labels"], second["labels"])
        np.testing.assert_array_equal(first["stratum_id"], second["stratum_id"])
    assert first_metadata["stratum_counts"] == second_metadata["stratum_counts"]


def test_generate_stratified_dataset_fails_when_buckets_do_not_fill(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def high_labels(
        bids: np.ndarray,
        options: optimizer.ObjectiveOptions,
        workers: int = 1,
        chunksize: int = 64,
    ) -> np.ndarray:
        del options, workers, chunksize
        return np.full((bids.shape[0],), 4.0, dtype=np.float32)

    monkeypatch.setattr(surrogate, "generate_labels", high_labels)

    with pytest.raises(ValueError, match="Could not fill all soft-loss buckets"):
        surrogate.generate_stratified_dataset(
            samples=20,
            seed=5,
            output_path=tmp_path / "failed.npz",
            workers=1,
            pool_batch_size=4,
            max_candidate_multiplier=1,
            elite_threshold=1.0,
            very_low_threshold=2.0,
            low_threshold=3.0,
        )


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

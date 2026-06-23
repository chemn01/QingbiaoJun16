"""
Neural surrogate for the DE soft-loss objective.

The model learns:

    (X1, X2, ..., X20) -> average DE soft loss for X5

Labels reuse de_softmax_optimizer.evaluate_scenario_loss so the surrogate stays
aligned with the current rule implementation.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import qmc

import de_softmax_optimizer as optimizer

torch: Any | None
nn: Any | None
DataLoader: Any | None
TensorDataset: Any | None

try:
    import torch as torch_import
    from torch import nn as nn_import
    from torch.utils.data import DataLoader as DataLoader_import
    from torch.utils.data import TensorDataset as TensorDataset_import
except ModuleNotFoundError:  # pragma: no cover - exercised only when torch is not installed.
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None
else:
    torch = torch_import
    nn = nn_import
    DataLoader = DataLoader_import
    TensorDataset = TensorDataset_import


FULL_BID_DIMENSION = optimizer.NUM_UNITS
FULL_BID_LOWER = 10.0
FULL_BID_UPPER = 30.0
DEFAULT_WIDTH = 256
DEFAULT_BLOCKS = 4
DEFAULT_HEAD_WIDTH = 128
DEFAULT_VALIDATION_FRACTION = 0.2
DEFAULT_BATCH_SIZE = 4096
DEFAULT_LEARNING_RATE = 3e-4
DEFAULT_WEIGHT_DECAY = 1e-4

SCENARIOS = tuple(
    optimizer.Scenario(
        q=int(q),
        b2=float(b2),
        exclude_lowest_price_count=int(exclude_count),
        target_n=int(target_n),
    )
    for q in optimizer.Q_VALUES
    for b2 in optimizer.B2_VALUES
    for exclude_count in optimizer.EXCLUDE_LOWEST_PRICE_VALUES
    for target_n in optimizer.TARGET_N_VALUES
)


def require_torch_modules() -> tuple[Any, Any, Any, Any]:
    if torch is None or nn is None or DataLoader is None or TensorDataset is None:
        raise RuntimeError(
            "PyTorch is required for training and prediction. "
            "Run `uv sync` or install the project dependencies first."
        )
    return torch, nn, DataLoader, TensorDataset


def as_jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): as_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [as_jsonable(item) for item in value]
    return value


def options_to_kwargs(options: optimizer.ObjectiveOptions) -> dict[str, float]:
    return {
        "price_scale": float(options.price_scale),
        "score_scale": float(options.score_scale),
        "temperature": float(options.temperature),
        "invalid_cost": float(options.invalid_cost),
    }


def full_bid_vector(full_bids: Sequence[float] | np.ndarray) -> np.ndarray:
    bids = np.asarray(full_bids, dtype=float)
    if bids.shape != (FULL_BID_DIMENSION,):
        raise ValueError(f"full_bids must have shape ({FULL_BID_DIMENSION},), got {bids.shape}.")
    return bids


def split_full_bids(full_bids: Sequence[float] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    bids = full_bid_vector(full_bids)
    decision_vars = np.asarray([bids[unit - 1] for unit in optimizer.ADJUSTABLE_UNITS], dtype=float)
    env_vars = np.asarray([bids[unit - 1] for unit in optimizer.NON_ADJUSTABLE_UNITS], dtype=float)
    return decision_vars, env_vars


def full_bid_soft_loss(
    full_bids: Sequence[float] | np.ndarray,
    options: optimizer.ObjectiveOptions | None = None,
) -> float:
    """
    Compute a unique soft-loss label for one complete 20-dimensional bid vector.

    The 108 scenarios enumerate Q, B2, excluded-lowest-count, and target_n.
    evaluate_scenario_loss already averages the three K2 values internally.
    """

    objective_options = options or optimizer.ObjectiveOptions()
    decision_vars, env_vars = split_full_bids(full_bids)
    total_loss = 0.0
    for scenario in SCENARIOS:
        scenario_loss = optimizer.evaluate_scenario_loss(
            decision_vars=decision_vars,
            env_vars=env_vars,
            scenario=scenario,
            options=objective_options,
        )
        total_loss += scenario_loss.loss
    return float(total_loss / len(SCENARIOS))


def sample_full_bids(samples: int, seed: int, use_sobol: bool = True) -> np.ndarray:
    if samples <= 0:
        raise ValueError("samples must be positive.")

    if use_sobol:
        sampler = qmc.Sobol(d=FULL_BID_DIMENSION, scramble=True, seed=seed)
        if samples & (samples - 1) == 0:
            unit_samples = sampler.random_base2(m=int(math.log2(samples)))
        else:
            unit_samples = sampler.random(samples)
    else:
        rng = np.random.default_rng(seed)
        unit_samples = rng.random((samples, FULL_BID_DIMENSION))

    return (FULL_BID_LOWER + unit_samples * (FULL_BID_UPPER - FULL_BID_LOWER)).astype(np.float32)


def resolve_worker_count(workers: int, task_count: int) -> int:
    if task_count <= 0:
        raise ValueError("task_count must be positive.")
    if workers == -1:
        return max(1, min(mp.cpu_count(), task_count))
    if workers <= 0:
        raise ValueError("workers must be -1 or a positive integer.")
    return max(1, min(int(workers), task_count))


def _label_worker(args: tuple[np.ndarray, dict[str, float]]) -> float:
    row, options_kwargs = args
    return full_bid_soft_loss(row, optimizer.ObjectiveOptions(**options_kwargs))


def generate_labels(
    bids: np.ndarray,
    options: optimizer.ObjectiveOptions,
    workers: int = 1,
    chunksize: int = 64,
) -> np.ndarray:
    if bids.ndim != 2 or bids.shape[1] != FULL_BID_DIMENSION:
        raise ValueError(f"bids must have shape (n, {FULL_BID_DIMENSION}).")

    actual_workers = resolve_worker_count(workers, int(bids.shape[0]))
    options_kwargs = options_to_kwargs(options)
    if actual_workers == 1:
        labels = [full_bid_soft_loss(row, options) for row in bids]
    else:
        with mp.Pool(processes=actual_workers) as pool:
            label_iter = pool.imap(
                _label_worker,
                ((row, options_kwargs) for row in bids),
                chunksize=max(1, chunksize),
            )
            labels = list(label_iter)
    return np.asarray(labels, dtype=np.float32)


def save_dataset(
    output_path: Path,
    bids: np.ndarray,
    labels: np.ndarray,
    metadata: dict[str, Any],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        bids=bids.astype(np.float32),
        labels=labels.astype(np.float32),
        metadata=np.asarray(json.dumps(as_jsonable(metadata), ensure_ascii=False)),
    )


def load_dataset(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    with np.load(path, allow_pickle=False) as data:
        bids = np.asarray(data["bids"], dtype=np.float32)
        labels = np.asarray(data["labels"], dtype=np.float32)
        metadata_raw = str(data["metadata"].item()) if "metadata" in data else "{}"
    metadata = json.loads(metadata_raw)
    return bids, labels, metadata


def generate_dataset(
    samples: int,
    seed: int,
    output_path: Path,
    workers: int = 1,
    chunksize: int = 64,
    use_sobol: bool = True,
    options: optimizer.ObjectiveOptions | None = None,
) -> dict[str, Any]:
    start_time = time.time()
    objective_options = options or optimizer.ObjectiveOptions()
    bids = sample_full_bids(samples=samples, seed=seed, use_sobol=use_sobol)
    labels = generate_labels(bids, options=objective_options, workers=workers, chunksize=chunksize)
    elapsed = time.time() - start_time

    metadata = {
        "samples": int(samples),
        "seed": int(seed),
        "use_sobol": bool(use_sobol),
        "full_bid_dimension": FULL_BID_DIMENSION,
        "full_bid_bounds": [FULL_BID_LOWER, FULL_BID_UPPER],
        "target_unit": optimizer.unit_key(optimizer.TARGET_UNIT),
        "label": "average_de_soft_loss_over_108_discrete_scenarios",
        "scenario_count": len(SCENARIOS),
        "k2_mode": "evaluate_scenario_loss_averages_all_k2_values",
        "objective_options": options_to_kwargs(objective_options),
        "elapsed_seconds": elapsed,
        "label_summary": {
            "min": float(labels.min()),
            "max": float(labels.max()),
            "mean": float(labels.mean()),
            "std": float(labels.std()),
        },
    }
    save_dataset(output_path, bids, labels, metadata)
    return metadata


_TORCH_MODULE_BASE = nn.Module if nn is not None else object


class ResidualBlock(_TORCH_MODULE_BASE):  # type: ignore[misc, valid-type]
    def __init__(self, width: int) -> None:
        super().__init__()
        _, nn_module, _, _ = require_torch_modules()
        self.net = nn_module.Sequential(
            nn_module.Linear(width, width),
            nn_module.LayerNorm(width),
            nn_module.SiLU(),
            nn_module.Linear(width, width),
        )
        self.norm = nn_module.LayerNorm(width)
        self.activation = nn_module.SiLU()

    def forward(self, inputs: Any) -> Any:
        return self.activation(self.norm(inputs + self.net(inputs)))


class ResidualMLP(_TORCH_MODULE_BASE):  # type: ignore[misc, valid-type]
    def __init__(
        self,
        input_dim: int = FULL_BID_DIMENSION,
        width: int = DEFAULT_WIDTH,
        blocks: int = DEFAULT_BLOCKS,
        head_width: int = DEFAULT_HEAD_WIDTH,
    ) -> None:
        super().__init__()
        _, nn_module, _, _ = require_torch_modules()
        self.input_layer = nn_module.Sequential(
            nn_module.Linear(input_dim, width),
            nn_module.LayerNorm(width),
            nn_module.SiLU(),
        )
        self.blocks = nn_module.Sequential(*(ResidualBlock(width) for _ in range(blocks)))
        self.head = nn_module.Sequential(
            nn_module.Linear(width, head_width),
            nn_module.SiLU(),
            nn_module.Linear(head_width, 1),
        )

    def forward(self, inputs: Any) -> Any:
        hidden = self.input_layer(inputs)
        hidden = self.blocks(hidden)
        return self.head(hidden).squeeze(-1)


def select_device(requested: str) -> str:
    torch_module, _, _, _ = require_torch_modules()
    if requested != "auto":
        return requested
    if torch_module.cuda.is_available():
        return "cuda"
    if hasattr(torch_module.backends, "mps") and torch_module.backends.mps.is_available():
        return "mps"
    return "cpu"


def normalize_bids(bids: np.ndarray, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return ((bids.astype(np.float32) - center.astype(np.float32)) / scale.astype(np.float32)).astype(np.float32)


def split_train_validation(
    bids: np.ndarray,
    labels: np.ndarray,
    validation_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1.")
    if bids.shape[0] != labels.shape[0]:
        raise ValueError("bids and labels must contain the same number of rows.")
    if bids.shape[0] < 2:
        raise ValueError("at least two samples are required for a train/validation split.")

    rng = np.random.default_rng(seed)
    order = rng.permutation(bids.shape[0])
    validation_size = max(1, int(round(bids.shape[0] * validation_fraction)))
    validation_size = min(validation_size, bids.shape[0] - 1)
    validation_indices = order[:validation_size]
    train_indices = order[validation_size:]
    return bids[train_indices], labels[train_indices], bids[validation_indices], labels[validation_indices]


def compute_regression_metrics(predictions: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    errors = predictions - labels
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors * errors)))
    max_error = float(np.max(np.abs(errors)))
    return {"mae": mae, "rmse": rmse, "max_error": max_error}


def evaluate_model(
    model: Any,
    x_values: np.ndarray,
    y_values: np.ndarray,
    label_mean: float,
    label_std: float,
    device: str,
    batch_size: int,
) -> dict[str, float]:
    torch_module, _, loader_cls, dataset_cls = require_torch_modules()
    model.eval()
    dataset = dataset_cls(
        torch_module.as_tensor(x_values, dtype=torch_module.float32),
        torch_module.as_tensor(y_values, dtype=torch_module.float32),
    )
    loader = loader_cls(dataset, batch_size=batch_size, shuffle=False)
    predictions: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    with torch_module.no_grad():
        for batch_x, batch_y in loader:
            pred_scaled = model(batch_x.to(device)).detach().cpu().numpy()
            predictions.append((pred_scaled * label_std + label_mean).astype(np.float32))
            labels.append((batch_y.numpy() * label_std + label_mean).astype(np.float32))
    return compute_regression_metrics(np.concatenate(predictions), np.concatenate(labels))


def train_model(
    data_path: Path,
    output_dir: Path,
    epochs: int = 200,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
    seed: int = 42,
    device: str = "auto",
    width: int = DEFAULT_WIDTH,
    blocks: int = DEFAULT_BLOCKS,
    head_width: int = DEFAULT_HEAD_WIDTH,
) -> dict[str, Any]:
    if epochs <= 0:
        raise ValueError("epochs must be positive.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    torch_module, nn_module, loader_cls, dataset_cls = require_torch_modules()
    selected_device = select_device(device)
    torch_module.manual_seed(seed)
    if selected_device == "cuda":
        torch_module.cuda.manual_seed_all(seed)

    bids, labels, dataset_metadata = load_dataset(data_path)
    train_bids, train_labels, val_bids, val_labels = split_train_validation(
        bids,
        labels,
        validation_fraction=validation_fraction,
        seed=seed,
    )

    x_center = np.full((FULL_BID_DIMENSION,), 20.0, dtype=np.float32)
    x_scale = np.full((FULL_BID_DIMENSION,), 10.0, dtype=np.float32)
    label_mean = float(train_labels.mean())
    label_std = float(train_labels.std())
    if label_std <= 1e-12:
        label_std = 1.0

    x_train = normalize_bids(train_bids, x_center, x_scale)
    x_val = normalize_bids(val_bids, x_center, x_scale)
    y_train = ((train_labels - label_mean) / label_std).astype(np.float32)
    y_val = ((val_labels - label_mean) / label_std).astype(np.float32)

    train_dataset = dataset_cls(
        torch_module.as_tensor(x_train, dtype=torch_module.float32),
        torch_module.as_tensor(y_train, dtype=torch_module.float32),
    )
    generator = torch_module.Generator()
    generator.manual_seed(seed)
    train_loader = loader_cls(train_dataset, batch_size=batch_size, shuffle=True, generator=generator)

    model = ResidualMLP(input_dim=FULL_BID_DIMENSION, width=width, blocks=blocks, head_width=head_width).to(
        selected_device
    )
    criterion = nn_module.HuberLoss(delta=1.0)
    optimizer_instance = torch_module.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    model_config = {
        "model_type": "ResidualMLP",
        "input_dim": FULL_BID_DIMENSION,
        "width": int(width),
        "blocks": int(blocks),
        "head_width": int(head_width),
    }
    normalization = {
        "x_center": x_center.tolist(),
        "x_scale": x_scale.tolist(),
        "label_mean": label_mean,
        "label_std": label_std,
    }

    best_val_rmse = float("inf")
    best_epoch = 0
    history: list[dict[str, float | int]] = []
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_rows = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(selected_device)
            batch_y = batch_y.to(selected_device)
            optimizer_instance.zero_grad(set_to_none=True)
            prediction = model(batch_x)
            loss = criterion(prediction, batch_y)
            loss.backward()
            optimizer_instance.step()

            rows = int(batch_x.shape[0])
            train_loss_sum += float(loss.detach().cpu().item()) * rows
            train_rows += rows

        val_metrics = evaluate_model(
            model,
            x_val,
            y_val,
            label_mean=label_mean,
            label_std=label_std,
            device=selected_device,
            batch_size=batch_size,
        )
        train_loss = train_loss_sum / max(1, train_rows)
        epoch_record = {
            "epoch": epoch,
            "train_huber_loss": train_loss,
            "val_mae": val_metrics["mae"],
            "val_rmse": val_metrics["rmse"],
            "val_max_error": val_metrics["max_error"],
        }
        history.append(epoch_record)

        if val_metrics["rmse"] < best_val_rmse:
            best_val_rmse = val_metrics["rmse"]
            best_epoch = epoch
            torch_module.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": model_config,
                    "normalization": normalization,
                    "dataset_metadata": dataset_metadata,
                },
                output_dir / "model.pt",
            )

    elapsed = time.time() - start_time
    metrics = {
        "best_epoch": best_epoch,
        "best_val_rmse": best_val_rmse,
        "elapsed_seconds": elapsed,
        "device": selected_device,
        "samples": int(bids.shape[0]),
        "train_samples": int(train_bids.shape[0]),
        "validation_samples": int(val_bids.shape[0]),
        "history": history,
    }
    (output_dir / "model_config.json").write_text(json.dumps(model_config, indent=2), encoding="utf-8")
    (output_dir / "normalization.json").write_text(json.dumps(normalization, indent=2), encoding="utf-8")
    (output_dir / "metrics.json").write_text(json.dumps(as_jsonable(metrics), indent=2), encoding="utf-8")
    return metrics


def load_model(model_dir: Path, device: str = "auto") -> tuple[Any, dict[str, Any], dict[str, Any]]:
    torch_module, _, _, _ = require_torch_modules()
    selected_device = select_device(device)
    checkpoint = torch_module.load(model_dir / "model.pt", map_location=selected_device)
    config = dict(checkpoint["model_config"])
    normalization = dict(checkpoint["normalization"])
    model = ResidualMLP(
        input_dim=int(config["input_dim"]),
        width=int(config["width"]),
        blocks=int(config["blocks"]),
        head_width=int(config["head_width"]),
    ).to(selected_device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, normalization, {"device": selected_device, "model_config": config}


def parse_bids_payload(payload: Any) -> np.ndarray:
    if isinstance(payload, dict):
        row = [float(payload[f"X{unit}"]) for unit in range(1, FULL_BID_DIMENSION + 1)]
        return np.asarray([row], dtype=np.float32)
    if isinstance(payload, list):
        array = np.asarray(payload, dtype=np.float32)
        if array.shape == (FULL_BID_DIMENSION,):
            return array.reshape(1, FULL_BID_DIMENSION)
        if array.ndim == 2 and array.shape[1] == FULL_BID_DIMENSION:
            return array
    raise ValueError("bids JSON must be an X1..X20 dict, a 20-value list, or a list of 20-value lists.")


def load_prediction_bids(path: Path) -> np.ndarray:
    return parse_bids_payload(json.loads(path.read_text(encoding="utf-8")))


def predict_soft_loss(model_dir: Path, bids: np.ndarray, device: str = "auto") -> np.ndarray:
    torch_module, _, _, _ = require_torch_modules()
    model, normalization, runtime = load_model(model_dir, device=device)
    selected_device = str(runtime["device"])
    x_center = np.asarray(normalization["x_center"], dtype=np.float32)
    x_scale = np.asarray(normalization["x_scale"], dtype=np.float32)
    label_mean = float(normalization["label_mean"])
    label_std = float(normalization["label_std"])

    x_values = normalize_bids(bids, x_center, x_scale)
    with torch_module.no_grad():
        prediction_scaled = model(torch_module.as_tensor(x_values, dtype=torch_module.float32).to(selected_device))
    return (prediction_scaled.detach().cpu().numpy() * label_std + label_mean).astype(np.float32)


def command_generate(args: argparse.Namespace) -> None:
    options = optimizer.ObjectiveOptions(
        price_scale=args.price_scale,
        score_scale=args.score_scale,
        temperature=args.temperature,
        invalid_cost=args.invalid_cost,
    )
    metadata = generate_dataset(
        samples=args.samples,
        seed=args.seed,
        output_path=args.output,
        workers=args.workers,
        chunksize=args.chunksize,
        use_sobol=not args.random,
        options=options,
    )
    print(json.dumps(as_jsonable(metadata), ensure_ascii=False, indent=2))


def command_train(args: argparse.Namespace) -> None:
    metrics = train_model(
        data_path=args.data,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        device=args.device,
        width=args.width,
        blocks=args.blocks,
        head_width=args.head_width,
    )
    print(json.dumps(as_jsonable(metrics), ensure_ascii=False, indent=2))


def command_predict(args: argparse.Namespace) -> None:
    bids = load_prediction_bids(args.bids_json)
    predictions = predict_soft_loss(args.model_dir, bids, device=args.device)
    rows: list[dict[str, Any]] = []
    for index, (row, prediction) in enumerate(zip(bids, predictions)):
        item: dict[str, Any] = {
            "index": index,
            "predicted_soft_loss": float(prediction),
        }
        if args.compare_exact:
            item["exact_soft_loss"] = full_bid_soft_loss(row)
        rows.append(item)
    print(json.dumps(rows, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and use a neural surrogate for the DE soft-loss function.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_parser = subparsers.add_parser("generate", help="Generate a 20-dimensional bid dataset.")
    generate_parser.add_argument("--samples", type=int, default=1024)
    generate_parser.add_argument("--seed", type=int, default=42)
    generate_parser.add_argument("--output", type=Path, required=True)
    generate_parser.add_argument("--workers", type=int, default=60)
    generate_parser.add_argument("--chunksize", type=int, default=64)
    generate_parser.add_argument("--random", action="store_true", help="Use pseudo-random sampling instead of Sobol.")
    generate_parser.add_argument("--price-scale", type=float, default=optimizer.ObjectiveOptions().price_scale)
    generate_parser.add_argument("--score-scale", type=float, default=optimizer.ObjectiveOptions().score_scale)
    generate_parser.add_argument("--temperature", type=float, default=optimizer.ObjectiveOptions().temperature)
    generate_parser.add_argument("--invalid-cost", type=float, default=optimizer.ObjectiveOptions().invalid_cost)
    generate_parser.set_defaults(func=command_generate)

    train_parser = subparsers.add_parser("train", help="Train a Residual MLP surrogate.")
    train_parser.add_argument("--data", type=Path, required=True)
    train_parser.add_argument("--output-dir", type=Path, required=True)
    train_parser.add_argument("--epochs", type=int, default=200)
    train_parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    train_parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    train_parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    train_parser.add_argument("--validation-fraction", type=float, default=DEFAULT_VALIDATION_FRACTION)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--device", default="auto")
    train_parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    train_parser.add_argument("--blocks", type=int, default=DEFAULT_BLOCKS)
    train_parser.add_argument("--head-width", type=int, default=DEFAULT_HEAD_WIDTH)
    train_parser.set_defaults(func=command_train)

    predict_parser = subparsers.add_parser("predict", help="Predict soft loss for one or more bid vectors.")
    predict_parser.add_argument("--model-dir", type=Path, required=True)
    predict_parser.add_argument("--bids-json", type=Path, required=True)
    predict_parser.add_argument("--device", default="auto")
    predict_parser.add_argument("--compare-exact", action="store_true")
    predict_parser.set_defaults(func=command_predict)

    return parser


def main(argv: Iterable[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

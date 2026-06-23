"""
Gradient optimizer for bids using a trained neural surrogate.

The optimizer searches the existing adjustable units for low predicted
average DE soft-loss under Sobol-sampled non-adjustable competitor bids.
It writes candidate JSON files compatible with validate_de_results.py.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import qmc

import de_softmax_optimizer as optimizer
import neural_surrogate as surrogate

TRAIN_DOMAIN_LOWER = surrogate.FULL_BID_LOWER
TRAIN_DOMAIN_UPPER = surrogate.FULL_BID_UPPER
DEFAULT_SAMPLES = 8192
DEFAULT_STARTS = 256
DEFAULT_STEPS = 500
DEFAULT_TOP_K = 20
DEFAULT_START_BATCH_SIZE = 32
DEFAULT_ENV_BATCH_SIZE = 1024
DEFAULT_LEARNING_RATE = 0.05
LOGIT_EPSILON = 1e-4


@dataclass(frozen=True)
class SurrogateRuntime:
    model: Any
    device: str
    x_center: Any
    x_scale: Any
    label_mean: float
    label_std: float


@dataclass(frozen=True)
class RankedCandidate:
    rank: int
    objective_value: float
    adjustable_bids: np.ndarray


def as_jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): as_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [as_jsonable(item) for item in value]
    return value


def unit_key(unit: int) -> str:
    return f"X{unit}"


def parse_unit_key(key: str) -> int:
    if not key.startswith("X"):
        raise ValueError(f"Unknown unit key {key!r}; expected X1..X{optimizer.NUM_UNITS}.")
    try:
        unit = int(key[1:])
    except ValueError as exc:
        raise ValueError(f"Unknown unit key {key!r}; expected X1..X{optimizer.NUM_UNITS}.") from exc
    if unit < 1 or unit > optimizer.NUM_UNITS:
        raise ValueError(f"Unknown unit key {key!r}; expected X1..X{optimizer.NUM_UNITS}.")
    return unit


def validate_range(unit: int, value: Any) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{unit_key(unit)} range must be a two-value list.")
    lower = float(value[0])
    upper = float(value[1])
    if not math.isfinite(lower) or not math.isfinite(upper):
        raise ValueError(f"{unit_key(unit)} range bounds must be finite.")
    if lower > upper:
        raise ValueError(f"{unit_key(unit)} range lower bound must be <= upper bound.")
    if lower < TRAIN_DOMAIN_LOWER or upper > TRAIN_DOMAIN_UPPER:
        raise ValueError(
            f"{unit_key(unit)} range [{lower}, {upper}] is outside surrogate training domain "
            f"[{TRAIN_DOMAIN_LOWER}, {TRAIN_DOMAIN_UPPER}]."
        )
    return lower, upper


def default_bid_ranges() -> dict[int, tuple[float, float]]:
    return {unit: (float(bounds[0]), float(bounds[1])) for unit, bounds in optimizer.BID_BOUNDS.items()}


def load_range_overrides(path: Path | None) -> dict[int, tuple[float, float]]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError("--ranges-json must contain a JSON object.")

    overrides: dict[int, tuple[float, float]] = {}
    for key, value in data.items():
        unit = parse_unit_key(str(key))
        overrides[unit] = validate_range(unit, value)
    return overrides


def resolve_bid_ranges(
    path: Path | None = None,
) -> tuple[dict[int, tuple[float, float]], dict[int, tuple[float, float]]]:
    ranges = default_bid_ranges()
    overrides = load_range_overrides(path)
    ranges.update(overrides)
    for unit, bounds in ranges.items():
        ranges[unit] = validate_range(unit, bounds)
    return ranges, overrides


def bounds_for_units(units: Iterable[int], ranges: Mapping[int, tuple[float, float]]) -> tuple[np.ndarray, np.ndarray]:
    lower = []
    upper = []
    for unit in units:
        if unit not in ranges:
            raise ValueError(f"Missing range for {unit_key(unit)}.")
        bounds = validate_range(unit, ranges[unit])
        lower.append(bounds[0])
        upper.append(bounds[1])
    return np.asarray(lower, dtype=np.float32), np.asarray(upper, dtype=np.float32)


def is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def sobol_samples(rows: int, dimension: int, seed: int) -> np.ndarray:
    if rows <= 0:
        raise ValueError("sample count must be positive.")
    if dimension <= 0:
        raise ValueError("dimension must be positive.")

    sampler = qmc.Sobol(d=dimension, scramble=True, seed=seed)
    if is_power_of_two(rows):
        return sampler.random_base2(m=int(math.log2(rows))).astype(np.float32)
    return sampler.random(rows).astype(np.float32)


def scale_unit_samples(unit_samples: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return (lower + unit_samples * (upper - lower)).astype(np.float32)


def generate_env_matrix(samples: int, seed: int, ranges: Mapping[int, tuple[float, float]]) -> np.ndarray:
    lower, upper = bounds_for_units(optimizer.NON_ADJUSTABLE_UNITS, ranges)
    unit_samples = sobol_samples(samples, len(optimizer.NON_ADJUSTABLE_UNITS), seed)
    return scale_unit_samples(unit_samples, lower, upper)


def generate_initial_adjustable_bids(starts: int, seed: int, ranges: Mapping[int, tuple[float, float]]) -> np.ndarray:
    lower, upper = bounds_for_units(optimizer.ADJUSTABLE_UNITS, ranges)
    unit_samples = sobol_samples(starts, len(optimizer.ADJUSTABLE_UNITS), seed + 1)
    return scale_unit_samples(unit_samples, lower, upper)


def compose_full_bid_matrix(adjustable_bids: np.ndarray, env_vars: np.ndarray) -> np.ndarray:
    adjustable = np.asarray(adjustable_bids, dtype=np.float32)
    env = np.asarray(env_vars, dtype=np.float32)
    if adjustable.shape != (len(optimizer.ADJUSTABLE_UNITS),):
        raise ValueError(f"adjustable_bids must have shape ({len(optimizer.ADJUSTABLE_UNITS)},).")
    if env.shape != (len(optimizer.NON_ADJUSTABLE_UNITS),):
        raise ValueError(f"env_vars must have shape ({len(optimizer.NON_ADJUSTABLE_UNITS)},).")

    full = np.zeros((optimizer.NUM_UNITS,), dtype=np.float32)
    for col, unit in enumerate(optimizer.ADJUSTABLE_UNITS):
        full[unit - 1] = adjustable[col]
    for col, unit in enumerate(optimizer.NON_ADJUSTABLE_UNITS):
        full[unit - 1] = env[col]
    return full


def optimized_bid_summary(adjustable_bids: np.ndarray) -> dict[str, float]:
    return {
        unit_key(unit): round(float(value), optimizer.OUTPUT_DECIMAL_PLACES)
        for unit, value in zip(optimizer.ADJUSTABLE_UNITS, adjustable_bids)
    }


def full_bid_summary_at_env_mean(adjustable_bids: np.ndarray, env_matrix: np.ndarray) -> dict[str, float]:
    env_mean = np.asarray(env_matrix, dtype=np.float32).mean(axis=0)
    full = compose_full_bid_matrix(adjustable_bids, env_mean)
    return {
        unit_key(unit): round(float(full[unit - 1]), optimizer.OUTPUT_DECIMAL_PLACES)
        for unit in range(1, optimizer.NUM_UNITS + 1)
    }


def env_summary(
    env_matrix: np.ndarray,
    ranges: Mapping[int, tuple[float, float]],
) -> dict[str, dict[str, float | list[float]]]:
    summary: dict[str, dict[str, float | list[float]]] = {}
    for col, unit in enumerate(optimizer.NON_ADJUSTABLE_UNITS):
        values = env_matrix[:, col]
        lower, upper = ranges[unit]
        summary[unit_key(unit)] = {
            "configured_range": [float(lower), float(upper)],
            "sample_min": float(values.min()),
            "sample_max": float(values.max()),
            "sample_mean": float(values.mean()),
        }
    return summary


def ranges_to_json(ranges: Mapping[int, tuple[float, float]]) -> dict[str, list[float]]:
    return {unit_key(unit): [float(bounds[0]), float(bounds[1])] for unit, bounds in sorted(ranges.items())}


def load_surrogate_runtime(model_dir: Path, device: str) -> SurrogateRuntime:
    torch_module, _, _, _ = surrogate.require_torch_modules()
    model, normalization, runtime = surrogate.load_model(model_dir, device=device)
    selected_device = str(runtime["device"])
    x_center = torch_module.as_tensor(normalization["x_center"], dtype=torch_module.float32, device=selected_device)
    x_scale = torch_module.as_tensor(normalization["x_scale"], dtype=torch_module.float32, device=selected_device)
    return SurrogateRuntime(
        model=model,
        device=selected_device,
        x_center=x_center,
        x_scale=x_scale,
        label_mean=float(normalization["label_mean"]),
        label_std=float(normalization["label_std"]),
    )


def initial_bids_to_logits(initial_bids: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    span = upper - lower
    ratios = np.full_like(initial_bids, 0.5, dtype=np.float32)
    np.divide(initial_bids - lower, span, out=ratios, where=span > 0)
    ratios = np.clip(ratios, LOGIT_EPSILON, 1.0 - LOGIT_EPSILON)
    return np.log(ratios / (1.0 - ratios)).astype(np.float32)


def bids_from_logits(z_values: Any, lower: Any, upper: Any) -> Any:
    return lower + (upper - lower) * z_values.sigmoid()


def predicted_loss_for_env_chunk(
    runtime: SurrogateRuntime,
    adjustable_bids: Any,
    env_chunk: Any,
) -> Any:
    torch_module, _, _, _ = surrogate.require_torch_modules()
    start_count = int(adjustable_bids.shape[0])
    env_count = int(env_chunk.shape[0])
    full = torch_module.empty(
        (start_count, env_count, optimizer.NUM_UNITS),
        dtype=torch_module.float32,
        device=runtime.device,
    )
    for col, unit in enumerate(optimizer.ADJUSTABLE_UNITS):
        full[:, :, unit - 1] = adjustable_bids[:, col].unsqueeze(1)
    for col, unit in enumerate(optimizer.NON_ADJUSTABLE_UNITS):
        full[:, :, unit - 1] = env_chunk[:, col].unsqueeze(0)

    x_values = (full.reshape(-1, optimizer.NUM_UNITS) - runtime.x_center) / runtime.x_scale
    prediction = runtime.model(x_values) * runtime.label_std + runtime.label_mean
    return prediction.reshape(start_count, env_count).mean(dim=1)


def optimize_start_batch(
    runtime: SurrogateRuntime,
    initial_bids: np.ndarray,
    env_matrix: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    steps: int,
    learning_rate: float,
    env_batch_size: int,
) -> np.ndarray:
    torch_module, _, _, _ = surrogate.require_torch_modules()
    if steps <= 0:
        raise ValueError("--steps must be positive.")
    if learning_rate <= 0:
        raise ValueError("--learning-rate must be positive.")
    if env_batch_size <= 0:
        raise ValueError("--env-batch-size must be positive.")

    z_init = initial_bids_to_logits(initial_bids, lower, upper)
    z_values = torch_module.as_tensor(z_init, dtype=torch_module.float32, device=runtime.device).clone()
    z_values.requires_grad_(True)
    lower_tensor = torch_module.as_tensor(lower, dtype=torch_module.float32, device=runtime.device)
    upper_tensor = torch_module.as_tensor(upper, dtype=torch_module.float32, device=runtime.device)
    env_tensor = torch_module.as_tensor(env_matrix, dtype=torch_module.float32, device=runtime.device)
    torch_optimizer = torch_module.optim.Adam([z_values], lr=learning_rate)

    runtime.model.eval()
    env_count = int(env_tensor.shape[0])
    for _ in range(steps):
        torch_optimizer.zero_grad(set_to_none=True)
        for start in range(0, env_count, env_batch_size):
            env_chunk = env_tensor[start : start + env_batch_size]
            weight = float(env_chunk.shape[0]) / float(env_count)
            adjustable = bids_from_logits(z_values, lower_tensor, upper_tensor)
            loss_per_start = predicted_loss_for_env_chunk(runtime, adjustable, env_chunk)
            objective = loss_per_start.mean() * weight
            objective.backward()
        torch_optimizer.step()

    with torch_module.no_grad():
        final_bids = bids_from_logits(z_values, lower_tensor, upper_tensor).detach().cpu().numpy()
    return final_bids.astype(np.float32)


def score_adjustable_bids(
    runtime: SurrogateRuntime,
    candidates: np.ndarray,
    env_matrix: np.ndarray,
    env_batch_size: int,
    start_batch_size: int,
) -> np.ndarray:
    torch_module, _, _, _ = surrogate.require_torch_modules()
    if env_batch_size <= 0:
        raise ValueError("--env-batch-size must be positive.")
    if start_batch_size <= 0:
        raise ValueError("--start-batch-size must be positive.")

    env_tensor = torch_module.as_tensor(env_matrix, dtype=torch_module.float32, device=runtime.device)
    scores = []
    runtime.model.eval()
    with torch_module.no_grad():
        for row_start in range(0, candidates.shape[0], start_batch_size):
            candidate_batch = torch_module.as_tensor(
                candidates[row_start : row_start + start_batch_size],
                dtype=torch_module.float32,
                device=runtime.device,
            )
            totals = torch_module.zeros(candidate_batch.shape[0], dtype=torch_module.float32, device=runtime.device)
            for env_start in range(0, env_tensor.shape[0], env_batch_size):
                env_chunk = env_tensor[env_start : env_start + env_batch_size]
                chunk_losses = predicted_loss_for_env_chunk(runtime, candidate_batch, env_chunk)
                totals += chunk_losses * float(env_chunk.shape[0])
            scores.append((totals / float(env_tensor.shape[0])).detach().cpu().numpy())
    return np.concatenate(scores).astype(np.float32)


def select_ranked_candidates(
    candidates: np.ndarray,
    objective_values: np.ndarray,
    top_k: int,
) -> list[RankedCandidate]:
    if top_k <= 0:
        raise ValueError("--top-k must be positive.")

    order = np.argsort(objective_values)
    ranked: list[RankedCandidate] = []
    seen: set[tuple[float, ...]] = set()
    for index in order:
        bids = candidates[index]
        key = tuple(round(float(value), optimizer.OUTPUT_DECIMAL_PLACES) for value in bids)
        if key in seen:
            continue
        seen.add(key)
        ranked.append(
            RankedCandidate(
                rank=len(ranked) + 1,
                objective_value=float(objective_values[index]),
                adjustable_bids=np.asarray(bids, dtype=np.float32),
            )
        )
        if len(ranked) >= top_k:
            break
    return ranked


def build_candidate_payload(
    candidate: RankedCandidate,
    env_matrix: np.ndarray,
    ranges: Mapping[int, tuple[float, float]],
    overrides: Mapping[int, tuple[float, float]],
    config: dict[str, Any],
    elapsed_seconds: float,
    model_dir: Path,
) -> dict[str, Any]:
    return {
        "iteration": candidate.rank,
        "rank": candidate.rank,
        "objective_value": candidate.objective_value,
        "target_unit": unit_key(optimizer.TARGET_UNIT),
        "optimized_units": [unit_key(unit) for unit in optimizer.ADJUSTABLE_UNITS],
        "best_adjustable_bids": optimized_bid_summary(candidate.adjustable_bids),
        "full_bids_at_env_mean": full_bid_summary_at_env_mean(candidate.adjustable_bids, env_matrix),
        "non_adjustable_env_summary": env_summary(env_matrix, ranges),
        "optimizer_type": "neural_surrogate_gradient_adam",
        "model_dir": str(model_dir),
        "bid_ranges": ranges_to_json(ranges),
        "range_overrides": ranges_to_json(overrides),
        "config": config,
        "elapsed_seconds": float(elapsed_seconds),
        "nfev": None,
        "message": "Generated by neural surrogate gradient optimizer; validate with validate_de_results.py.",
    }


def save_result_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(as_jsonable(payload), file, ensure_ascii=False, indent=2)


def save_best_result_text(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        file.write("神经网络 Surrogate 梯度报价优化结果\n")
        file.write("=" * 72 + "\n")
        file.write(f"目标单位: {payload['target_unit']}\n")
        file.write(f"预测平均 soft-loss: {float(payload['objective_value']):.8f}\n")
        file.write(f"耗时: {float(payload['elapsed_seconds']):.1f} 秒\n")
        file.write(f"模型目录: {payload['model_dir']}\n")
        file.write("\n最优合作单位报价:\n")
        for name, value in payload["best_adjustable_bids"].items():
            file.write(f"  {name:<4} = {float(value):.{optimizer.OUTPUT_DECIMAL_PLACES}f}\n")
        file.write("\n注意: 这是 neural surrogate 预测损失，不是真实中标概率。\n")
        file.write("请继续使用 validate_de_results.py 做真实规则验证。\n")


def chunk_rows(array: np.ndarray, chunk_size: int) -> Iterable[np.ndarray]:
    for start in range(0, array.shape[0], chunk_size):
        yield array[start : start + chunk_size]


def optimize_with_surrogate(
    model_dir: Path,
    output_dir: Path,
    ranges_json: Path | None = None,
    samples: int = DEFAULT_SAMPLES,
    starts: int = DEFAULT_STARTS,
    steps: int = DEFAULT_STEPS,
    top_k: int = DEFAULT_TOP_K,
    seed: int = 42,
    device: str = "auto",
    start_batch_size: int = DEFAULT_START_BATCH_SIZE,
    env_batch_size: int = DEFAULT_ENV_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
) -> list[dict[str, Any]]:
    if samples <= 0:
        raise ValueError("--samples must be positive.")
    if starts <= 0:
        raise ValueError("--starts must be positive.")
    if top_k <= 0:
        raise ValueError("--top-k must be positive.")
    if start_batch_size <= 0:
        raise ValueError("--start-batch-size must be positive.")

    output_dir.mkdir(parents=True, exist_ok=True)
    ranges, overrides = resolve_bid_ranges(ranges_json)
    env_matrix = generate_env_matrix(samples, seed, ranges)
    initial_bids = generate_initial_adjustable_bids(starts, seed, ranges)
    lower, upper = bounds_for_units(optimizer.ADJUSTABLE_UNITS, ranges)
    runtime = load_surrogate_runtime(model_dir, device=device)

    config = {
        "samples": int(samples),
        "starts": int(starts),
        "steps": int(steps),
        "top_k": int(top_k),
        "seed": int(seed),
        "device": str(device),
        "selected_device": runtime.device,
        "start_batch_size": int(start_batch_size),
        "env_batch_size": int(env_batch_size),
        "learning_rate": float(learning_rate),
        "objective": "mean_predicted_soft_loss_over_sobol_competitor_samples",
        "parameterization": "bid = lower + (upper - lower) * sigmoid(z)",
        "range_source": str(ranges_json) if ranges_json is not None else "de_softmax_optimizer.BID_BOUNDS",
    }

    print("=" * 72)
    print("神经网络 Surrogate 梯度报价求解器")
    print("=" * 72)
    print(f"模型目录: {model_dir}")
    print(f"输出目录: {output_dir}")
    print(f"优化变量: {', '.join(unit_key(unit) for unit in optimizer.ADJUSTABLE_UNITS)}")
    print(f"竞争对手环境: {', '.join(unit_key(unit) for unit in optimizer.NON_ADJUSTABLE_UNITS)}")
    print(f"samples={samples}, starts={starts}, steps={steps}, top_k={top_k}, seed={seed}")
    print(f"device={device} -> {runtime.device}, start_batch_size={start_batch_size}, env_batch_size={env_batch_size}")
    print("=" * 72)

    start_time = time.time()
    optimized_batches = []
    for batch_index, batch in enumerate(chunk_rows(initial_bids, start_batch_size), start=1):
        final_batch = optimize_start_batch(
            runtime=runtime,
            initial_bids=batch,
            env_matrix=env_matrix,
            lower=lower,
            upper=upper,
            steps=steps,
            learning_rate=learning_rate,
            env_batch_size=env_batch_size,
        )
        optimized_batches.append(final_batch)
        print(f"[batch] {batch_index} optimized {final_batch.shape[0]} start(s)")

    optimized_bids = np.concatenate(optimized_batches, axis=0)
    objective_values = score_adjustable_bids(
        runtime=runtime,
        candidates=optimized_bids,
        env_matrix=env_matrix,
        env_batch_size=env_batch_size,
        start_batch_size=start_batch_size,
    )
    ranked_candidates = select_ranked_candidates(optimized_bids, objective_values, top_k)
    elapsed = time.time() - start_time

    payloads: list[dict[str, Any]] = []
    for candidate in ranked_candidates:
        payload = build_candidate_payload(
            candidate=candidate,
            env_matrix=env_matrix,
            ranges=ranges,
            overrides=overrides,
            config=config,
            elapsed_seconds=elapsed,
            model_dir=model_dir,
        )
        payloads.append(payload)
        save_result_json(output_dir / f"candidate_rank_{candidate.rank:03d}.json", payload)

    if payloads:
        save_result_json(output_dir / "best_result.json", payloads[0])
        save_best_result_text(output_dir / "best_result.txt", payloads[0])

    summary = {
        "config": config,
        "elapsed_seconds": elapsed,
        "candidate_count": len(payloads),
        "best_objective_value": payloads[0]["objective_value"] if payloads else None,
        "candidates": [
            {
                "rank": payload["rank"],
                "objective_value": payload["objective_value"],
                "best_adjustable_bids": payload["best_adjustable_bids"],
            }
            for payload in payloads
        ],
    }
    save_result_json(output_dir / "optimizer_summary.json", summary)

    if payloads:
        print("\n优化完成")
        print(f"最佳预测平均 soft-loss: {float(payloads[0]['objective_value']):.8f}")
    print(f"输出候选数: {len(payloads)}")
    print(f"耗时: {elapsed:.1f} 秒 ({elapsed / 60:.1f} 分钟)")
    print(f"结果已保存到: {output_dir}")
    return payloads


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize adjustable bids with a trained neural surrogate.")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ranges-json", type=Path)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--starts", type=int, default=DEFAULT_STARTS)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--start-batch-size", type=int, default=DEFAULT_START_BATCH_SIZE)
    parser.add_argument("--env-batch-size", type=int, default=DEFAULT_ENV_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    optimize_with_surrogate(
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        ranges_json=args.ranges_json,
        samples=args.samples,
        starts=args.starts,
        steps=args.steps,
        top_k=args.top_k,
        seed=args.seed,
        device=args.device,
        start_batch_size=args.start_batch_size,
        env_batch_size=args.env_batch_size,
        learning_rate=args.learning_rate,
    )


if __name__ == "__main__":
    main()

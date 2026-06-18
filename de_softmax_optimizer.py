"""
差分进化报价优化器：CRN + 失败环节软惩罚。

本程序按 qingbiao.md 的当前规则优化目标单位 X5。优化变量固定为：
X3, X5, X6, X7, X9, X10, X11, X13, X16, X17, X19, X20。

目标函数不是真实中标概率，而是一个可优化的近似损失：
- 按正式规则硬执行流程，定位 X5 最早失败环节。
- 在失败环节用连续距离惩罚提供优化信号。
- 环节五使用遵守 X_i > K 的 one-sided softmax 距离。
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

import numpy as np
from scipy.optimize import differential_evolution
from scipy.stats import qmc


# ===================== 当前规则常量 =====================

NUM_UNITS = 20
TARGET_UNIT = 5

A_SCORES = np.array(
    [
        86.09,
        82.15,
        82.02,
        81.70,
        79.57,
        80.76,
        81.23,
        80.20,
        79.86,
        81.16,
        79.92,
        75.50,
        81.19,
        75.00,
        75.00,
        75.00,
        77.35,
        75.00,
        87.63,
        75.00,
    ],
    dtype=float,
)

ENTRY_RANKINGS = np.array(
    [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
    ],
    dtype=int,
)

TECH_RANKINGS = np.array(
    [
        1,
        3,
        9,
        2,
        4,
        5,
        8,
        7,
        10,
        11,
        6,
        20,
        20,
        20,
        20,
        12,
        20,
        20,
        20,
        20,
    ],
    dtype=int,
)

PERFORMANCE_SCORES = np.array(
    [
        20,
        20,
        20,
        20,
        20,
        20,
        20,
        20,
        20,
        20,
        20,
        20,
        10,
        20,
        20,
        0,
        0,
        20,
        0,
        20,
    ],
    dtype=float,
)

BID_BOUNDS: dict[int, tuple[float, float]] = {
    1: (19.0, 20.0),
    2: (13.0, 19.0),
    3: (13.0, 19.0),
    4: (18.0, 22.0),
    5: (13.0, 19.0),
    6: (19.0, 22.0),
    7: (13.0, 19.0),
    8: (15.0, 25.0),
    9: (10.0, 27.0),
    10: (13.0, 19.0),
    11: (19.0, 22.0),
    12: (19.0, 22.0),
    13: (13.0, 19.0),
    14: (16.0, 19.0),
    15: (16.0, 21.0),
    16: (13.0, 19.0),
    17: (13.0, 15.0),
    18: (24.0, 27.0),
    19: (18.5, 20.5),
    20: (10.0, 27.0),
}

ADJUSTABLE_UNITS = [3, 5, 6, 7, 9, 10, 11, 13, 16, 17, 19, 20]
NON_ADJUSTABLE_UNITS = [
    unit for unit in range(1, NUM_UNITS + 1) if unit not in set(ADJUSTABLE_UNITS)
]

Q_VALUES = np.array([15, 16, 17, 18, 19, 20], dtype=int)
B2_VALUES = np.array([0.5, 1.0, 1.5], dtype=float)
EXCLUDE_LOWEST_PRICE_VALUES = np.array([1, 2], dtype=int)
TARGET_N_VALUES = np.array([3, 4, 5], dtype=int)
K2_VALUES = np.array([0.0, 0.25, 0.5], dtype=float)


# ===================== CRN 全局上下文 =====================

_GLOBAL_CONTEXT: "CRNContext | None" = None
_GLOBAL_OPTIONS: "ObjectiveOptions | None" = None


@dataclass(frozen=True)
class ObjectiveOptions:
    price_scale: float = 0.25
    score_scale: float = 2.0
    temperature: float = 0.1
    invalid_cost: float = 10.0


@dataclass(frozen=True)
class CRNContext:
    env_matrix: np.ndarray
    q_list: np.ndarray
    b2_list: np.ndarray
    exclude_list: np.ndarray
    target_n_list: np.ndarray
    seed: int

    @property
    def n_samples(self) -> int:
        return int(self.env_matrix.shape[0])


@dataclass(frozen=True)
class Scenario:
    q: int
    b2: float
    exclude_lowest_price_count: int
    target_n: int


@dataclass(frozen=True)
class ScenarioLoss:
    loss: float
    stage: str
    reason: str
    penalty: float


# ===================== 基础工具 =====================


def unit_key(unit: int) -> str:
    return f"X{unit}"


def round_half_up(value: float) -> int:
    return int(math.floor(float(value) + 0.5))


def round_half_up_places(value: float, places: int = 2) -> float:
    multiplier = 10**places
    return math.floor(float(value) * multiplier + 0.5) / multiplier


def lower_quote_key(unit: int, bids: np.ndarray) -> tuple[float, int]:
    """真实报价从低到高排序：下浮率越大，真实报价越低。"""

    return (-float(bids[unit]), unit)


def higher_quote_key(unit: int, bids: np.ndarray) -> tuple[float, int]:
    """真实报价从高到低排序：下浮率越小，真实报价越高。"""

    return (float(bids[unit]), unit)


def score_key(unit: int, scores: dict[int, float]) -> tuple[float, int]:
    return (-scores[unit], unit)


def bounded_margin_penalty(margin: float, scale: float) -> float:
    """
    把正向失败距离压到 [0, 1)。

    margin <= 0 表示已经达到或超过阈值，不额外惩罚。
    """

    if margin <= 0:
        return 0.0
    scaled = max(-60.0, min(60.0, float(margin) / max(scale, 1e-12)))
    sigmoid = 1.0 / (1.0 + math.exp(-scaled))
    return max(0.0, 2.0 * sigmoid - 1.0)


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


def adjustable_bounds() -> list[tuple[float, float]]:
    return [BID_BOUNDS[unit] for unit in ADJUSTABLE_UNITS]


def build_bid_vector(decision_vars: Iterable[float], env_vars: Iterable[float]) -> np.ndarray:
    """
    构造 1-indexed 报价数组，bids[unit] 对应 X_unit。
    """

    bids = np.zeros(NUM_UNITS + 1, dtype=float)
    for unit, value in zip(ADJUSTABLE_UNITS, decision_vars):
        bids[unit] = float(value)
    for unit, value in zip(NON_ADJUSTABLE_UNITS, env_vars):
        bids[unit] = float(value)
    return bids


def representative_bid_summary(decision_vars: Iterable[float], context: CRNContext) -> dict[str, float]:
    env_means = context.env_matrix.mean(axis=0)
    bids = build_bid_vector(decision_vars, env_means)
    return {unit_key(unit): round(float(bids[unit]), 4) for unit in range(1, NUM_UNITS + 1)}


def optimized_bid_summary(decision_vars: Iterable[float]) -> dict[str, float]:
    return {
        unit_key(unit): round(float(value), 4)
        for unit, value in zip(ADJUSTABLE_UNITS, decision_vars)
    }


def env_summary(context: CRNContext) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for col, unit in enumerate(NON_ADJUSTABLE_UNITS):
        values = context.env_matrix[:, col]
        summary[unit_key(unit)] = {
            "min": round(float(values.min()), 4),
            "max": round(float(values.max()), 4),
            "mean": round(float(values.mean()), 4),
        }
    return summary


# ===================== CRN 初始化 =====================


def initialize_crn_context(n_samples: int, seed: int = 42, use_sobol: bool = True) -> CRNContext:
    if n_samples <= 0:
        raise ValueError("n_samples 必须为正整数。")

    env_bounds = [BID_BOUNDS[unit] for unit in NON_ADJUSTABLE_UNITS]
    env_dimension = len(NON_ADJUSTABLE_UNITS)

    if use_sobol:
        sampler = qmc.Sobol(d=env_dimension, scramble=True, seed=seed)
        if n_samples > 0 and (n_samples & (n_samples - 1)) == 0:
            samples_01 = sampler.random_base2(m=int(math.log2(n_samples)))
        else:
            samples_01 = sampler.random(n_samples)
    else:
        rng = np.random.default_rng(seed)
        samples_01 = rng.random((n_samples, env_dimension))

    env_matrix = np.zeros((n_samples, env_dimension), dtype=float)
    for col, (lower, upper) in enumerate(env_bounds):
        env_matrix[:, col] = lower + samples_01[:, col] * (upper - lower)

    rng = np.random.default_rng(seed + 100)
    q_list = rng.choice(Q_VALUES, size=n_samples, replace=True)
    b2_list = rng.choice(B2_VALUES, size=n_samples, replace=True, p=[1 / 3, 1 / 3, 1 / 3])
    exclude_list = rng.choice(EXCLUDE_LOWEST_PRICE_VALUES, size=n_samples, replace=True)
    target_n_list = rng.choice(TARGET_N_VALUES, size=n_samples, replace=True)

    return CRNContext(
        env_matrix=env_matrix,
        q_list=q_list,
        b2_list=b2_list,
        exclude_list=exclude_list,
        target_n_list=target_n_list,
        seed=seed,
    )


def set_global_context(context: CRNContext, options: ObjectiveOptions) -> None:
    global _GLOBAL_CONTEXT, _GLOBAL_OPTIONS
    _GLOBAL_CONTEXT = context
    _GLOBAL_OPTIONS = options


def _init_worker(context: CRNContext, options: ObjectiveOptions) -> None:
    set_global_context(context, options)


# ===================== 快速规则 evaluator =====================


def calculate_business_baseline(qualified: list[int], bids: np.ndarray) -> float:
    quote_low_to_high = sorted(qualified, key=lambda unit: lower_quote_key(unit, bids))
    num_to_remove = round_half_up(len(qualified) * 0.15)
    if num_to_remove == 0:
        middle_units = quote_low_to_high
    else:
        middle_units = quote_low_to_high[num_to_remove:-num_to_remove]

    rounded_unique_values = sorted({round_half_up(float(bids[unit])) for unit in middle_units})
    return round_half_up_places(mean(rounded_unique_values), 2)


def calculate_b_value(recommended: list[int], bids: np.ndarray, b2: float) -> float:
    removed_unit = sorted(recommended, key=lambda unit: higher_quote_key(unit, bids))[0]
    remaining_units = [unit for unit in recommended if unit != removed_unit]
    if not remaining_units:
        b1 = float(bids[removed_unit])
    else:
        b1 = mean(float(bids[unit]) for unit in remaining_units)
    return float(b1 + b2)


def calculate_bid_scores(recommended: list[int], bids: np.ndarray, b_value: float) -> dict[int, float]:
    ordered = sorted(
        recommended,
        key=lambda unit: (abs(float(bids[unit]) - b_value), -float(bids[unit]), unit),
    )

    scores: dict[int, float] = {}
    current_score = 20.0
    i = 0
    while i < len(ordered):
        current_unit = ordered[i]
        current_diff = abs(float(bids[current_unit]) - b_value)
        current_bid = float(bids[current_unit])

        j = i + 1
        while j < len(ordered):
            next_unit = ordered[j]
            next_diff = abs(float(bids[next_unit]) - b_value)
            next_bid = float(bids[next_unit])
            if next_diff != current_diff or next_bid != current_bid:
                break
            j += 1

        same_count = j - i
        for unit in ordered[i:j]:
            scores[unit] = max(0.0, current_score)

        current_score -= same_count
        i = j

    return scores


def calculate_fulfillment_scores(recommended: list[int]) -> dict[int, float]:
    indices = [unit - 1 for unit in recommended]
    values = A_SCORES[indices]
    min_value = float(values.min())
    max_value = float(values.max())
    if max_value == min_value:
        return {unit: 0.0 for unit in recommended}
    return {
        unit: 10.0 * (float(A_SCORES[unit - 1]) - min_value) / (max_value - min_value)
        for unit in recommended
    }


def business_margin_to_recommend(
    target_unit: int,
    business_recommended: list[int],
    business_baseline: float,
    bids: np.ndarray,
) -> float:
    if target_unit in business_recommended:
        return 0.0
    if not business_recommended:
        return 1.0
    target_distance = abs(float(bids[target_unit]) - business_baseline)
    cutoff_unit = business_recommended[-1]
    cutoff_distance = abs(float(bids[cutoff_unit]) - business_baseline)
    margin = target_distance - cutoff_distance
    if margin == 0 and float(bids[target_unit]) < float(bids[cutoff_unit]):
        margin = 0.01
    return max(0.0, margin)


def tech_margin_to_recommend(
    target_unit: int,
    tech_recommended: list[int],
) -> float:
    if target_unit in tech_recommended:
        return 0.0
    if not tech_recommended:
        return 1.0
    target_rank = int(TECH_RANKINGS[target_unit - 1])
    cutoff_rank = max(int(TECH_RANKINGS[unit - 1]) for unit in tech_recommended)
    return max(0.0, float(target_rank - cutoff_rank) / 10.0)


def blocked_low_price_margin(
    target_unit: int,
    ordered_low_price_units: list[int],
    blocked_count: int,
    bids: np.ndarray,
) -> float:
    if target_unit not in ordered_low_price_units[:blocked_count]:
        return 0.0
    if len(ordered_low_price_units) <= blocked_count:
        return 1.0
    cutoff_safe_unit = ordered_low_price_units[blocked_count]
    return max(0.0, float(bids[target_unit]) - float(bids[cutoff_safe_unit]))


def score_margin_to_finalist(
    target_unit: int,
    finalists_ordered: list[int],
    target_n: int,
    total_scores: dict[int, float],
) -> float:
    if target_unit in finalists_ordered[:target_n]:
        return 0.0
    if len(finalists_ordered) < target_n:
        return 0.0
    threshold_unit = finalists_ordered[target_n - 1]
    return max(0.0, float(total_scores[threshold_unit] - total_scores[target_unit]))


def one_sided_softmax_probability(
    finalists: list[int],
    bids: np.ndarray,
    final_k: float,
    target_unit: int = TARGET_UNIT,
    temperature: float = 0.1,
    invalid_cost: float = 10.0,
) -> float:
    """
    环节五 softmax：只有 X_i > K 的单位有正常 winner cost。

    合格单位 cost = X_i - K，越小越好；不合格单位 cost 加 invalid_cost。
    """

    costs = []
    for unit in finalists:
        gap = float(bids[unit]) - final_k
        if gap > 0:
            cost = gap
        else:
            cost = invalid_cost + abs(gap)
        costs.append(cost)

    cost_array = np.array(costs, dtype=float)
    shifted = cost_array - float(cost_array.min())
    exp_scores = np.exp(-shifted / max(temperature, 1e-12))
    denominator = float(exp_scores.sum())
    if denominator <= 0 or target_unit not in finalists:
        return 0.0
    target_index = finalists.index(target_unit)
    return float(exp_scores[target_index] / denominator)


def hard_winner(finalists: list[int], bids: np.ndarray, final_k: float) -> int | None:
    eligible = [unit for unit in finalists if float(bids[unit]) > final_k]
    if not eligible:
        return None
    return sorted(eligible, key=lambda unit: (float(bids[unit]) - final_k, unit))[0]


def evaluate_scenario_loss(
    decision_vars: np.ndarray,
    env_vars: np.ndarray,
    scenario: Scenario,
    options: ObjectiveOptions,
) -> ScenarioLoss:
    bids = build_bid_vector(decision_vars, env_vars)
    q1 = round_half_up(scenario.q / 2)

    # 环节一：入围
    qualified = sorted(
        range(1, NUM_UNITS + 1),
        key=lambda unit: (int(ENTRY_RANKINGS[unit - 1]), unit),
    )[: scenario.q]
    if TARGET_UNIT not in qualified:
        target_rank = int(ENTRY_RANKINGS[TARGET_UNIT - 1])
        margin = max(0.0, float(target_rank - scenario.q))
        penalty = bounded_margin_penalty(margin, 1.0)
        return ScenarioLoss(4.0 + penalty, "entry", "target_not_qualified", penalty)

    # 环节二：推优
    ordered_low_price = sorted(qualified, key=lambda unit: lower_quote_key(unit, bids))
    blocked_low_price_units = ordered_low_price[:2]
    recommend_eligible = [
        unit for unit in qualified if unit not in set(blocked_low_price_units)
    ]

    tech_recommended = sorted(
        recommend_eligible,
        key=lambda unit: (int(TECH_RANKINGS[unit - 1]), unit),
    )[:q1]

    business_baseline = calculate_business_baseline(qualified, bids)
    business_recommended = sorted(
        recommend_eligible,
        key=lambda unit: (
            abs(float(bids[unit]) - business_baseline),
            -float(bids[unit]),
            unit,
        ),
    )[:q1]

    recommended = sorted(set(tech_recommended) | set(business_recommended))

    if TARGET_UNIT not in recommended:
        if TARGET_UNIT in blocked_low_price_units:
            margin = blocked_low_price_margin(TARGET_UNIT, ordered_low_price, 2, bids)
            reason = "target_blocked_as_lowest_price"
        else:
            business_margin = business_margin_to_recommend(
                TARGET_UNIT,
                business_recommended,
                business_baseline,
                bids,
            )
            tech_margin = tech_margin_to_recommend(TARGET_UNIT, tech_recommended)
            margin = min(business_margin, tech_margin)
            reason = "target_not_recommended"
        penalty = bounded_margin_penalty(margin, options.price_scale)
        return ScenarioLoss(3.0 + penalty, "recommendation", reason, penalty)

    # 环节三：清标得分
    b_value = calculate_b_value(recommended, bids, scenario.b2)
    bid_scores = calculate_bid_scores(recommended, bids, b_value)
    fulfillment_scores = calculate_fulfillment_scores(recommended)
    total_scores = {
        unit: 50.0
        + fulfillment_scores[unit]
        + float(PERFORMANCE_SCORES[unit - 1])
        + bid_scores[unit]
        for unit in recommended
    }

    # 环节四：定标
    ordered_recommended_low_price = sorted(
        recommended,
        key=lambda unit: lower_quote_key(unit, bids),
    )
    excluded_lowest_price = ordered_recommended_low_price[
        : scenario.exclude_lowest_price_count
    ]
    candidates = [
        unit for unit in recommended if unit not in set(excluded_lowest_price)
    ]
    finalists_ordered = sorted(candidates, key=lambda unit: score_key(unit, total_scores))
    finalists = finalists_ordered[: scenario.target_n]

    if TARGET_UNIT in excluded_lowest_price:
        margin = blocked_low_price_margin(
            TARGET_UNIT,
            ordered_recommended_low_price,
            scenario.exclude_lowest_price_count,
            bids,
        )
        penalty = bounded_margin_penalty(margin, options.price_scale)
        return ScenarioLoss(2.0 + penalty, "finalist", "target_excluded_lowest_price", penalty)

    if TARGET_UNIT not in finalists:
        margin = score_margin_to_finalist(
            TARGET_UNIT,
            finalists_ordered,
            scenario.target_n,
            total_scores,
        )
        penalty = bounded_margin_penalty(margin, options.score_scale)
        return ScenarioLoss(2.0 + penalty, "finalist", "target_score_below_cutoff", penalty)

    # 环节五：中标，K2 三种取值在同一 CRN 场景内平均。
    final_k1 = mean(float(bids[unit]) for unit in finalists)
    k2_losses = []
    for k2 in K2_VALUES:
        final_k = float(final_k1 + k2)
        winner = hard_winner(finalists, bids, final_k)
        if winner == TARGET_UNIT:
            k2_losses.append(0.0)
            continue
        p_x5 = one_sided_softmax_probability(
            finalists,
            bids,
            final_k,
            target_unit=TARGET_UNIT,
            temperature=options.temperature,
            invalid_cost=options.invalid_cost,
        )
        k2_losses.append(1.0 + (1.0 - p_x5))

    loss = float(mean(k2_losses))
    return ScenarioLoss(loss, "winner", "target_not_winner", 0.0)


def robust_objective(decision_vars: np.ndarray) -> float:
    if _GLOBAL_CONTEXT is None or _GLOBAL_OPTIONS is None:
        raise RuntimeError("CRN context has not been initialized.")

    context = _GLOBAL_CONTEXT
    options = _GLOBAL_OPTIONS
    total_loss = 0.0
    for sample_idx in range(context.n_samples):
        scenario = Scenario(
            q=int(context.q_list[sample_idx]),
            b2=float(context.b2_list[sample_idx]),
            exclude_lowest_price_count=int(context.exclude_list[sample_idx]),
            target_n=int(context.target_n_list[sample_idx]),
        )
        scenario_loss = evaluate_scenario_loss(
            np.asarray(decision_vars, dtype=float),
            context.env_matrix[sample_idx],
            scenario,
            options,
        )
        total_loss += scenario_loss.loss
    return float(total_loss / context.n_samples)


def _objective_for_worker(decision_vars: np.ndarray) -> float:
    return robust_objective(decision_vars)


# ===================== checkpoint / 输出 =====================


def save_result_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(as_jsonable(payload), file, ensure_ascii=False, indent=2)


def build_result_payload(
    decision_vars: np.ndarray,
    objective_value: float,
    context: CRNContext,
    options: ObjectiveOptions,
    config: dict[str, Any],
    elapsed_seconds: float,
    iteration: int | None = None,
    nfev: int | None = None,
    message: str | None = None,
) -> dict[str, Any]:
    return {
        "iteration": iteration,
        "objective_value": float(objective_value),
        "target_unit": unit_key(TARGET_UNIT),
        "optimized_units": [unit_key(unit) for unit in ADJUSTABLE_UNITS],
        "best_adjustable_bids": optimized_bid_summary(decision_vars),
        "full_bids_at_crn_env_mean": representative_bid_summary(decision_vars, context),
        "non_adjustable_env_summary": env_summary(context),
        "crn_discrete_summary": {
            "q_counts": {
                str(value): int(np.sum(context.q_list == value)) for value in Q_VALUES
            },
            "b2_counts": {
                str(value): int(np.sum(context.b2_list == value)) for value in B2_VALUES
            },
            "exclude_counts": {
                str(value): int(np.sum(context.exclude_list == value))
                for value in EXCLUDE_LOWEST_PRICE_VALUES
            },
            "target_n_counts": {
                str(value): int(np.sum(context.target_n_list == value))
                for value in TARGET_N_VALUES
            },
        },
        "objective_options": options.__dict__,
        "config": config,
        "elapsed_seconds": float(elapsed_seconds),
        "nfev": nfev,
        "message": message,
    }


def save_result_text(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        file.write("差分进化报价优化结果\n")
        file.write("=" * 60 + "\n")
        file.write(f"目标单位: {payload['target_unit']}\n")
        file.write(f"目标函数值: {payload['objective_value']:.8f}\n")
        file.write(f"耗时: {payload['elapsed_seconds']:.1f} 秒\n")
        if payload.get("nfev") is not None:
            file.write(f"评估次数: {payload['nfev']}\n")
        if payload.get("message"):
            file.write(f"收敛信息: {payload['message']}\n")
        file.write("\n最优可调变量:\n")
        for name, value in payload["best_adjustable_bids"].items():
            file.write(f"  {name:<4} = {float(value):.4f}\n")
        file.write("\n用 CRN 环境均值补齐的 20 个报价摘要:\n")
        for name, value in payload["full_bids_at_crn_env_mean"].items():
            file.write(f"  {name:<4} = {float(value):.4f}\n")
        file.write("\n注意: 该目标函数是中标概率近似，不是真实中标概率。\n")


def make_checkpoint_callback(
    output_dir: Path,
    checkpoint_every: int,
    context: CRNContext,
    options: ObjectiveOptions,
    config: dict[str, Any],
    start_time: float,
):
    state = {"iteration": 0, "best_fun": math.inf, "best_x": None}

    def callback(xk: np.ndarray, convergence: float | None = None) -> bool:
        state["iteration"] += 1
        iteration = int(state["iteration"])

        should_save = checkpoint_every > 0 and iteration % checkpoint_every == 0
        if not should_save:
            return False

        objective_value = robust_objective(np.asarray(xk, dtype=float))
        if objective_value < state["best_fun"]:
            state["best_fun"] = objective_value
            state["best_x"] = np.asarray(xk, dtype=float).copy()

        best_x = state["best_x"] if state["best_x"] is not None else np.asarray(xk, dtype=float)
        best_fun = float(state["best_fun"])
        payload = build_result_payload(
            best_x,
            best_fun,
            context,
            options,
            config,
            elapsed_seconds=time.time() - start_time,
            iteration=iteration,
        )
        checkpoint_path = output_dir / f"checkpoint_iter_{iteration:04d}.json"
        save_result_json(checkpoint_path, payload)
        print(f"[checkpoint] iter={iteration}, objective={best_fun:.8f}, file={checkpoint_path}")
        return False

    return callback


# ===================== 优化入口 =====================


def optimize_with_differential_evolution(
    n_samples: int = 8192,
    maxiter: int = 800,
    popsize: int = 90,
    workers: int = -1,
    seed: int = 42,
    checkpoint_every: int = 50,
    output_dir: Path | str = Path("de_softmax_results"),
    price_scale: float = 0.25,
    score_scale: float = 2.0,
    temperature: float = 0.1,
    invalid_cost: float = 10.0,
    polish: bool = True,
    disp: bool = True,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    options = ObjectiveOptions(
        price_scale=price_scale,
        score_scale=score_scale,
        temperature=temperature,
        invalid_cost=invalid_cost,
    )
    context = initialize_crn_context(n_samples=n_samples, seed=seed, use_sobol=True)
    set_global_context(context, options)

    config = {
        "samples": n_samples,
        "maxiter": maxiter,
        "popsize": popsize,
        "workers": workers,
        "seed": seed,
        "checkpoint_every": checkpoint_every,
        "strategy": "best1bin",
        "mutation": [0.5, 1.5],
        "recombination": 0.7,
        "polish": polish,
        "updating": "deferred",
        "use_crn": True,
        "use_sobol": True,
        "k2_mode": "average_all_values",
    }

    print("=" * 72)
    print("差分进化报价优化器：X5 softmax 近似目标")
    print("=" * 72)
    print(f"优化变量: {', '.join(unit_key(unit) for unit in ADJUSTABLE_UNITS)}")
    print(f"不可调环境: {', '.join(unit_key(unit) for unit in NON_ADJUSTABLE_UNITS)}")
    print(f"CRN 样本数: {n_samples}")
    print(f"maxiter={maxiter}, popsize={popsize}, workers={workers}, seed={seed}")
    print(f"checkpoint_every={checkpoint_every}, output_dir={output_path}")
    print("注意: 目标函数是中标概率近似，不是真实中标概率。")
    print("=" * 72)

    start_time = time.time()
    pool: mp.pool.Pool | None = None
    map_workers: Any = 1
    if workers != 1:
        actual_workers = min(mp.cpu_count(), 90) if workers == -1 else int(workers)
        pool = mp.Pool(
            processes=actual_workers,
            initializer=_init_worker,
            initargs=(context, options),
        )
        map_workers = pool.map
        print(f"并行进程数: {actual_workers}")

    callback = make_checkpoint_callback(
        output_path,
        checkpoint_every,
        context,
        options,
        config,
        start_time,
    )

    try:
        result = differential_evolution(
            func=_objective_for_worker,
            bounds=adjustable_bounds(),
            strategy="best1bin",
            maxiter=maxiter,
            popsize=popsize,
            tol=1e-5,
            mutation=(0.5, 1.5),
            recombination=0.7,
            seed=seed,
            workers=map_workers,
            disp=disp,
            polish=polish,
            init="latinhypercube",
            atol=0,
            updating="deferred",
            callback=callback,
        )
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    elapsed = time.time() - start_time
    payload = build_result_payload(
        np.asarray(result.x, dtype=float),
        float(result.fun),
        context,
        options,
        config,
        elapsed_seconds=elapsed,
        iteration=int(getattr(result, "nit", maxiter)),
        nfev=int(result.nfev),
        message=str(result.message),
    )
    save_result_json(output_path / "best_result.json", payload)
    save_result_text(output_path / "best_result.txt", payload)

    print("\n优化完成")
    print(f"目标函数值: {float(result.fun):.8f}")
    print(f"耗时: {elapsed:.1f} 秒 ({elapsed / 60:.1f} 分钟)")
    print(f"评估次数: {result.nfev}")
    print(f"结果已保存到: {output_path}")

    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="X5 差分进化 softmax 报价优化器")
    parser.add_argument("--samples", type=int, default=8192, help="CRN 环境样本数")
    parser.add_argument("--maxiter", type=int, default=800, help="DE 最大迭代代数")
    parser.add_argument("--popsize", type=int, default=90, help="DE 种群倍率")
    parser.add_argument("--workers", type=int, default=-1, help="-1 表示最多使用全部 CPU")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--checkpoint-every", type=int, default=50, help="每多少代保存一次 checkpoint")
    parser.add_argument("--output-dir", type=Path, default=Path("de_softmax_results"), help="结果输出目录")
    parser.add_argument("--price-scale", type=float, default=0.25, help="报价距离惩罚 scale")
    parser.add_argument("--score-scale", type=float, default=2.0, help="清标分距离惩罚 scale")
    parser.add_argument("--temperature", type=float, default=0.1, help="环节五 softmax 温度")
    parser.add_argument("--invalid-cost", type=float, default=10.0, help="X_i <= K 的 softmax 额外 cost")
    parser.add_argument("--no-polish", action="store_true", help="关闭 scipy 最后局部 polish")
    parser.add_argument("--quiet", action="store_true", help="关闭 scipy 每代输出")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    optimize_with_differential_evolution(
        n_samples=args.samples,
        maxiter=args.maxiter,
        popsize=args.popsize,
        workers=args.workers,
        seed=args.seed,
        checkpoint_every=args.checkpoint_every,
        output_dir=args.output_dir,
        price_scale=args.price_scale,
        score_scale=args.score_scale,
        temperature=args.temperature,
        invalid_cost=args.invalid_cost,
        polish=not args.no_polish,
        disp=not args.quiet,
    )


if __name__ == "__main__":
    mp.freeze_support()
    main()

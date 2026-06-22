"""
Validate Differential Evolution bid candidates under the current qingbiao rules.

The optimizer's objective is a surrogate loss. This script estimates the true
winning probability for X5 by sampling the 8 non-adjustable bid variables and
exactly enumerating all 324 discrete rule scenarios for each sample.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import sys
import time
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
from scipy.stats import qmc

import de_softmax_optimizer as optimizer

DEFAULT_SAMPLES = 32768
DEFAULT_REFINE_SAMPLES = 65536
DEFAULT_TOP = 20
DEFAULT_REFINE_TOP = 5
DEFAULT_SEED = 42
DEFAULT_OUTPUT_DIR = Path("de_softmax_validation_results")
WINDOWS_MAX_POOL_WORKERS = 60
OUTPUT_DECIMAL_PLACES = 2

Q_VALUES = [int(value) for value in optimizer.Q_VALUES]
B2_VALUES = [float(value) for value in optimizer.B2_VALUES]
EXCLUDE_VALUES = [int(value) for value in optimizer.EXCLUDE_LOWEST_PRICE_VALUES]
TARGET_N_VALUES = [int(value) for value in optimizer.TARGET_N_VALUES]
K2_VALUES = [float(value) for value in optimizer.K2_VALUES]
DISCRETE_SCENARIO_COUNT = (
    len(Q_VALUES)
    * len(B2_VALUES)
    * len(EXCLUDE_VALUES)
    * len(TARGET_N_VALUES)
    * len(K2_VALUES)
)

STATUS_ENTRY = "环节一：入围淘汰"
STATUS_RECOMMENDATION = "环节二：推优淘汰"
STATUS_EXCLUDED = "环节四：报价最低剔除"
STATUS_SCORE = "环节四：清标得分不足"
STATUS_NOT_WINNER = "环节五：未中标"
STATUS_WINNER = "中标"


@dataclass(frozen=True)
class RuleScenario:
    q: int
    b2: float
    exclude_lowest_price_count: int
    target_n: int
    k2: float


@dataclass(frozen=True)
class ScenarioOutcome:
    winner: int | None
    target_status: str


@dataclass(frozen=True)
class Candidate:
    name: str
    source_path: Path
    iteration: int | None
    objective_value: float
    decision_vars: np.ndarray
    adjustable_bids: dict[str, float]
    payload: dict[str, Any]


@dataclass
class PartialStats:
    samples: int = 0
    total_scenarios: int = 0
    target_wins: int = 0
    env_probability_sum: float = 0.0
    env_probability_sq_sum: float = 0.0
    winner_counts: Counter[str] = field(default_factory=Counter)
    status_counts: Counter[str] = field(default_factory=Counter)
    q_wins: Counter[str] = field(default_factory=Counter)
    b2_wins: Counter[str] = field(default_factory=Counter)
    exclude_wins: Counter[str] = field(default_factory=Counter)
    target_n_wins: Counter[str] = field(default_factory=Counter)
    k2_wins: Counter[str] = field(default_factory=Counter)

    def merge(self, other: PartialStats) -> None:
        self.samples += other.samples
        self.total_scenarios += other.total_scenarios
        self.target_wins += other.target_wins
        self.env_probability_sum += other.env_probability_sum
        self.env_probability_sq_sum += other.env_probability_sq_sum
        self.winner_counts.update(other.winner_counts)
        self.status_counts.update(other.status_counts)
        self.q_wins.update(other.q_wins)
        self.b2_wins.update(other.b2_wins)
        self.exclude_wins.update(other.exclude_wins)
        self.target_n_wins.update(other.target_n_wins)
        self.k2_wins.update(other.k2_wins)


@dataclass(frozen=True)
class ValidationResult:
    phase: str
    candidate_name: str
    source_path: str
    iteration: int | None
    objective_value: float
    samples: int
    total_scenarios: int
    target_wins: int
    target_win_probability: float
    standard_error: float
    ci95_half_width: float
    winner_distribution: dict[str, dict[str, float]]
    status_distribution: dict[str, dict[str, float]]
    marginal_probabilities: dict[str, dict[str, dict[str, float]]]
    adjustable_bids: dict[str, float]


def scenario_grid() -> list[RuleScenario]:
    return [
        RuleScenario(q, b2, exclude_count, target_n, k2)
        for q in Q_VALUES
        for b2 in B2_VALUES
        for exclude_count in EXCLUDE_VALUES
        for target_n in TARGET_N_VALUES
        for k2 in K2_VALUES
    ]


SCENARIOS = scenario_grid()


def as_jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Counter):
        return dict(value)
    if isinstance(value, dict):
        return {str(key): as_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [as_jsonable(item) for item in value]
    return value


def unit_key(unit: int) -> str:
    return f"X{unit}"


def winner_key(winner: int | None) -> str:
    return "无中标人" if winner is None else unit_key(winner)


def value_key(value: float | int) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def round_output_decimal(value: float) -> float:
    return round(float(value), OUTPUT_DECIMAL_PLACES)


def format_output_decimal(value: float) -> str:
    return f"{float(value):.{OUTPUT_DECIMAL_PLACES}f}"


def format_output_percent(value: float) -> str:
    return f"{float(value):.{OUTPUT_DECIMAL_PLACES}%}"


def is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def total_scenarios_for_samples(samples: int) -> int:
    return int(samples) * DISCRETE_SCENARIO_COUNT


def round_half_up(value: float) -> int:
    return int(math.floor(float(value) + 0.5))


def rounded_integer_mean_half_up_places(values: list[int], places: int = 2) -> float:
    multiplier = 10**places
    numerator = sum(values) * multiplier
    denominator = len(values)
    rounded_units = (2 * numerator + denominator) // (2 * denominator)
    return rounded_units / multiplier


def lower_quote_key(unit: int, bids: np.ndarray) -> tuple[float, int]:
    return (-float(bids[unit]), unit)


def higher_quote_key(unit: int, bids: np.ndarray) -> tuple[float, int]:
    return (float(bids[unit]), unit)


def score_key(unit: int, scores: dict[int, float]) -> tuple[float, int]:
    return (-scores[unit], unit)


def final_stage_selection_key(unit: int, bids: np.ndarray, final_k: float) -> tuple[int, float, int]:
    gap = float(bids[unit]) - final_k
    if gap > 0:
        return (0, gap, unit)
    return (1, abs(gap), unit)


def final_stage_winner(finalists: list[int], bids: np.ndarray, final_k: float) -> int | None:
    if not finalists:
        return None
    return sorted(finalists, key=lambda unit: final_stage_selection_key(unit, bids, final_k))[0]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return data


def decision_vars_from_payload(payload: dict[str, Any], path: Path) -> tuple[np.ndarray, dict[str, float]]:
    bids = payload.get("best_adjustable_bids")
    if not isinstance(bids, dict):
        raise ValueError(f"{path} is missing best_adjustable_bids.")

    adjustable_bids: dict[str, float] = {}
    values = []
    for unit in optimizer.ADJUSTABLE_UNITS:
        key = unit_key(unit)
        if key not in bids:
            raise ValueError(f"{path} is missing {key} in best_adjustable_bids.")
        value = float(bids[key])
        adjustable_bids[key] = value
        values.append(value)
    return np.asarray(values, dtype=float), adjustable_bids


def candidate_from_file(path: Path) -> Candidate:
    payload = load_json(path)
    decision_vars, adjustable_bids = decision_vars_from_payload(payload, path)
    objective_value = float(payload.get("objective_value", math.inf))
    iteration_raw = payload.get("iteration")
    iteration = None if iteration_raw is None else int(iteration_raw)
    return Candidate(
        name=path.stem,
        source_path=path,
        iteration=iteration,
        objective_value=objective_value,
        decision_vars=decision_vars,
        adjustable_bids=adjustable_bids,
        payload=payload,
    )


def load_candidates(result_dir: Path) -> list[Candidate]:
    paths: list[Path] = []
    best_path = result_dir / "best_result.json"
    if best_path.exists():
        paths.append(best_path)
    paths.extend(sorted(result_dir.glob("checkpoint_iter_*.json")))

    if not paths:
        raise FileNotFoundError(
            f"No best_result.json or checkpoint_iter_*.json files found in {result_dir}."
        )

    candidates = [candidate_from_file(path) for path in paths]
    candidates.sort(key=lambda item: (item.objective_value, item.name))
    return candidates


def select_top_candidates(candidates: list[Candidate], top: int) -> list[Candidate]:
    if top == 0:
        return list(candidates)
    if top < 0:
        raise ValueError("--top must be non-negative; use 0 for all candidates.")
    return list(candidates[:top])


def generate_env_matrix(samples: int, seed: int) -> np.ndarray:
    if samples <= 0:
        raise ValueError("samples must be a positive integer.")

    env_units = optimizer.NON_ADJUSTABLE_UNITS
    sampler = qmc.Sobol(d=len(env_units), scramble=True, seed=seed)
    if is_power_of_two(samples):
        samples_01 = sampler.random_base2(m=int(math.log2(samples)))
    else:
        samples_01 = sampler.random(samples)

    env_matrix = np.zeros((samples, len(env_units)), dtype=float)
    for col, unit in enumerate(env_units):
        lower, upper = optimizer.BID_BOUNDS[unit]
        env_matrix[:, col] = lower + samples_01[:, col] * (upper - lower)
    return env_matrix


def build_bid_vector(decision_vars: Iterable[float], env_vars: Iterable[float]) -> np.ndarray:
    bids = np.zeros(optimizer.NUM_UNITS + 1, dtype=float)
    for unit, value in zip(optimizer.ADJUSTABLE_UNITS, decision_vars):
        bids[unit] = float(value)
    for unit, value in zip(optimizer.NON_ADJUSTABLE_UNITS, env_vars):
        bids[unit] = float(value)
    return bids


def calculate_business_baseline(qualified: list[int], bids: np.ndarray) -> float:
    quote_low_to_high = sorted(qualified, key=lambda unit: lower_quote_key(unit, bids))
    num_to_remove = round_half_up(len(qualified) * 0.15)
    if num_to_remove == 0:
        middle_units = quote_low_to_high
    else:
        middle_units = quote_low_to_high[num_to_remove:-num_to_remove]

    rounded_unique_values = sorted({round_half_up(float(bids[unit])) for unit in middle_units})
    return rounded_integer_mean_half_up_places(rounded_unique_values, 2)


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
    values = optimizer.A_SCORES[indices]
    min_value = float(values.min())
    max_value = float(values.max())
    if max_value == min_value:
        return {unit: 0.0 for unit in recommended}

    return {
        unit: 10.0 * (float(optimizer.A_SCORES[unit - 1]) - min_value) / (max_value - min_value)
        for unit in recommended
    }


def set_status_once(current_status: str | None, new_status: str) -> str | None:
    return current_status if current_status is not None else new_status


def evaluate_rule_scenario(
    bids: np.ndarray,
    scenario: RuleScenario,
    target_unit: int = optimizer.TARGET_UNIT,
) -> ScenarioOutcome:
    target_status: str | None = None
    q1 = round_half_up(scenario.q / 2)

    qualified = sorted(
        range(1, optimizer.NUM_UNITS + 1),
        key=lambda unit: (int(optimizer.ENTRY_RANKINGS[unit - 1]), unit),
    )[: scenario.q]

    if target_unit not in qualified:
        target_status = set_status_once(target_status, STATUS_ENTRY)

    ordered_low_price = sorted(qualified, key=lambda unit: lower_quote_key(unit, bids))
    blocked_low_price_units = ordered_low_price[:2]
    recommend_eligible = [
        unit for unit in qualified if unit not in set(blocked_low_price_units)
    ]

    tech_recommended = sorted(
        recommend_eligible,
        key=lambda unit: (int(optimizer.TECH_RANKINGS[unit - 1]), unit),
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

    if target_unit not in recommended:
        target_status = set_status_once(target_status, STATUS_RECOMMENDATION)

    if not recommended:
        return ScenarioOutcome(winner=None, target_status=target_status or STATUS_RECOMMENDATION)

    b_value = calculate_b_value(recommended, bids, scenario.b2)
    bid_scores = calculate_bid_scores(recommended, bids, b_value)
    fulfillment_scores = calculate_fulfillment_scores(recommended)
    total_scores = {
        unit: 50.0
        + fulfillment_scores[unit]
        + float(optimizer.PERFORMANCE_SCORES[unit - 1])
        + bid_scores[unit]
        for unit in recommended
    }

    excluded_lowest_price = sorted(
        recommended,
        key=lambda unit: lower_quote_key(unit, bids),
    )[: scenario.exclude_lowest_price_count]
    candidates = [unit for unit in recommended if unit not in set(excluded_lowest_price)]
    finalists_ordered = sorted(candidates, key=lambda unit: score_key(unit, total_scores))
    finalists = finalists_ordered[: scenario.target_n]

    if target_unit in excluded_lowest_price:
        target_status = set_status_once(target_status, STATUS_EXCLUDED)
    elif target_unit in candidates and target_unit not in finalists:
        target_status = set_status_once(target_status, STATUS_SCORE)

    if not finalists:
        return ScenarioOutcome(winner=None, target_status=target_status or STATUS_SCORE)

    final_k1 = mean(float(bids[unit]) for unit in finalists)
    final_k = float(final_k1 + scenario.k2)
    final_winner = final_stage_winner(finalists, bids, final_k)

    if final_winner == target_unit:
        target_status = STATUS_WINNER
    elif target_unit in finalists:
        target_status = set_status_once(target_status, STATUS_NOT_WINNER)

    return ScenarioOutcome(
        winner=final_winner,
        target_status=target_status or STATUS_NOT_WINNER,
    )


def evaluate_candidate_chunk(args: tuple[int, np.ndarray, np.ndarray]) -> tuple[int, PartialStats]:
    candidate_index, decision_vars, env_chunk = args
    partial = PartialStats(samples=int(env_chunk.shape[0]))

    for env_vars in env_chunk:
        bids = build_bid_vector(decision_vars, env_vars)
        env_wins = 0
        for scenario in SCENARIOS:
            outcome = evaluate_rule_scenario(bids, scenario)
            partial.total_scenarios += 1
            partial.winner_counts[winner_key(outcome.winner)] += 1
            partial.status_counts[outcome.target_status] += 1
            if outcome.winner == optimizer.TARGET_UNIT:
                partial.target_wins += 1
                env_wins += 1
                partial.q_wins[value_key(scenario.q)] += 1
                partial.b2_wins[value_key(scenario.b2)] += 1
                partial.exclude_wins[value_key(scenario.exclude_lowest_price_count)] += 1
                partial.target_n_wins[value_key(scenario.target_n)] += 1
                partial.k2_wins[value_key(scenario.k2)] += 1

        env_probability = env_wins / DISCRETE_SCENARIO_COUNT
        partial.env_probability_sum += env_probability
        partial.env_probability_sq_sum += env_probability * env_probability

    return candidate_index, partial


def distribution(
    counter: Counter[str],
    total: int,
    ordered_keys: Sequence[str] | None = None,
) -> dict[str, dict[str, float]]:
    keys = ordered_keys if ordered_keys is not None else sorted(counter.keys())
    return {
        key: {
            "count": int(counter.get(key, 0)),
            "rate": float(counter.get(key, 0) / total) if total else 0.0,
        }
        for key in keys
        if counter.get(key, 0) or ordered_keys is not None
    }


def marginal_distribution(
    wins: Counter[str],
    values: Sequence[int | float],
    denominator_per_value: int,
) -> dict[str, dict[str, float]]:
    return {
        value_key(value): {
            "wins": int(wins.get(value_key(value), 0)),
            "total": int(denominator_per_value),
            "win_probability": (
                float(wins.get(value_key(value), 0) / denominator_per_value)
                if denominator_per_value
                else 0.0
            ),
        }
        for value in values
    }


def finalize_result(candidate: Candidate, phase: str, stats: PartialStats) -> ValidationResult:
    probability = stats.target_wins / stats.total_scenarios if stats.total_scenarios else 0.0
    if stats.samples > 1:
        mean_probability = stats.env_probability_sum / stats.samples
        variance = (
            stats.env_probability_sq_sum
            - stats.samples * mean_probability * mean_probability
        ) / (stats.samples - 1)
        standard_error = math.sqrt(max(0.0, variance) / stats.samples)
    else:
        standard_error = 0.0

    marginal_probabilities = {
        "q": marginal_distribution(
            stats.q_wins,
            Q_VALUES,
            stats.samples
            * len(B2_VALUES)
            * len(EXCLUDE_VALUES)
            * len(TARGET_N_VALUES)
            * len(K2_VALUES),
        ),
        "b2": marginal_distribution(
            stats.b2_wins,
            B2_VALUES,
            stats.samples
            * len(Q_VALUES)
            * len(EXCLUDE_VALUES)
            * len(TARGET_N_VALUES)
            * len(K2_VALUES),
        ),
        "exclude_lowest_price_count": marginal_distribution(
            stats.exclude_wins,
            EXCLUDE_VALUES,
            stats.samples
            * len(Q_VALUES)
            * len(B2_VALUES)
            * len(TARGET_N_VALUES)
            * len(K2_VALUES),
        ),
        "target_n": marginal_distribution(
            stats.target_n_wins,
            TARGET_N_VALUES,
            stats.samples
            * len(Q_VALUES)
            * len(B2_VALUES)
            * len(EXCLUDE_VALUES)
            * len(K2_VALUES),
        ),
        "k2": marginal_distribution(
            stats.k2_wins,
            K2_VALUES,
            stats.samples
            * len(Q_VALUES)
            * len(B2_VALUES)
            * len(EXCLUDE_VALUES)
            * len(TARGET_N_VALUES),
        ),
    }

    return ValidationResult(
        phase=phase,
        candidate_name=candidate.name,
        source_path=str(candidate.source_path),
        iteration=candidate.iteration,
        objective_value=candidate.objective_value,
        samples=stats.samples,
        total_scenarios=stats.total_scenarios,
        target_wins=stats.target_wins,
        target_win_probability=probability,
        standard_error=standard_error,
        ci95_half_width=1.96 * standard_error,
        winner_distribution=distribution(
            stats.winner_counts,
            stats.total_scenarios,
            ordered_keys=[unit_key(unit) for unit in range(1, optimizer.NUM_UNITS + 1)]
            + ["无中标人"],
        ),
        status_distribution=distribution(stats.status_counts, stats.total_scenarios),
        marginal_probabilities=marginal_probabilities,
        adjustable_bids=dict(candidate.adjustable_bids),
    )


def chunk_env_matrix(env_matrix: np.ndarray, chunk_size: int) -> list[np.ndarray]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    return [
        env_matrix[start : start + chunk_size]
        for start in range(0, env_matrix.shape[0], chunk_size)
    ]


def resolve_worker_count(workers: int, task_count: int, phase: str) -> int:
    requested_workers = min(mp.cpu_count(), task_count) if workers == -1 else int(workers)
    requested_workers = min(requested_workers, task_count)
    if sys.platform == "win32" and requested_workers > WINDOWS_MAX_POOL_WORKERS:
        print(
            f"[{phase}] Windows worker cap: requested {requested_workers}, "
            f"using {WINDOWS_MAX_POOL_WORKERS} to avoid multiprocessing handle limits"
        )
        return WINDOWS_MAX_POOL_WORKERS
    return requested_workers


def evaluate_candidates(
    candidates: list[Candidate],
    samples: int,
    seed: int,
    phase: str,
    workers: int,
    chunk_size: int,
) -> list[ValidationResult]:
    if not candidates:
        return []
    if workers == 0 or workers < -1:
        raise ValueError("workers must be -1 or a positive integer.")

    env_matrix = generate_env_matrix(samples, seed)
    chunks = chunk_env_matrix(env_matrix, chunk_size)
    tasks: list[tuple[int, np.ndarray, np.ndarray]] = []
    for candidate_index, candidate in enumerate(candidates):
        for chunk in chunks:
            tasks.append((candidate_index, candidate.decision_vars, chunk))

    print(
        f"[{phase}] candidates={len(candidates)}, samples={samples}, "
        f"discrete={DISCRETE_SCENARIO_COUNT}, tasks={len(tasks)}"
    )
    started_at = time.time()
    stats_by_candidate = [PartialStats() for _ in candidates]

    if workers == 1:
        for task_index, task in enumerate(tasks, start=1):
            candidate_index, partial = evaluate_candidate_chunk(task)
            stats_by_candidate[candidate_index].merge(partial)
            if task_index % max(1, len(tasks) // 10) == 0 or task_index == len(tasks):
                print(f"[{phase}] progress {task_index}/{len(tasks)} tasks")
    else:
        actual_workers = resolve_worker_count(workers, len(tasks), phase)
        with mp.Pool(processes=actual_workers) as pool:
            for task_index, (candidate_index, partial) in enumerate(
                pool.imap_unordered(evaluate_candidate_chunk, tasks),
                start=1,
            ):
                stats_by_candidate[candidate_index].merge(partial)
                if task_index % max(1, len(tasks) // 10) == 0 or task_index == len(tasks):
                    print(f"[{phase}] progress {task_index}/{len(tasks)} tasks")

    elapsed = time.time() - started_at
    print(f"[{phase}] completed in {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    return [
        finalize_result(candidate, phase, stats)
        for candidate, stats in zip(candidates, stats_by_candidate)
    ]


def choose_refinement_candidates(
    candidates: list[Candidate],
    first_pass_results: list[ValidationResult],
    refine_top: int,
) -> list[Candidate]:
    if refine_top <= 0:
        return []

    by_name = {candidate.name: candidate for candidate in candidates}
    ordered = sorted(
        first_pass_results,
        key=lambda item: (
            -item.target_win_probability,
            item.standard_error,
            item.objective_value,
            item.candidate_name,
        ),
    )
    return [by_name[result.candidate_name] for result in ordered[:refine_top]]


def result_to_row(result: ValidationResult) -> dict[str, Any]:
    row: dict[str, Any] = {
        "phase": result.phase,
        "candidate": result.candidate_name,
        "source_file": result.source_path,
        "iteration": "" if result.iteration is None else result.iteration,
        "surrogate_objective": result.objective_value,
        "samples": result.samples,
        "total_scenarios": result.total_scenarios,
        "x5_wins": result.target_wins,
        "x5_win_probability": format_output_decimal(result.target_win_probability),
        "standard_error": result.standard_error,
        "ci95_half_width": result.ci95_half_width,
    }

    for key, value in result.adjustable_bids.items():
        row[f"bid_{key}"] = format_output_decimal(value)

    for dimension, values in result.marginal_probabilities.items():
        for option_value, stats in values.items():
            row[f"{dimension}_{option_value}_win_probability"] = format_output_decimal(stats["win_probability"])

    for winner, stats in result.winner_distribution.items():
        row[f"winner_{winner}_rate"] = stats["rate"]

    for status, stats in result.status_distribution.items():
        row[f"status_{status}_rate"] = stats["rate"]

    return row


def write_csv(path: Path, results: list[ValidationResult]) -> None:
    rows = [result_to_row(result) for result in results]
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(as_jsonable(payload), file, ensure_ascii=False, indent=2)


def result_to_json_dict(result: ValidationResult) -> dict[str, Any]:
    payload = dict(result.__dict__)
    payload["target_win_probability"] = round_output_decimal(result.target_win_probability)
    payload["adjustable_bids"] = {
        key: round_output_decimal(value)
        for key, value in result.adjustable_bids.items()
    }
    payload["marginal_probabilities"] = {
        dimension: {
            option_value: {
                **stats,
                "win_probability": round_output_decimal(stats["win_probability"]),
            }
            for option_value, stats in values.items()
        }
        for dimension, values in result.marginal_probabilities.items()
    }
    return payload


def final_results_by_candidate(results: list[ValidationResult]) -> list[ValidationResult]:
    latest_by_candidate: dict[str, ValidationResult] = {}
    for result in results:
        if result.phase == "refine" or result.candidate_name not in latest_by_candidate:
            latest_by_candidate[result.candidate_name] = result

    return sorted(
        latest_by_candidate.values(),
        key=lambda item: (
            -item.target_win_probability,
            item.standard_error,
            item.objective_value,
            item.candidate_name,
        ),
    )


def write_report(path: Path, results: list[ValidationResult], config: dict[str, Any]) -> None:
    final_results = final_results_by_candidate(results)
    with path.open("w", encoding="utf-8") as file:
        file.write("DE 报价候选真实中标概率验证报告\n")
        file.write("=" * 72 + "\n")
        file.write(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        file.write(f"初筛样本数: {config['samples']}\n")
        file.write(f"复验Top: {config['refine_top']}\n")
        file.write(f"复验样本数: {config['refine_samples']}\n")
        file.write(f"每个连续样本离散场景数: {DISCRETE_SCENARIO_COUNT}\n")
        file.write("\n最终排序（优先使用复验结果）\n")
        file.write("-" * 72 + "\n")
        for rank, result in enumerate(final_results, start=1):
            ci = result.ci95_half_width
            file.write(
                f"{rank:>2}. {result.candidate_name:<24} "
                f"phase={result.phase:<7} "
                f"P(X5中标)={format_output_percent(result.target_win_probability)} "
                f"95%CI约±{format_output_percent(ci)} "
                f"samples={result.samples} "
                f"objective={result.objective_value:.8f}\n"
            )

        if final_results:
            best = final_results[0]
            file.write("\n最佳候选可调报价\n")
            file.write("-" * 72 + "\n")
            for key, value in best.adjustable_bids.items():
                file.write(f"  {key:<4} = {format_output_decimal(value)}\n")

            file.write("\n最佳候选 X5 状态分布\n")
            file.write("-" * 72 + "\n")
            for status, stats in sorted(
                best.status_distribution.items(),
                key=lambda item: (-item[1]["count"], item[0]),
            ):
                file.write(
                    f"  {status:<20} {int(stats['count']):>10} "
                    f"({format_output_percent(stats['rate'])})\n"
                )

            file.write("\n最佳候选中标人分布（非零项）\n")
            file.write("-" * 72 + "\n")
            for winner, stats in sorted(
                best.winner_distribution.items(),
                key=lambda item: (-item[1]["count"], item[0]),
            ):
                if stats["count"] <= 0:
                    continue
                file.write(
                    f"  {winner:<10} {int(stats['count']):>10} "
                    f"({format_output_percent(stats['rate'])})\n"
                )


def run_validation(args: argparse.Namespace) -> list[ValidationResult]:
    result_dir = Path(args.result_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates = select_top_candidates(load_candidates(result_dir), args.top)
    print(f"Loaded {len(candidates)} candidate(s) from {result_dir}")

    first_pass = evaluate_candidates(
        candidates=candidates,
        samples=args.samples,
        seed=args.seed,
        phase="initial",
        workers=args.workers,
        chunk_size=args.chunk_size,
    )

    refine_candidates = choose_refinement_candidates(candidates, first_pass, args.refine_top)
    refine_results = evaluate_candidates(
        candidates=refine_candidates,
        samples=args.refine_samples,
        seed=args.seed,
        phase="refine",
        workers=args.workers,
        chunk_size=args.chunk_size,
    )

    all_results = first_pass + refine_results
    config = {
        "result_dir": str(result_dir),
        "output_dir": str(output_dir),
        "samples": args.samples,
        "top": args.top,
        "refine_top": args.refine_top,
        "refine_samples": args.refine_samples,
        "seed": args.seed,
        "workers": args.workers,
        "chunk_size": args.chunk_size,
        "discrete_scenarios": DISCRETE_SCENARIO_COUNT,
        "continuous_env_units": [unit_key(unit) for unit in optimizer.NON_ADJUSTABLE_UNITS],
        "target_unit": unit_key(optimizer.TARGET_UNIT),
    }

    csv_path = output_dir / "validation_summary.csv"
    json_path = output_dir / "validation_results.json"
    report_path = output_dir / "validation_report.txt"
    write_csv(csv_path, all_results)
    write_json(
        json_path,
        {
            "config": config,
            "results": [result_to_json_dict(result) for result in all_results],
            "final_ranking": [
                result_to_json_dict(result)
                for result in final_results_by_candidate(all_results)
            ],
        },
    )
    write_report(report_path, all_results, config)

    print(f"Saved CSV: {csv_path}")
    print(f"Saved JSON: {json_path}")
    print(f"Saved report: {report_path}")
    return all_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate DE bid candidates under current rules.")
    parser.add_argument("--result-dir", type=Path, default=Path("de_softmax_results"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--top", type=int, default=DEFAULT_TOP, help="0 means all candidates.")
    parser.add_argument("--refine-top", type=int, default=DEFAULT_REFINE_TOP)
    parser.add_argument("--refine-samples", type=int, default=DEFAULT_REFINE_SAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--workers",
        type=int,
        default=-1,
        help="-1 uses all available CPUs; capped at 60 on Windows.",
    )
    parser.add_argument("--chunk-size", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_validation(args)


if __name__ == "__main__":
    mp.freeze_support()
    main()

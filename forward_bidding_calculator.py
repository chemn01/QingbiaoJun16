"""
当前规则正向计算器 (forward_bidding_calculator.py)

功能：
1. 给定 20 家单位的报价下浮率 X_1 ... X_20，按 qingbiao.md 的五个环节正向计算。
2. 输出类似“战报”的详细过程，重点追踪目标单位 X5 在哪个环节胜出或被淘汰。
3. 支持随机单次模拟、手动/JSON 输入报价、以及枚举全部离散随机场景计算 X5 中标概率。

说明：
- X_i 是下浮率，不是真实报价。X_i 越大，真实报价越低。
- “报价最低/报价低于 K”在程序中按“下浮率更高/下浮率 > K”处理。
- 环节五按用户确认的规则实现：只有 X_i > K 的定标候选人参与中标选择。
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from statistics import mean

# ===================== 基础数据：来自 qingbiao.md =====================

NUM_UNITS = 20
TARGET_UNIT = 5
LOG_DIR = Path("simulation_logs")

A_SCORES = [
    86.09, 82.15, 82.02, 81.70, 79.57,
    80.76, 81.23, 80.20, 79.86, 81.16,
    79.92, 75.50, 81.19, 75.00, 75.00,
    75.00, 77.35, 75.00, 87.63, 75.00,
]

ENTRY_RANKINGS = [
    1, 2, 3, 4, 5,
    6, 7, 8, 9, 10,
    11, 12, 13, 14, 15,
    16, 17, 18, 19, 20,
]

TECH_RANKINGS = [
    1, 3, 9, 2, 4,
    5, 8, 7, 10, 11,
    6, 20, 20, 20, 20,
    12, 20, 20, 20, 20,
]

PERFORMANCE_SCORES = [
    20, 20, 20, 20, 20,
    20, 20, 20, 20, 20,
    20, 20, 10, 20, 20,
    0, 0, 20, 0, 20,
]

BID_BOUNDS = {
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

Q_VALUES = [15, 16, 17, 18, 19, 20]
B2_VALUES = [0.5, 1.0, 1.5]
EXCLUDE_LOWEST_PRICE_VALUES = [1, 2]
TARGET_N_VALUES = [3, 4, 5]
K2_VALUES = [0.0, 0.25, 0.5]


@dataclass(frozen=True)
class Scenario:
    """一组离散随机参数。"""

    q: int
    b2: float
    exclude_lowest_price_count: int
    target_n: int
    k2: float


@dataclass
class SimulationResult:
    """单次正向计算结果。"""

    logs: list[str]
    winner: int | None
    target_status: str
    target_reason: str
    scenario: Scenario


def unit_name(unit: int) -> str:
    return f"单位{unit}"


def x_name(unit: int) -> str:
    return f"X{unit}"


def unit_label(unit: int) -> str:
    return f"{unit_name(unit)}({x_name(unit)})"


def round_half_up(value: float) -> int:
    """正数四舍五入取整。"""

    return int(math.floor(value + 0.5))


def round_half_up_places(value: float, places: int = 2) -> float:
    """四舍五入到指定小数位。"""

    quant = Decimal("1").scaleb(-places)
    return float(Decimal(str(value)).quantize(quant, rounding=ROUND_HALF_UP))


def all_units() -> list[int]:
    return list(range(1, NUM_UNITS + 1))


def format_units(units: list[int]) -> str:
    if not units:
        return "无"
    return ", ".join(unit_label(unit) for unit in units)


def entry_rank(unit: int) -> int:
    return ENTRY_RANKINGS[unit - 1]


def tech_rank(unit: int) -> int:
    return TECH_RANKINGS[unit - 1]


def a_score(unit: int) -> float:
    return A_SCORES[unit - 1]


def performance_score(unit: int) -> float:
    return float(PERFORMANCE_SCORES[unit - 1])


def lower_quote_key(unit: int, bids: list[float]) -> tuple[float, int]:
    """真实报价从低到高排序：下浮率越大，真实报价越低。"""

    return (-float(bids[unit]), unit)


def higher_quote_key(unit: int, bids: list[float]) -> tuple[float, int]:
    """真实报价从高到低排序：下浮率越小，真实报价越高。"""

    return (float(bids[unit]), unit)


def score_key(unit: int, scores: dict[int, float]) -> tuple[float, int]:
    """清标得分从高到低排序。得分完全相同时按单位编号稳定排序。"""

    return (-scores[unit], unit)


def lowest_price_units(
    units: list[int],
    bids: list[float],
    count: int,
) -> list[int]:
    """真实报价最低的若干单位，也就是下浮率最高的若干单位。"""

    return sorted(units, key=lambda unit: lower_quote_key(unit, bids))[:count]


def highest_price_units(
    units: list[int],
    bids: list[float],
    count: int,
) -> list[int]:
    """真实报价最高的若干单位，也就是下浮率最低的若干单位。"""

    return sorted(units, key=lambda unit: higher_quote_key(unit, bids))[:count]


def validate_bids(raw_bids: list[float]) -> list[float]:
    if len(raw_bids) != NUM_UNITS:
        raise ValueError(f"必须提供 {NUM_UNITS} 个报价下浮率，当前为 {len(raw_bids)} 个。")

    bids: list[float] = [math.nan] + [float(value) for value in raw_bids]
    return bids


def bid_bound_warnings(bids: list[float]) -> list[str]:
    warnings: list[str] = []
    for unit in all_units():
        lower, upper = BID_BOUNDS[unit]
        value = float(bids[unit])
        if value < lower or value > upper:
            warnings.append(
                f"[提示] {unit_label(unit)}={value:.2f} 超出已知范围 [{lower:g}, {upper:g}]。"
            )
    return warnings


def random_bids(rng: random.Random) -> list[float]:
    values = []
    for unit in all_units():
        lower, upper = BID_BOUNDS[unit]
        values.append(rng.uniform(lower, upper))
    return validate_bids(values)


def midpoint_bids() -> list[float]:
    values = [(lower + upper) / 2 for lower, upper in BID_BOUNDS.values()]
    return validate_bids(values)


def random_scenario(rng: random.Random) -> Scenario:
    return Scenario(
        q=rng.choice(Q_VALUES),
        b2=rng.choice(B2_VALUES),
        exclude_lowest_price_count=rng.choice(EXCLUDE_LOWEST_PRICE_VALUES),
        target_n=rng.choice(TARGET_N_VALUES),
        k2=rng.choice(K2_VALUES),
    )


def scenario_display(scenario: Scenario) -> str:
    q1 = round_half_up(scenario.q / 2)
    return (
        f"Q={scenario.q}, Q1={q1}, B2={scenario.b2:g}, "
        f"剔除报价最低人数={scenario.exclude_lowest_price_count}, "
        f"N={scenario.target_n}, K2={scenario.k2:g}"
    )


def calculate_business_baseline(
    qualified: list[int],
    bids: list[float],
) -> tuple[float, list[int], list[int]]:
    """环节二商务推优的评标基准价 K1。"""

    quote_low_to_high = sorted(qualified, key=lambda unit: lower_quote_key(unit, bids))
    num_to_remove = round_half_up(len(qualified) * 0.15)
    if num_to_remove == 0:
        middle_units = quote_low_to_high
    else:
        middle_units = quote_low_to_high[num_to_remove:-num_to_remove]

    rounded_unique_values = sorted({round_half_up(float(bids[unit])) for unit in middle_units})
    baseline = round_half_up_places(mean(rounded_unique_values), 2)
    return baseline, middle_units, rounded_unique_values


def calculate_b_value(
    recommended: list[int],
    bids: list[float],
    b2: float,
) -> tuple[float, float, int]:
    """环节三 B 值：推荐名单中去掉报价最高者后取均值，再加 B2。"""

    removed_unit = highest_price_units(recommended, bids, 1)[0]
    remaining_units = [unit for unit in recommended if unit != removed_unit]
    if not remaining_units:
        b1 = float(bids[removed_unit])
    else:
        b1 = mean(float(bids[unit]) for unit in remaining_units)
    return b1 + b2, b1, removed_unit


def calculate_bid_scores(
    recommended: list[int],
    bids: list[float],
    b_value: float,
) -> dict[int, float]:
    """环节三总投标报价得分。"""

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
    values = [a_score(unit) for unit in recommended]
    min_value = min(values)
    max_value = max(values)
    if max_value == min_value:
        return {unit: 0.0 for unit in recommended}

    return {
        unit: 10.0 * (a_score(unit) - min_value) / (max_value - min_value)
        for unit in recommended
    }


def set_target_status_once(
    current_status: str | None,
    current_reason: str,
    status: str,
    reason: str,
) -> tuple[str, str]:
    if current_status is not None:
        return current_status, current_reason
    return status, reason


def simulate_and_log(
    bids: list[float],
    scenario: Scenario,
    target_unit: int = TARGET_UNIT,
) -> SimulationResult:
    logs: list[str] = []

    def log(message: str = "") -> None:
        logs.append(message)

    target_status: str | None = None
    target_reason = ""
    q1 = round_half_up(scenario.q / 2)

    log(f"【场景参数】{scenario_display(scenario)}")
    log(f"【目标单位】{unit_label(target_unit)}，下浮率={float(bids[target_unit]):.2f}")
    for warning in bid_bound_warnings(bids):
        log(warning)

    # =================== 环节一：入围 ===================
    log()
    log(f"--- 环节一：入围（前 {scenario.q} 名）---")
    qualified = sorted(all_units(), key=lambda unit: (entry_rank(unit), unit))[: scenario.q]
    log(f"入围名单: {format_units(qualified)}")

    if target_unit not in qualified:
        target_rank = entry_rank(target_unit)
        target_status, target_reason = set_target_status_once(
            target_status,
            target_reason,
            "环节一：入围淘汰",
            f"入围排名第 {target_rank}，未进入前 {scenario.q}",
        )
        log(f"[淘汰] {unit_label(target_unit)} {target_reason}。")
    else:
        log(f"[通过] {unit_label(target_unit)} 成功入围。")

    # =================== 环节二：推优 ===================
    log()
    log(f"--- 环节二：推优（技术/商务各 {q1} 名）---")
    blocked_low_price_units = lowest_price_units(qualified, bids, 2)
    recommend_eligible = [
        unit for unit in qualified if unit not in set(blocked_low_price_units)
    ]
    log(
        "不得推优名单（报价最低及次低，即下浮率最高的2家）: "
        f"{format_units(blocked_low_price_units)}"
    )

    tech_recommended = sorted(recommend_eligible, key=lambda unit: (tech_rank(unit), unit))[:q1]
    business_baseline, middle_units, rounded_unique_values = calculate_business_baseline(
        qualified,
        bids,
    )

    business_recommended = sorted(
        recommend_eligible,
        key=lambda unit: (
            abs(float(bids[unit]) - business_baseline),
            -float(bids[unit]),
            unit,
        ),
    )[:q1]

    recommended = sorted(set(tech_recommended) | set(business_recommended))
    log(f"技术推优名单: {format_units(tech_recommended)}")
    log(
        "商务评标基准价K1: "
        f"{business_baseline:.2f} "
        f"（中间单位: {format_units(middle_units)}；去重整数: {rounded_unique_values}）"
    )
    log(f"商务推优名单: {format_units(business_recommended)}")
    log(f"最终推优名单: {format_units(recommended)}")

    if target_unit in recommended and target_status is None:
        sources = []
        if target_unit in tech_recommended:
            sources.append("技术推优")
        if target_unit in business_recommended:
            sources.append("商务推优")
        log(f"[通过] {unit_label(target_unit)} 进入推优名单（{', '.join(sources)}）。")
    elif target_unit not in recommended and target_status is None:
        if target_unit in blocked_low_price_units:
            reason = "属于报价最低或次低单位，按基本推优原则不得被推荐"
        else:
            reason = "未进入技术推优或商务推优名单"
        target_status, target_reason = set_target_status_once(
            target_status,
            target_reason,
            "环节二：推优淘汰",
            reason,
        )
        log(f"[淘汰] {unit_label(target_unit)} {reason}。")

    if not recommended:
        log("[异常] 推优名单为空，后续环节无法计算。")
        return SimulationResult(
            logs=logs,
            winner=None,
            target_status=target_status or "环节二：推优名单为空",
            target_reason=target_reason or "推优名单为空",
            scenario=scenario,
        )

    # =================== 环节三：清标得分 ===================
    log()
    log("--- 环节三：清标得分 ---")
    b_value, b1, removed_for_b = calculate_b_value(recommended, bids, scenario.b2)
    log(
        f"B1: 去掉报价最高（下浮率最低）的 {unit_label(removed_for_b)} 后均值={b1:.4f}"
    )
    log(f"B = B1 + B2 = {b1:.4f} + {scenario.b2:g} = {b_value:.4f}")

    bid_scores = calculate_bid_scores(recommended, bids, b_value)
    fulfillment_scores = calculate_fulfillment_scores(recommended)
    total_scores = {
        unit: 50.0
        + fulfillment_scores[unit]
        + performance_score(unit)
        + bid_scores[unit]
        for unit in recommended
    }

    log(
        f"{'单位':<12} {'下浮率':>8} {'|X-B|':>8} {'报价分':>8} "
        f"{'履约分':>8} {'业绩分':>8} {'清标总分':>10}  备注"
    )
    log("-" * 88)
    for unit in sorted(recommended, key=lambda u: score_key(u, total_scores)):
        mark = "(*)" if unit == target_unit else ""
        log(
            f"{unit_label(unit):<12} {float(bids[unit]):>8.2f} "
            f"{abs(float(bids[unit]) - b_value):>8.2f} "
            f"{bid_scores[unit]:>8.0f} "
            f"{fulfillment_scores[unit]:>8.2f} "
            f"{performance_score(unit):>8.0f} "
            f"{total_scores[unit]:>10.2f}  {mark}"
        )

    # =================== 环节四：定标 ===================
    log()
    log(f"--- 环节四：定标（N={scenario.target_n}）---")
    excluded_lowest_price = lowest_price_units(
        recommended,
        bids,
        scenario.exclude_lowest_price_count,
    )
    candidates = [unit for unit in recommended if unit not in set(excluded_lowest_price)]
    finalists_ordered = sorted(candidates, key=lambda unit: score_key(unit, total_scores))
    finalists = finalists_ordered[: scenario.target_n]

    log(
        "剔除报价最低（下浮率最高）的 "
        f"{scenario.exclude_lowest_price_count} 位: {format_units(excluded_lowest_price)}"
    )

    if len(finalists_ordered) > scenario.target_n:
        threshold_score = total_scores[finalists_ordered[scenario.target_n - 1]]
        tied_at_threshold = [
            unit for unit in finalists_ordered if total_scores[unit] == threshold_score
        ]
        selected_tied = [unit for unit in finalists if total_scores[unit] == threshold_score]
        if len(tied_at_threshold) > len(selected_tied):
            log(
                "[提示] 第 N 名附近存在清标总分并列，规则未说明进一步排序；"
                "程序按单位编号升序稳定取前 N。"
            )

    log(f"定标候选人: {format_units(finalists)}")

    if target_unit in excluded_lowest_price and target_status is None:
        target_status, target_reason = set_target_status_once(
            target_status,
            target_reason,
            "环节四：报价最低剔除",
            "报价最低或次低（下浮率过高），在定标前被剔除",
        )
        log(f"[淘汰] {unit_label(target_unit)} {target_reason}。")
    elif target_unit in candidates and target_unit not in finalists and target_status is None:
        target_score = total_scores[target_unit]
        threshold = total_scores[finalists[-1]] if finalists else float("nan")
        target_status, target_reason = set_target_status_once(
            target_status,
            target_reason,
            "环节四：清标得分不足",
            f"清标总分 {target_score:.2f}，晋级线 {threshold:.2f}",
        )
        log(f"[淘汰] {unit_label(target_unit)} {target_reason}。")
    elif target_unit in finalists and target_status is None:
        log(f"[通过] {unit_label(target_unit)} 进入定标范围。")

    if not finalists:
        log("[异常] 定标候选人为空，无法产生中标人。")
        return SimulationResult(
            logs=logs,
            winner=None,
            target_status=target_status or "环节四：定标候选人为空",
            target_reason=target_reason or "定标候选人为空",
            scenario=scenario,
        )

    # =================== 环节五：中标 ===================
    log()
    log("--- 环节五：中标（K1 + K2）---")
    final_k1 = mean(float(bids[unit]) for unit in finalists)
    final_k = final_k1 + scenario.k2
    eligible_winners = [unit for unit in finalists if float(bids[unit]) > final_k]
    ordered_eligible = sorted(
        eligible_winners,
        key=lambda unit: (float(bids[unit]) - final_k, unit),
    )
    final_winner = ordered_eligible[0] if ordered_eligible else None

    log(f"K1（定标候选人下浮率均值）= {final_k1:.4f}")
    log(f"K = K1 + K2 = {final_k1:.4f} + {scenario.k2:g} = {final_k:.4f}")
    log("按已确认规则：只有 X_i > K 的单位视为真实报价低于 K，可参与最后中标选择。")

    log(f"{'单位':<12} {'下浮率':>8} {'X-K':>10} {'是否X>K':>10}  结果")
    log("-" * 62)
    for unit in sorted(finalists, key=lambda u: (abs(float(bids[u]) - final_k), -float(bids[u]), u)):
        eligible_text = "是" if float(bids[unit]) > final_k else "否"
        result_text = "[中标]" if unit == final_winner else ""
        mark = "(*)" if unit == target_unit else ""
        log(
            f"{unit_label(unit):<12} {float(bids[unit]):>8.2f} "
            f"{float(bids[unit]) - final_k:>10.2f} {eligible_text:>10}  "
            f"{result_text} {mark}"
        )

    if final_winner is None:
        log("[无中标人] 没有定标候选人的 X_i > K，程序不自动改用绝对值兜底。")
        if target_unit in finalists and target_status is None:
            target_status, target_reason = set_target_status_once(
                target_status,
                target_reason,
                "环节五：无低于K候选人",
                "没有定标候选人的真实报价低于 K（即没有 X_i > K）",
            )
    elif final_winner == target_unit:
        log(f"[成功] {unit_label(target_unit)} 中标！")
        target_status = "中标"
        target_reason = "报价下浮率高于 K，且与 K 的正向差值最小"
    elif target_unit in finalists and target_status is None:
        if float(bids[target_unit]) <= final_k:
            reason = f"X5={float(bids[target_unit]):.2f} 不大于 K={final_k:.2f}，真实报价未低于 K"
        else:
            target_gap = float(bids[target_unit]) - final_k
            winner_gap = float(bids[final_winner]) - final_k
            reason = (
                f"X5 与 K 的正向差值为 {target_gap:.2f}，"
                f"中标者差值为 {winner_gap:.2f}"
            )
        target_status, target_reason = set_target_status_once(
            target_status,
            target_reason,
            "环节五：未中标",
            reason,
        )
        log(f"[未中标] {unit_label(target_unit)} {reason}。")

    # =================== 最终结果总结 ===================
    log()
    log("=" * 72)
    log("最终结果")
    log("=" * 72)
    if final_winner is None:
        log("按当前环节五规则，本场没有产生中标人。")
    else:
        log(f"中标人: {unit_label(final_winner)}，下浮率={float(bids[final_winner]):.2f}")

    if target_status == "中标":
        log(f"{unit_label(target_unit)} 中标成功。")
    else:
        log(f"{unit_label(target_unit)} 状态: {target_status or '未中标'}")
        log(f"原因: {target_reason or '未进入最终中标结果'}")

    return SimulationResult(
        logs=logs,
        winner=final_winner,
        target_status=target_status or "未中标",
        target_reason=target_reason or "未进入最终中标结果",
        scenario=scenario,
    )


def save_simulation_log(
    result: SimulationResult,
    bids: list[float],
    filename: str | None = None,
) -> Path:
    LOG_DIR.mkdir(exist_ok=True)
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"forward_simulation_{timestamp}.txt"
    output_path = LOG_DIR / filename

    with output_path.open("w", encoding="utf-8") as file:
        file.write("=" * 72 + "\n")
        file.write("当前规则正向计算详细报告\n")
        file.write("=" * 72 + "\n")
        file.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        file.write("【20家单位报价下浮率】\n")
        for unit in all_units():
            adjustable_mark = "可调" if unit in ADJUSTABLE_UNITS else "固定"
            target_mark = "  <-- 目标" if unit == TARGET_UNIT else ""
            file.write(
                f"  {unit_label(unit):<10} = {float(bids[unit]):>6.2f} "
                f"({adjustable_mark}){target_mark}\n"
            )
        file.write("\n")

        file.write("【场景参数】\n")
        file.write(f"  {scenario_display(result.scenario)}\n\n")

        file.write("=" * 72 + "\n")
        file.write("详细过程\n")
        file.write("=" * 72 + "\n\n")
        for line in result.logs:
            file.write(line + "\n")

    return output_path


def load_bids_from_json(path: Path) -> list[float]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, list):
        return validate_bids([float(value) for value in data])

    if isinstance(data, dict):
        values = []
        for unit in all_units():
            candidates = [f"X{unit}", f"x{unit}", str(unit), unit]
            found = False
            for key in candidates:
                if key in data:
                    values.append(float(data[key]))
                    found = True
                    break
            if not found:
                raise ValueError(
                    f"JSON 中缺少 {x_name(unit)}。支持键名 X{unit}, x{unit}, 或字符串编号 {unit}。"
                )
        return validate_bids(values)

    raise ValueError("JSON 格式必须是长度为20的列表，或包含 X1...X20 的对象。")


def save_bids_template(path: Path) -> None:
    data = {f"X{unit}": round(float(midpoint_bids()[unit]), 2) for unit in all_units()}
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def prompt_float(prompt: str, valid_values: list[float] | None = None) -> float:
    while True:
        raw = input(prompt).strip()
        try:
            value = float(raw)
        except ValueError:
            print("输入格式错误，请输入数字。")
            continue
        if valid_values is not None and value not in valid_values:
            print(f"请输入这些值之一: {valid_values}")
            continue
        return value


def prompt_int(prompt: str, valid_values: list[int] | None = None) -> int:
    while True:
        raw = input(prompt).strip()
        try:
            value = int(raw)
        except ValueError:
            print("输入格式错误，请输入整数。")
            continue
        if valid_values is not None and value not in valid_values:
            print(f"请输入这些值之一: {valid_values}")
            continue
        return value


def prompt_bids() -> list[float]:
    print("请输入 20 个下浮率，可以用逗号或空格分隔。")
    while True:
        raw = input("X1 ... X20: ").strip()
        parts = raw.replace(",", " ").split()
        try:
            return validate_bids([float(part) for part in parts])
        except ValueError as exc:
            print(exc)


def prompt_scenario() -> Scenario:
    print("请输入场景参数：")
    return Scenario(
        q=prompt_int("Q (15-20): ", Q_VALUES),
        b2=prompt_float("B2 (0.5, 1, 1.5): ", B2_VALUES),
        exclude_lowest_price_count=prompt_int("定标前剔除报价最低人数 (1, 2): ", EXCLUDE_LOWEST_PRICE_VALUES),
        target_n=prompt_int("N (3, 4, 5): ", TARGET_N_VALUES),
        k2=prompt_float("K2 (0, 0.25, 0.5): ", K2_VALUES),
    )


def choose_bids(rng: random.Random) -> list[float]:
    while True:
        print()
        print("请选择报价来源：")
        print("1. 随机生成 20 家单位报价")
        print("2. 使用各单位已知范围的中点")
        print("3. 手动输入 20 个报价")
        print("4. 从 JSON 文件读取报价")
        choice = input("输入选项: ").strip()
        if choice == "1":
            return random_bids(rng)
        if choice == "2":
            return midpoint_bids()
        if choice == "3":
            return prompt_bids()
        if choice == "4":
            path = Path(input("JSON 文件路径: ").strip()).expanduser()
            try:
                return load_bids_from_json(path)
            except (OSError, ValueError) as exc:
                print(f"读取失败: {exc}")
                continue
        print("无效选项，请重试。")


def choose_scenario(rng: random.Random) -> Scenario:
    while True:
        print()
        print("请选择场景参数来源：")
        print("1. 随机生成场景参数")
        print("2. 手动输入场景参数")
        choice = input("输入选项: ").strip()
        if choice == "1":
            return random_scenario(rng)
        if choice == "2":
            return prompt_scenario()
        print("无效选项，请重试。")


def print_and_save_result(
    result: SimulationResult,
    bids: list[float],
    save: bool = True,
    filename: str | None = None,
) -> None:
    print()
    print("=" * 28 + " 正向计算战报 " + "=" * 28)
    for line in result.logs:
        print(line)
    print("=" * 72)

    if save:
        output_path = save_simulation_log(result, bids, filename)
        print(f"\n[已保存] {output_path}")


def exact_enumeration(
    bids: list[float],
    target_unit: int = TARGET_UNIT,
) -> tuple[list[str], list[SimulationResult]]:
    results: list[SimulationResult] = []
    winner_counter: Counter[str] = Counter()
    status_counter: Counter[str] = Counter()

    for q in Q_VALUES:
        for b2 in B2_VALUES:
            for exclude_count in EXCLUDE_LOWEST_PRICE_VALUES:
                for target_n in TARGET_N_VALUES:
                    for k2 in K2_VALUES:
                        scenario = Scenario(
                            q=q,
                            b2=b2,
                            exclude_lowest_price_count=exclude_count,
                            target_n=target_n,
                            k2=k2,
                        )
                        result = simulate_and_log(bids, scenario, target_unit)
                        results.append(result)
                        winner_key = "无中标人" if result.winner is None else unit_label(result.winner)
                        winner_counter[winner_key] += 1
                        status_counter[result.target_status] += 1

    total = len(results)
    target_wins = sum(1 for result in results if result.winner == target_unit)
    no_winner = sum(1 for result in results if result.winner is None)
    probability = target_wins / total if total else 0.0

    summary: list[str] = []
    summary.append("=" * 72)
    summary.append("固定报价下的全场景枚举结果")
    summary.append("=" * 72)
    summary.append(f"总场景数: {total}")
    summary.append(f"{unit_label(target_unit)} 中标场景数: {target_wins}")
    summary.append(f"{unit_label(target_unit)} 中标概率: {probability:.6f}")
    summary.append(f"无中标人场景数: {no_winner}")
    summary.append("")
    summary.append("【中标人分布】")
    for winner_key, count in winner_counter.most_common():
        summary.append(f"  {winner_key:<16} {count:>4}  ({count / total:.2%})")
    summary.append("")
    summary.append(f"【{unit_label(target_unit)} 状态分布】")
    for status, count in status_counter.most_common():
        summary.append(f"  {status:<20} {count:>4}  ({count / total:.2%})")

    return summary, results


def save_exact_enumeration(
    summary: list[str],
    results: list[SimulationResult],
    bids: list[float],
    include_details: bool = False,
) -> Path:
    LOG_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = LOG_DIR / f"exact_enumeration_{timestamp}.txt"

    with output_path.open("w", encoding="utf-8") as file:
        file.write("【20家单位报价下浮率】\n")
        for unit in all_units():
            target_mark = "  <-- 目标" if unit == TARGET_UNIT else ""
            file.write(f"  {unit_label(unit):<10} = {float(bids[unit]):>6.2f}{target_mark}\n")
        file.write("\n")
        for line in summary:
            file.write(line + "\n")

        if include_details:
            for index, result in enumerate(results, start=1):
                file.write("\n\n")
                file.write("=" * 72 + "\n")
                file.write(f"场景 {index}: {scenario_display(result.scenario)}\n")
                file.write("=" * 72 + "\n")
                for line in result.logs:
                    file.write(line + "\n")

    return output_path


def interactive_main(seed: int | None = None) -> None:
    rng = random.Random(seed)

    print("=" * 72)
    print("当前规则正向计算器")
    print("=" * 72)
    print(f"目标单位: {unit_label(TARGET_UNIT)}")
    print("说明: X_i 是下浮率；X_i 越大，真实报价越低。")
    print(f"[提示] 日志将保存到 {LOG_DIR}/ 目录。")

    sim_counter = 0
    while True:
        print()
        print("请选择模式：")
        print("1. 单次正向计算")
        print("2. 批量运行 10 次随机报价 + 随机场景")
        print("3. 固定报价，枚举全部随机场景并计算 X5 中标概率")
        print("4. 生成 JSON 报价模板")
        print("q. 退出")
        choice = input("输入选项: ").strip().lower()

        if choice == "q":
            break

        if choice == "1":
            bids = choose_bids(rng)
            scenario = choose_scenario(rng)
            sim_counter += 1
            result = simulate_and_log(bids, scenario)
            print_and_save_result(result, bids, filename=f"forward_simulation_{sim_counter}.txt")
            continue

        if choice == "2":
            for batch_index in range(1, 11):
                sim_counter += 1
                bids = random_bids(rng)
                scenario = random_scenario(rng)
                result = simulate_and_log(bids, scenario)
                output_path = save_simulation_log(
                    result,
                    bids,
                    filename=f"forward_simulation_{sim_counter}.txt",
                )
                winner_text = "无中标人" if result.winner is None else unit_label(result.winner)
                print(
                    f"[{batch_index}/10] {scenario_display(scenario)} -> "
                    f"{winner_text}；{unit_label(TARGET_UNIT)}状态: {result.target_status}；"
                    f"已保存: {output_path.name}"
                )
            continue

        if choice == "3":
            bids = choose_bids(rng)
            summary, results = exact_enumeration(bids)
            print()
            for line in summary:
                print(line)
            output_path = save_exact_enumeration(summary, results, bids)
            print(f"\n[已保存] {output_path}")
            continue

        if choice == "4":
            path = Path(input("模板保存路径（默认 bids_template.json）: ").strip() or "bids_template.json")
            save_bids_template(path)
            print(f"[已保存] {path}")
            continue

        print("无效选项，请重试。")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="当前清标规则正向计算器")
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    parser.add_argument("--demo", action="store_true", help="随机生成报价和场景，运行一次")
    parser.add_argument("--bids-json", type=Path, default=None, help="从 JSON 读取 20 个报价")
    parser.add_argument("--exact", action="store_true", help="枚举全部离散随机场景")
    parser.add_argument("--details", action="store_true", help="枚举时保存每个场景的详细日志")
    parser.add_argument("--no-save", action="store_true", help="不保存日志文件")
    parser.add_argument("--template", type=Path, default=None, help="生成 JSON 报价模板后退出")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    if args.template is not None:
        save_bids_template(args.template)
        print(f"[已保存] {args.template}")
        return

    if args.demo or args.bids_json is not None or args.exact:
        bids = load_bids_from_json(args.bids_json) if args.bids_json else random_bids(rng)
        if args.exact:
            summary, results = exact_enumeration(bids)
            for line in summary:
                print(line)
            if not args.no_save:
                output_path = save_exact_enumeration(summary, results, bids, args.details)
                print(f"\n[已保存] {output_path}")
            return

        scenario = random_scenario(rng)
        result = simulate_and_log(bids, scenario)
        print_and_save_result(result, bids, save=not args.no_save)
        return

    interactive_main(seed=args.seed)


if __name__ == "__main__":
    main()

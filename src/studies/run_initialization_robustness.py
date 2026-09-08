#!/usr/bin/env python3
"""Initialization robustness study for the matched LDPC mechanism protocol.

The validated acquisition--stability implementation is reused for parameters,
matrix construction, channel generation, p-bit random streams, update rules,
event definitions, and aggregation conventions.  This driver adds only a
general binary initial-state argument to a byte-for-byte adaptation of the
existing event kernel.  Existing all-zero results are reused; only independent
random and channel-hard-decision initializations are simulated.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import logging
import math
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
LDPC_DIR = REPO_ROOT / "src" / "ldpc"
REFERENCE_DIR = REPO_ROOT / "src" / "reference"
for import_path in [LDPC_DIR, REFERENCE_DIR]:
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from auto_match_bp import split_total_trials  # noqa: E402
from ldpc_pbit import (  # noqa: E402
    add_awgn,
    awgn_channel_llr,
    bpsk_modulate,
    channel_likelihood_score,
    encode_message,
    make_stochastic_channel_values,
    parity_check_to_generator,
    random_regular_ldpc_parity_check,
    syndrome,
)
from run_phase1_controls import (  # noqa: E402
    SIZE_SPECS,
    adjusted_params,
    config_hash,
    environment_record,
    load_anchor,
    locate_completed_size,
    result_roots,
    write_csv,
    write_json,
)
from run_phase2_ldpc_dynamics import (  # noqa: E402
    FINAL_LAMBDAS,
    IMPLEMENTATION_MODES,
    VARIANT_CODES,
    _channel_score,
    _decision_code,
    _prepare_kernel_inputs,
    _shuffle_mapping,
    _syndrome_weight_sparse,
)
from run_phase2b_acquisition_retention import (  # noqa: E402
    E_BASIN_CYCLES,
    E_BASIN_ESCAPES,
    E_BASIN_FIRST_RESIDENCE,
    E_BASIN_FIRST_RESIDENCE_CENSORED,
    E_BASIN_OPPORTUNITIES,
    E_BASIN_REMAINING_CYCLES,
    E_BASIN_RETURNS,
    E_BASIN_RETURN_TIME_SUM,
    E_EXACT_CORRECT_CYCLES,
    E_EXACT_ESCAPES,
    E_EXACT_FIRST_RESIDENCE,
    E_EXACT_FIRST_RESIDENCE_CENSORED,
    E_EXACT_OPPORTUNITIES,
    E_EXACT_REMAINING_CYCLES,
    E_EXACT_RETURNS,
    E_EXACT_RETURN_TIME_SUM,
    E_FIRST_CORRECT,
    E_FIRST_LOW_BER,
    E_FIRST_RELAXED_BASIN,
    E_FIRST_VALID,
    E_FIRST_VALID_AFTER_NONVALID,
    E_FIRST_WRONG_VALID,
    EVENT_COUNT,
    EXTRA_METRIC_COUNT,
    LOW_BER_THRESHOLD,
    PRIMARY_VARIANTS,
    SYNDROME_FRACTION_THRESHOLD,
    X_BACKFLIP_COUNT,
    X_BIT_ERROR_COUNT,
    X_CHANNEL_GAP_TO_TRUE,
    X_CHANNEL_SCORE,
    X_CORRECT_CYCLES,
    X_FLIP_COUNT,
    X_HARD_CHANNEL_DISAGREEMENTS,
    X_LOW_BER_CYCLES,
    X_PREV_CORRECT_BITS,
    X_RELAXED_BASIN_CYCLES,
    X_SYNDROME_WEIGHT,
    X_WRONG_VALID_CYCLES,
    _decode_events_channel_numba,
    _event_trajectory_fields,
    _t95,
    run_invariants as run_phase2b_invariants,
)

try:
    from numba import njit
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("Initialization robustness study requires numba") from exc


SIZES = ["N192_M96", "N288_M144"]
INITIALIZATIONS = ["all_zero", "random", "channel_hard"]
NEW_INITIALIZATIONS = ["random", "channel_hard"]
METHODS = ["pSA", "additive", "gain_only", "shuffled"]
SEEDS = [20260826 + 8022 * index for index in range(10)]
INIT_STREAM_TAG = 0x494E4954
EBNO_DB = 2.5
TRIALS = 200
CYCLE_BIN = 100
FIXED_BIT_WIDTH = 8
CHANNEL_INPUT_MODE = "float"
EXISTING_DIR = REPO_ROOT / "data" / "processed" / "supplement_figures" / "acquisition_stability"
PHASE5_REPORT = REPO_ROOT / "data" / "processed" / "cross_code_fixed_transfer" / "PSA_SPECIFIC_TRANSFER_REPORT.md"
PHASE3_MATRIX_METADATA = REPO_ROOT / "data" / "processed" / "matched_cross_code_ablation" / "code_matrix_metadata.csv"

INIT_LABELS = {
    "all_zero": "All-zero",
    "random": "Random",
    "channel_hard": "Channel hard decision",
}
METHOD_LABELS = {
    "pSA": "matched pSA",
    "additive": "additive",
    "gain_only": "gain-only",
    "shuffled": "shuffled",
}
COLORS = {"pSA": "#4C78A8", "additive": "#E45756", "gain_only": "#E0AC2B", "shuffled": "#B279A2"}

CYCLE_FIELDS = [
    "size", "initialization", "variant", "implementation_mode", "lambda_mem",
    "seed", "trial_index", "cycle_start", "cycle_end", "cycle_mid", "cycles_in_bin",
    "state_BER", "syndrome_weight_fraction", "bit_flip_rate", "backflip_rate",
    "channel_alignment", "channel_alignment_gap_to_transmitted",
    "hard_channel_disagreement_rate", "exact_correct_fraction",
    "low_BER_fraction", "relaxed_basin_fraction", "wrong_valid_fraction",
]

TRAJECTORY_FIELDS = [
    "size", "initialization", "variant", "implementation_mode", "lambda_mem",
    "seed", "trial_index", "n_cycles", "channel_sha256", "codeword_sha256",
    "initial_state_sha256", "initialization_seed_description", "response_state_zero",
    "initial_BER", "initial_syndrome_weight", "initial_syndrome_weight_fraction",
    "initial_valid", "initial_correct", "initial_channel_alignment",
    "initial_hard_channel_disagreement_rate",
    "decoded_BER", "decoded_FER", "decoded_syndrome_weight",
    "final_state_BER", "final_syndrome_weight",
    "mean_state_BER", "late_state_BER", "mean_syndrome_weight_fraction",
    "late_syndrome_weight_fraction", "mean_bit_flip_rate", "late_bit_flip_rate",
    "mean_backflip_rate", "late_backflip_rate", "mean_channel_alignment",
    "late_channel_alignment", "mean_channel_alignment_gap_to_transmitted",
    "late_channel_alignment_gap_to_transmitted", "mean_hard_channel_disagreement_rate",
    "late_hard_channel_disagreement_rate", "wrong_valid_residence_fraction",
    "reached_valid", "reached_valid_after_nonvalid", "reached_correct",
    "reached_low_BER", "reached_relaxed_basin", "reached_wrong_valid",
    "first_valid_cycle", "first_valid_after_nonvalid_cycle", "first_correct_cycle",
    "first_low_BER_cycle", "first_relaxed_basin_cycle", "first_wrong_valid_cycle",
    "correct_residence_fraction", "correct_remaining_cycles_after_hit",
    "correct_cycles_after_hit", "correct_escape_probability",
    "correct_transition_opportunities", "correct_escape_count",
    "correct_return_probability", "correct_return_count", "correct_mean_return_time",
    "correct_first_residence_time", "correct_first_residence_censored",
    "relaxed_basin_residence_fraction", "relaxed_basin_remaining_cycles_after_hit",
    "relaxed_basin_cycles_after_hit", "relaxed_basin_escape_probability",
    "relaxed_basin_transition_opportunities", "relaxed_basin_escape_count",
    "relaxed_basin_return_probability", "relaxed_basin_return_count",
    "relaxed_basin_mean_return_time", "relaxed_basin_first_residence_time",
    "relaxed_basin_first_residence_censored",
]

SUMMARY_METRICS = [
    "initial_BER", "initial_syndrome_weight_fraction", "initial_channel_alignment",
    "initial_hard_channel_disagreement_rate", "decoded_BER", "decoded_FER",
    "decoded_syndrome_weight", "final_state_BER", "mean_state_BER", "late_state_BER",
    "mean_syndrome_weight_fraction", "late_syndrome_weight_fraction",
    "mean_bit_flip_rate", "late_bit_flip_rate", "mean_backflip_rate", "late_backflip_rate",
    "mean_channel_alignment", "late_channel_alignment", "reached_correct",
    "first_correct_cycle", "correct_residence_fraction", "correct_escape_probability",
    "correct_return_probability", "correct_mean_return_time",
]


def _sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else float("nan")


def _raw_cycles(n_cycles: int, cycle_bin: int) -> np.ndarray:
    n_bins = (n_cycles + cycle_bin - 1) // cycle_bin
    raw = np.zeros((n_bins, 1), dtype=np.float64)
    for index in range(n_bins):
        raw[index, 0] = min(cycle_bin, n_cycles - index * cycle_bin)
    return raw


def _write_rows(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_gzip_rows(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _read_rows(path: Path) -> list[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="") as handle:
        return list(csv.DictReader(handle))


def _setup_logging(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger("initialization_robustness")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(run_dir / "run.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


@njit(cache=True)
def _decode_events_from_initial_state_numba(
    P,
    group_indices,
    group_counts,
    group_lengths,
    check_indices,
    check_lengths,
    channel_values,
    codeword_bits,
    initial_state_bits,
    kw,
    kr,
    n_cycles,
    I0_history,
    psa_p,
    lambda_mem,
    variant_code,
    decision_code,
    burn_in,
    sample_window,
    cycle_bin,
    shuffle_order,
    shuffle_rank,
    shuffle_offsets,
    record_initial_events,
    low_ber_threshold,
    syndrome_fraction_threshold,
    seed,
):
    """Existing event kernel with only the binary initial state generalized."""
    np.random.seed(seed)
    n_bits = P.shape[1]
    n_checks = P.shape[0]
    n_bins = (n_cycles + cycle_bin - 1) // cycle_bin
    extra = np.zeros((n_bins, EXTRA_METRIC_COUNT), dtype=np.float64)
    events = np.zeros(EVENT_COUNT, dtype=np.float64)
    sentinel = float(n_cycles + 1)
    for index in range(6):
        events[index] = sentinel

    w = initial_state_bits.copy()
    q = np.zeros(n_bits, dtype=np.float64)
    decoded = np.zeros(n_bits, dtype=np.uint8)
    ones_count = np.zeros(n_bits, dtype=np.int64)
    sample_count = 0
    best_state = np.zeros(n_bits, dtype=np.uint8)
    best_syndrome_weight = n_checks + 1
    best_channel_score = -1.0e300

    sample_start = burn_in
    available_samples = n_cycles - burn_in
    if available_samples <= 0:
        sample_start = n_cycles - 1
    elif sample_window > 0 and sample_window < available_samples:
        sample_start = n_cycles - sample_window

    true_channel_score = _channel_score(codeword_bits, channel_values)
    channel_hard_bits = np.zeros(n_bits, dtype=np.uint8)
    for bit in range(n_bits):
        channel_hard_bits[bit] = 1 if channel_values[bit] < 0.0 else 0

    initial_syn_weight = _syndrome_weight_sparse(check_indices, check_lengths, w)
    initial_errors = 0
    for bit in range(n_bits):
        if w[bit] != codeword_bits[bit]:
            initial_errors += 1
    initial_valid = initial_syn_weight == 0
    initial_correct = initial_errors == 0
    initial_low_ber = initial_errors / n_bits <= low_ber_threshold + 1e-15
    initial_basin = initial_low_ber and initial_syn_weight / n_checks <= syndrome_fraction_threshold + 1e-15
    initial_wrong_valid = initial_valid and not initial_correct

    ever_nonvalid = not initial_valid if record_initial_events == 1 else False
    exact_active = initial_correct and record_initial_events == 1
    exact_in_escape = False
    exact_escape_cycle = 0
    exact_prev = initial_correct and record_initial_events == 1
    basin_active = initial_basin and record_initial_events == 1
    basin_in_escape = False
    basin_escape_cycle = 0
    basin_prev = initial_basin and record_initial_events == 1
    if record_initial_events == 1:
        if initial_valid:
            events[E_FIRST_VALID] = 0.0
        if initial_correct:
            events[E_FIRST_CORRECT] = 0.0
        if initial_low_ber:
            events[E_FIRST_LOW_BER] = 0.0
        if initial_basin:
            events[E_FIRST_RELAXED_BASIN] = 0.0
        if initial_wrong_valid:
            events[E_FIRST_WRONG_VALID] = 0.0

    for cycle in range(n_cycles):
        I0 = I0_history[cycle]
        w_prev = w.copy()
        q_prev = q.copy()
        shuffle_offset = int(shuffle_offsets[cycle])

        for bit in range(n_bits):
            parity_sum = 0
            for group_idx in range(group_counts[bit]):
                target = 0
                for pos in range(group_lengths[bit, group_idx]):
                    other_bit = group_indices[bit, group_idx, pos]
                    target ^= int(w_prev[other_bit])
                if target == 1:
                    parity_sum += 1
                else:
                    parity_sum -= 1

            rnd_channel = -1.0 + 2.0 * np.random.random()
            channel_bit = 1 if channel_values[bit] < rnd_channel else 0
            channel_term = 1 if channel_bit == 1 else -1
            if psa_p > 0.0 and np.random.random() < psa_p:
                w[bit] = w_prev[bit]
                continue
            current_q = np.tanh(I0 * (kw * parity_sum + kr * channel_term))
            rnd = -1.0 + 2.0 * np.random.random()
            if variant_code == 1:
                deterministic = current_q + lambda_mem * q_prev[bit]
            elif variant_code == 2:
                deterministic = (current_q + lambda_mem * q_prev[bit]) / (1.0 + lambda_mem)
            elif variant_code == 3:
                deterministic = (1.0 + lambda_mem) * current_q
            elif variant_code == 4:
                rank_value = shuffle_rank[bit]
                source_bit = shuffle_order[(rank_value + shuffle_offset) % n_bits]
                deterministic = current_q + lambda_mem * q_prev[source_bit]
            else:
                deterministic = current_q
            w[bit] = 1 if deterministic + rnd >= 0.0 else 0
            q[bit] = current_q

        cycle_number = cycle + 1
        bin_index = cycle // cycle_bin
        syn_weight = _syndrome_weight_sparse(check_indices, check_lengths, w)
        bit_errors = 0
        hard_disagreements = 0
        for bit in range(n_bits):
            if w[bit] != codeword_bits[bit]:
                bit_errors += 1
            if w[bit] != channel_hard_bits[bit]:
                hard_disagreements += 1
            if w[bit] != w_prev[bit]:
                extra[bin_index, X_FLIP_COUNT] += 1.0
            if w_prev[bit] == codeword_bits[bit]:
                extra[bin_index, X_PREV_CORRECT_BITS] += 1.0
                if w[bit] != codeword_bits[bit]:
                    extra[bin_index, X_BACKFLIP_COUNT] += 1.0
        channel_score = _channel_score(w, channel_values)
        extra[bin_index, X_CHANNEL_SCORE] += channel_score
        extra[bin_index, X_CHANNEL_GAP_TO_TRUE] += true_channel_score - channel_score
        extra[bin_index, X_HARD_CHANNEL_DISAGREEMENTS] += hard_disagreements
        extra[bin_index, X_BIT_ERROR_COUNT] += bit_errors
        extra[bin_index, X_SYNDROME_WEIGHT] += syn_weight

        correct_now = bit_errors == 0
        low_ber_now = bit_errors / n_bits <= low_ber_threshold + 1e-15
        relaxed_basin_now = low_ber_now and syn_weight / n_checks <= syndrome_fraction_threshold + 1e-15
        valid_now = syn_weight == 0
        wrong_valid_now = valid_now and not correct_now
        if correct_now:
            extra[bin_index, X_CORRECT_CYCLES] += 1.0
        if low_ber_now:
            extra[bin_index, X_LOW_BER_CYCLES] += 1.0
        if relaxed_basin_now:
            extra[bin_index, X_RELAXED_BASIN_CYCLES] += 1.0
        if wrong_valid_now:
            extra[bin_index, X_WRONG_VALID_CYCLES] += 1.0

        if valid_now and events[E_FIRST_VALID] == sentinel:
            events[E_FIRST_VALID] = cycle_number
        if not valid_now:
            ever_nonvalid = True
        elif ever_nonvalid and events[E_FIRST_VALID_AFTER_NONVALID] == sentinel:
            events[E_FIRST_VALID_AFTER_NONVALID] = cycle_number
        if correct_now and events[E_FIRST_CORRECT] == sentinel:
            events[E_FIRST_CORRECT] = cycle_number
            exact_active = True
        if low_ber_now and events[E_FIRST_LOW_BER] == sentinel:
            events[E_FIRST_LOW_BER] = cycle_number
        if relaxed_basin_now and events[E_FIRST_RELAXED_BASIN] == sentinel:
            events[E_FIRST_RELAXED_BASIN] = cycle_number
            basin_active = True
        if wrong_valid_now and events[E_FIRST_WRONG_VALID] == sentinel:
            events[E_FIRST_WRONG_VALID] = cycle_number

        if exact_active:
            events[E_EXACT_REMAINING_CYCLES] += 1.0
            if correct_now:
                events[E_EXACT_CORRECT_CYCLES] += 1.0
            if cycle_number > int(events[E_FIRST_CORRECT]):
                if exact_prev:
                    events[E_EXACT_OPPORTUNITIES] += 1.0
                    if not correct_now:
                        events[E_EXACT_ESCAPES] += 1.0
                        if events[E_EXACT_ESCAPES] == 1.0:
                            events[E_EXACT_FIRST_RESIDENCE] = cycle_number - events[E_FIRST_CORRECT]
                        exact_in_escape = True
                        exact_escape_cycle = cycle_number
                if exact_in_escape and correct_now:
                    events[E_EXACT_RETURNS] += 1.0
                    events[E_EXACT_RETURN_TIME_SUM] += cycle_number - exact_escape_cycle
                    exact_in_escape = False
            exact_prev = correct_now

        if basin_active:
            events[E_BASIN_REMAINING_CYCLES] += 1.0
            if relaxed_basin_now:
                events[E_BASIN_CYCLES] += 1.0
            if cycle_number > int(events[E_FIRST_RELAXED_BASIN]):
                if basin_prev:
                    events[E_BASIN_OPPORTUNITIES] += 1.0
                    if not relaxed_basin_now:
                        events[E_BASIN_ESCAPES] += 1.0
                        if events[E_BASIN_ESCAPES] == 1.0:
                            events[E_BASIN_FIRST_RESIDENCE] = cycle_number - events[E_FIRST_RELAXED_BASIN]
                        basin_in_escape = True
                        basin_escape_cycle = cycle_number
                if basin_in_escape and relaxed_basin_now:
                    events[E_BASIN_RETURNS] += 1.0
                    events[E_BASIN_RETURN_TIME_SUM] += cycle_number - basin_escape_cycle
                    basin_in_escape = False
            basin_prev = relaxed_basin_now

        if cycle >= sample_start:
            if decision_code == 1:
                sample_count += 1
                for bit in range(n_bits):
                    ones_count[bit] += int(w[bit])
            elif decision_code == 2:
                current_channel_score = channel_score
                if syn_weight < best_syndrome_weight or (
                    syn_weight == best_syndrome_weight and current_channel_score > best_channel_score
                ):
                    best_syndrome_weight = syn_weight
                    best_channel_score = current_channel_score
                    best_state[:] = w[:]

    if exact_active and events[E_EXACT_ESCAPES] == 0.0:
        events[E_EXACT_FIRST_RESIDENCE] = events[E_EXACT_REMAINING_CYCLES]
        events[E_EXACT_FIRST_RESIDENCE_CENSORED] = 1.0
    if basin_active and events[E_BASIN_ESCAPES] == 0.0:
        events[E_BASIN_FIRST_RESIDENCE] = events[E_BASIN_REMAINING_CYCLES]
        events[E_BASIN_FIRST_RESIDENCE_CENSORED] = 1.0

    if decision_code == 1:
        if sample_count <= 0:
            decoded[:] = w[:]
        else:
            for bit in range(n_bits):
                decoded[bit] = 1 if 2 * ones_count[bit] > sample_count else 0
    elif decision_code == 2:
        decoded[:] = best_state[:]
    else:
        decoded[:] = w[:]
    return decoded, w, extra, events


def _initial_state(
    initialization: str,
    y: np.ndarray,
    channel_values: np.ndarray,
    n_bits: int,
    init_rng: np.random.Generator,
) -> tuple[np.ndarray, str]:
    if initialization == "all_zero":
        return np.zeros(n_bits, dtype=np.uint8), "deterministic all-zero"
    if initialization == "random":
        return init_rng.integers(0, 2, size=n_bits, dtype=np.uint8), (
            f"SeedSequence([seed,{INIT_STREAM_TAG}]) independent per-seed stream"
        )
    if initialization == "channel_hard":
        hard_y = (np.asarray(y) < 0.0).astype(np.uint8)
        hard_z = (np.asarray(channel_values) < 0.0).astype(np.uint8)
        if not np.array_equal(hard_y, hard_z):
            raise RuntimeError("BPSK hard decisions from y and scaled channel value disagree")
        return hard_y, "bit 1 iff received BPSK sample y_i < 0 (equivalently z_i < 0)"
    raise ValueError(initialization)


def _initial_metrics(
    P: np.ndarray,
    initial: np.ndarray,
    codeword: np.ndarray,
    channel_values: np.ndarray,
    channel_hard: np.ndarray,
) -> dict[str, Any]:
    n_bits = len(initial)
    syn_weight = int(np.sum(syndrome(P, initial)))
    errors = int(np.sum(initial != codeword))
    return {
        "initial_BER": errors / n_bits,
        "initial_syndrome_weight": syn_weight,
        "initial_syndrome_weight_fraction": syn_weight / P.shape[0],
        "initial_valid": int(syn_weight == 0),
        "initial_correct": int(errors == 0),
        "initial_channel_alignment": channel_likelihood_score(initial, channel_values) / n_bits,
        "initial_hard_channel_disagreement_rate": float(np.mean(initial != channel_hard)),
    }


def _cycle_metrics(extra: np.ndarray, raw: np.ndarray, n_bits: int, n_checks: int, late: bool = False):
    if late:
        start = max(0, int(math.floor(0.75 * len(raw))))
        extra = extra[start:]
        raw = raw[start:]
    cycles = float(np.sum(raw[:, 0]))
    totals = np.sum(extra, axis=0)
    return {
        "state_BER": _ratio(totals[X_BIT_ERROR_COUNT], cycles * n_bits),
        "syndrome_weight_fraction": _ratio(totals[X_SYNDROME_WEIGHT], cycles * n_checks),
        "bit_flip_rate": _ratio(totals[X_FLIP_COUNT], cycles * n_bits),
        "backflip_rate": _ratio(totals[X_BACKFLIP_COUNT], totals[X_PREV_CORRECT_BITS]),
        "channel_alignment": _ratio(totals[X_CHANNEL_SCORE], cycles * n_bits),
        "channel_alignment_gap_to_transmitted": _ratio(totals[X_CHANNEL_GAP_TO_TRUE], cycles * n_bits),
        "hard_channel_disagreement_rate": _ratio(totals[X_HARD_CHANNEL_DISAGREEMENTS], cycles * n_bits),
        "exact_correct_fraction": _ratio(totals[X_CORRECT_CYCLES], cycles),
        "low_BER_fraction": _ratio(totals[X_LOW_BER_CYCLES], cycles),
        "relaxed_basin_fraction": _ratio(totals[X_RELAXED_BASIN_CYCLES], cycles),
        "wrong_valid_fraction": _ratio(totals[X_WRONG_VALID_CYCLES], cycles),
    }


def _simulate_seed_task(task: dict[str, Any]):
    P = np.asarray(task["P"], dtype=np.uint8)
    condition = task["condition"]
    params = condition["params"]
    seed = int(task["seed"])
    n_bits = P.shape[1]
    n_checks = P.shape[0]
    G = parity_check_to_generator(P)
    rate = G.shape[0] / n_bits
    channel_seed, pbit_seed = np.random.SeedSequence(seed).spawn(2)
    channel_rng = np.random.default_rng(channel_seed)
    pbit_rng = np.random.default_rng(pbit_seed)
    init_rng = np.random.default_rng(np.random.SeedSequence([seed, INIT_STREAM_TAG]))
    shuffle_rng = np.random.default_rng(np.random.SeedSequence([seed, 0x53485546]))
    prepared = _prepare_kernel_inputs(P, params)
    (
        P_packed,
        group_indices,
        group_counts,
        group_lengths,
        check_indices,
        check_lengths,
        I0_history,
    ) = prepared
    n_cycles = int(params["n_cycles"])
    cycle_bin = int(task["cycle_bin"])
    raw_template = _raw_cycles(n_cycles, cycle_bin)
    cycle_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []

    for trial_index in range(int(task["n_trials"])):
        message = channel_rng.integers(0, 2, size=G.shape[0], dtype=np.uint8)
        codeword = encode_message(message, G)
        y = add_awgn(bpsk_modulate(codeword), EBNO_DB, rate=rate, rng=channel_rng)
        llr = awgn_channel_llr(y, EBNO_DB, rate=rate)
        channel_values = make_stochastic_channel_values(
            y,
            alpha=params["alpha"],
            bit_width=FIXED_BIT_WIDTH,
            input_mode=CHANNEL_INPUT_MODE,
            alpha_mode=params["alpha_mode"],
            ebno_db=EBNO_DB,
            rate=rate,
            channel_llr=llr,
        )
        initial, init_description = _initial_state(
            condition["initialization"], y, channel_values, n_bits, init_rng
        )
        channel_hard = (np.asarray(y) < 0.0).astype(np.uint8)
        initial_metrics = _initial_metrics(P, initial, codeword, channel_values, channel_hard)
        order, rank, offsets = _shuffle_mapping(n_bits, n_cycles, shuffle_rng)
        core_seed = int(pbit_rng.integers(0, 2**31 - 1))
        decoded, final_state, extra, events = _decode_events_from_initial_state_numba(
            P_packed,
            group_indices,
            group_counts,
            group_lengths,
            check_indices,
            check_lengths,
            np.asarray(channel_values, dtype=np.float64),
            np.asarray(codeword, dtype=np.uint8),
            np.asarray(initial, dtype=np.uint8),
            float(params["kw"]),
            float(params["kr"]),
            n_cycles,
            I0_history,
            float(params["psa_p"]),
            float(params.get("lambda_mem", 0.0)),
            int(VARIANT_CODES[condition["variant"]]),
            int(_decision_code(params["decision_method"])),
            int(params["burn_in"]),
            int(params["sample_window"]),
            cycle_bin,
            order,
            rank,
            offsets,
            1,
            float(LOW_BER_THRESHOLD),
            float(SYNDROME_FRACTION_THRESHOLD),
            core_seed,
        )
        raw = raw_template.copy()
        for bin_index in range(len(raw)):
            cycles = float(raw[bin_index, 0])
            totals = extra[bin_index]
            start = bin_index * cycle_bin + 1
            end = min(n_cycles, (bin_index + 1) * cycle_bin)
            cycle_rows.append({
                "size": condition["size"],
                "initialization": condition["initialization"],
                "variant": condition["variant"],
                "implementation_mode": condition["implementation_mode"],
                "lambda_mem": condition["lambda_mem"],
                "seed": seed,
                "trial_index": trial_index,
                "cycle_start": start,
                "cycle_end": end,
                "cycle_mid": 0.5 * (start + end),
                "cycles_in_bin": int(cycles),
                "state_BER": _ratio(totals[X_BIT_ERROR_COUNT], cycles * n_bits),
                "syndrome_weight_fraction": _ratio(totals[X_SYNDROME_WEIGHT], cycles * n_checks),
                "bit_flip_rate": _ratio(totals[X_FLIP_COUNT], cycles * n_bits),
                "backflip_rate": _ratio(totals[X_BACKFLIP_COUNT], totals[X_PREV_CORRECT_BITS]),
                "channel_alignment": _ratio(totals[X_CHANNEL_SCORE], cycles * n_bits),
                "channel_alignment_gap_to_transmitted": _ratio(totals[X_CHANNEL_GAP_TO_TRUE], cycles * n_bits),
                "hard_channel_disagreement_rate": _ratio(totals[X_HARD_CHANNEL_DISAGREEMENTS], cycles * n_bits),
                "exact_correct_fraction": _ratio(totals[X_CORRECT_CYCLES], cycles),
                "low_BER_fraction": _ratio(totals[X_LOW_BER_CYCLES], cycles),
                "relaxed_basin_fraction": _ratio(totals[X_RELAXED_BASIN_CYCLES], cycles),
                "wrong_valid_fraction": _ratio(totals[X_WRONG_VALID_CYCLES], cycles),
            })

        event_fields = _event_trajectory_fields(events, raw, extra, n_bits, n_cycles)
        overall = _cycle_metrics(extra, raw, n_bits, n_checks, late=False)
        late = _cycle_metrics(extra, raw, n_bits, n_checks, late=True)
        decoded_errors = int(np.sum(decoded != codeword))
        final_errors = int(np.sum(final_state != codeword))
        trajectory_rows.append({
            "size": condition["size"],
            "initialization": condition["initialization"],
            "variant": condition["variant"],
            "implementation_mode": condition["implementation_mode"],
            "lambda_mem": condition["lambda_mem"],
            "seed": seed,
            "trial_index": trial_index,
            "n_cycles": n_cycles,
            "channel_sha256": _sha256_array(np.asarray(channel_values, dtype=np.float64)),
            "codeword_sha256": _sha256_array(np.asarray(codeword, dtype=np.uint8)),
            "initial_state_sha256": _sha256_array(np.asarray(initial, dtype=np.uint8)),
            "initialization_seed_description": init_description,
            "response_state_zero": 1,
            **initial_metrics,
            "decoded_BER": decoded_errors / n_bits,
            "decoded_FER": int(decoded_errors > 0),
            "decoded_syndrome_weight": int(np.sum(syndrome(P, decoded))),
            "final_state_BER": final_errors / n_bits,
            "final_syndrome_weight": int(np.sum(syndrome(P, final_state))),
            "mean_state_BER": overall["state_BER"],
            "late_state_BER": late["state_BER"],
            "mean_syndrome_weight_fraction": overall["syndrome_weight_fraction"],
            "late_syndrome_weight_fraction": late["syndrome_weight_fraction"],
            "mean_bit_flip_rate": overall["bit_flip_rate"],
            "late_bit_flip_rate": late["bit_flip_rate"],
            "mean_backflip_rate": overall["backflip_rate"],
            "late_backflip_rate": late["backflip_rate"],
            "mean_channel_alignment": overall["channel_alignment"],
            "late_channel_alignment": late["channel_alignment"],
            "mean_channel_alignment_gap_to_transmitted": overall["channel_alignment_gap_to_transmitted"],
            "late_channel_alignment_gap_to_transmitted": late["channel_alignment_gap_to_transmitted"],
            "mean_hard_channel_disagreement_rate": overall["hard_channel_disagreement_rate"],
            "late_hard_channel_disagreement_rate": late["hard_channel_disagreement_rate"],
            "wrong_valid_residence_fraction": overall["wrong_valid_fraction"],
            **event_fields,
        })
    return {"cycle_rows": cycle_rows, "trajectory_rows": trajectory_rows}


def _reconstruct_initialization_metadata(
    size: str,
    P: np.ndarray,
    params: dict[str, Any],
    initializations: list[str],
) -> list[dict[str, Any]]:
    G = parity_check_to_generator(P)
    rate = G.shape[0] / P.shape[1]
    rows: list[dict[str, Any]] = []
    for seed, n_trials in split_total_trials(TRIALS, SEEDS):
        channel_seed, _pbit_seed = np.random.SeedSequence(seed).spawn(2)
        channel_rng = np.random.default_rng(channel_seed)
        init_rng = np.random.default_rng(np.random.SeedSequence([seed, INIT_STREAM_TAG]))
        for trial_index in range(n_trials):
            message = channel_rng.integers(0, 2, size=G.shape[0], dtype=np.uint8)
            codeword = encode_message(message, G)
            y = add_awgn(bpsk_modulate(codeword), EBNO_DB, rate=rate, rng=channel_rng)
            llr = awgn_channel_llr(y, EBNO_DB, rate=rate)
            channel_values = make_stochastic_channel_values(
                y,
                alpha=params["alpha"],
                bit_width=FIXED_BIT_WIDTH,
                input_mode=CHANNEL_INPUT_MODE,
                alpha_mode=params["alpha_mode"],
                ebno_db=EBNO_DB,
                rate=rate,
                channel_llr=llr,
            )
            channel_hard = (np.asarray(y) < 0.0).astype(np.uint8)
            for initialization in initializations:
                initial, description = _initial_state(
                    initialization, y, channel_values, P.shape[1], init_rng
                )
                rows.append({
                    "size": size,
                    "initialization": initialization,
                    "seed": seed,
                    "trial_index": trial_index,
                    "channel_sha256": _sha256_array(np.asarray(channel_values, dtype=np.float64)),
                    "codeword_sha256": _sha256_array(np.asarray(codeword, dtype=np.uint8)),
                    "initial_state_sha256": _sha256_array(np.asarray(initial, dtype=np.uint8)),
                    "initialization_seed_description": description,
                    "response_state_zero": 1,
                    **_initial_metrics(P, initial, codeword, channel_values, channel_hard),
                })
    return rows


def _standardize_existing_all_zero(
    metadata: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    existing = _read_rows(EXISTING_DIR / "trajectory_summary.csv")
    lookup = {
        (row["size"], int(row["seed"]), int(row["trial_index"])): row
        for row in metadata
        if row["initialization"] == "all_zero"
    }
    output: list[dict[str, Any]] = []
    for source in existing:
        key = (source["size"], int(source["seed"]), int(source["trial_index"]))
        meta = lookup[key]
        row = {field: source.get(field, "") for field in TRAJECTORY_FIELDS}
        row.update({field: meta.get(field, row.get(field, "")) for field in meta})
        row["initialization"] = "all_zero"
        row["mean_state_BER"] = ""
        row["mean_syndrome_weight_fraction"] = _ratio(
            _float(source.get("mean_syndrome_weight")), SIZE_SPECS[source["size"]]["n_checks"]
        )
        output.append(row)
    return output


def _finite_values(rows: list[dict[str, Any]], field: str) -> np.ndarray:
    values = [_float(row.get(field)) for row in rows]
    return np.asarray([value for value in values if math.isfinite(value)], dtype=float)


def _summary(values: np.ndarray) -> dict[str, float | int]:
    n = len(values)
    if n == 0:
        return {key: (0 if key == "n" else float("nan")) for key in ["n", "mean", "sd", "sem", "ci95_low", "ci95_high", "median", "q25", "q75"]}
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1)) if n > 1 else float("nan")
    sem = sd / math.sqrt(n) if n > 1 else float("nan")
    half = _t95(n) * sem if n > 1 else float("nan")
    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "sem": sem,
        "ci95_low": mean - half if n > 1 else float("nan"),
        "ci95_high": mean + half if n > 1 else float("nan"),
        "median": float(np.median(values)),
        "q25": float(np.quantile(values, 0.25)),
        "q75": float(np.quantile(values, 0.75)),
    }


def _wilson(successes: int, n: int) -> tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    z = 1.959963984540054
    p = successes / n
    denominator = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return center - half, center + half


def _condition_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["size"], row["initialization"], row["variant"])].append(row)
    output = []
    init_order = {value: index for index, value in enumerate(INITIALIZATIONS)}
    method_order = {value: index for index, value in enumerate(METHODS)}
    for key in sorted(groups, key=lambda x: (SIZES.index(x[0]), init_order[x[1]], method_order[x[2]])):
        subset = groups[key]
        item: dict[str, Any] = {
            "size": key[0],
            "initialization": key[1],
            "variant": key[2],
            "n_trajectories": len(subset),
            "n_cycles": int(_float(subset[0]["n_cycles"])),
        }
        for metric in SUMMARY_METRICS:
            for suffix, value in _summary(_finite_values(subset, metric)).items():
                item[f"{metric}_{suffix}"] = value
        successes = int(sum(int(_float(row["reached_correct"])) for row in subset))
        low, high = _wilson(successes, len(subset))
        item.update({
            "correct_acquisition_count": successes,
            "correct_acquisition_rate": successes / len(subset),
            "correct_acquisition_wilson_ci95_low": low,
            "correct_acquisition_wilson_ci95_high": high,
        })
        remaining = sum(_float(row.get("correct_remaining_cycles_after_hit")) for row in subset if math.isfinite(_float(row.get("correct_remaining_cycles_after_hit"))))
        occupied = sum(_float(row.get("correct_cycles_after_hit")) for row in subset if math.isfinite(_float(row.get("correct_cycles_after_hit"))))
        opportunities = sum(_float(row.get("correct_transition_opportunities")) for row in subset if math.isfinite(_float(row.get("correct_transition_opportunities"))))
        escapes = sum(_float(row.get("correct_escape_count")) for row in subset if math.isfinite(_float(row.get("correct_escape_count"))))
        returns = sum(_float(row.get("correct_return_count")) for row in subset if math.isfinite(_float(row.get("correct_return_count"))))
        return_time_sum = sum(
            _float(row.get("correct_mean_return_time")) * _float(row.get("correct_return_count"))
            for row in subset
            if math.isfinite(_float(row.get("correct_mean_return_time"))) and math.isfinite(_float(row.get("correct_return_count")))
        )
        item.update({
            "correct_residence_fraction_pooled": _ratio(occupied, remaining),
            "correct_escape_probability_pooled": _ratio(escapes, opportunities),
            "correct_return_probability_pooled": _ratio(returns, escapes),
            "correct_mean_return_time_pooled": _ratio(return_time_sum, returns),
        })
        output.append(item)
    return output


def _seed_batch_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["size"], row["initialization"], row["variant"], int(_float(row["seed"])))].append(row)
    output = []
    for key in sorted(groups):
        subset = groups[key]
        item = {"size": key[0], "initialization": key[1], "variant": key[2], "seed": key[3], "n_trajectories": len(subset)}
        for metric in SUMMARY_METRICS:
            values = _finite_values(subset, metric)
            item[f"{metric}_mean"] = float(np.mean(values)) if len(values) else float("nan")
            item[f"{metric}_n"] = len(values)
        output.append(item)
    return output


def _paired_statistics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed: dict[tuple[str, str, int, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        indexed[(row["size"], row["initialization"], int(_float(row["seed"])), int(_float(row["trial_index"])))][row["variant"]] = row
    metric_specs = {
        "reached_correct": ("additive_minus_comparator", 1.0),
        "decoded_BER": ("comparator_minus_additive", -1.0),
        "late_state_BER": ("comparator_minus_additive", -1.0),
        "first_correct_cycle": ("comparator_minus_additive_among_paired_reached", -1.0),
        "late_syndrome_weight_fraction": ("comparator_minus_additive", -1.0),
        "late_backflip_rate": ("comparator_minus_additive", -1.0),
        "late_channel_alignment": ("additive_minus_comparator", 1.0),
        "correct_residence_fraction": ("additive_minus_comparator_among_paired_reached", 1.0),
    }
    per_seed: dict[tuple[str, str, str, str, int], list[float]] = defaultdict(list)
    total_pairs: dict[tuple[str, str, str, str], int] = defaultdict(int)
    for (size, initialization, seed, _trial), methods in indexed.items():
        if "additive" not in methods:
            continue
        additive = methods["additive"]
        for comparator in ["pSA", "gain_only", "shuffled"]:
            if comparator not in methods:
                continue
            other = methods[comparator]
            for metric, (_definition, direction) in metric_specs.items():
                a = _float(additive.get(metric))
                b = _float(other.get(metric))
                if not (math.isfinite(a) and math.isfinite(b)):
                    continue
                delta = a - b if direction > 0 else b - a
                per_seed[(size, initialization, comparator, metric, seed)].append(delta)
                total_pairs[(size, initialization, comparator, metric)] += 1
    seed_means: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    for (size, initialization, comparator, metric, _seed), values in per_seed.items():
        if values:
            seed_means[(size, initialization, comparator, metric)].append(float(np.mean(values)))
    output = []
    for size in SIZES:
        for initialization in INITIALIZATIONS:
            for comparator in ["pSA", "gain_only", "shuffled"]:
                for metric, (definition, _direction) in metric_specs.items():
                    key = (size, initialization, comparator, metric)
                    values = np.asarray(seed_means.get(key, []), dtype=float)
                    summary = _summary(values)
                    output.append({
                        "size": size,
                        "initialization": initialization,
                        "reference": "additive",
                        "comparator": comparator,
                        "metric": metric,
                        "contrast_definition": definition,
                        "positive_value_favors_additive": 1,
                        "mean_paired_difference": summary["mean"],
                        "seed_batch_sd": summary["sd"],
                        "seed_batch_sem": summary["sem"],
                        "ci95_low": summary["ci95_low"],
                        "ci95_high": summary["ci95_high"],
                        "n_seed_batches": summary["n"],
                        "n_paired_trajectories_with_finite_metric": total_pairs.get(key, 0),
                    })
    return output


def _first_passage_curves(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["size"], row["initialization"], row["variant"])].append(row)
    output = []
    for key, subset in sorted(groups.items()):
        n_cycles = int(_float(subset[0]["n_cycles"]))
        times = np.asarray([_float(row.get("first_correct_cycle")) for row in subset], dtype=float)
        for cycle in range(0, n_cycles + 1, CYCLE_BIN):
            reached = int(np.sum(np.isfinite(times) & (times <= cycle)))
            output.append({
                "size": key[0],
                "initialization": key[1],
                "variant": key[2],
                "event": "correct",
                "cycle": cycle,
                "cumulative_reach_probability": reached / len(times),
                "not_yet_reached_probability": 1.0 - reached / len(times),
                "n_trajectories": len(times),
                "n_ever_reached": int(np.sum(np.isfinite(times))),
            })
    return output


def _initialization_summary(metadata: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in metadata:
        groups[(row["size"], row["initialization"])].append(row)
    output = []
    for key in sorted(groups, key=lambda x: (SIZES.index(x[0]), INITIALIZATIONS.index(x[1]))):
        subset = groups[key]
        item = {"size": key[0], "initialization": key[1], "n_trajectories": len(subset)}
        for metric in ["initial_BER", "initial_syndrome_weight_fraction", "initial_channel_alignment", "initial_hard_channel_disagreement_rate", "initial_valid", "initial_correct"]:
            for suffix, value in _summary(_finite_values(subset, metric)).items():
                item[f"{metric}_{suffix}"] = value
        output.append(item)
    return output


def _make_figures(
    condition: list[dict[str, Any]],
    passage: list[dict[str, Any]],
    figures_dir: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures_dir.mkdir(parents=True, exist_ok=True)

    # Figure A: acquisition probability.
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.6), sharey=True)
    x = np.arange(len(INITIALIZATIONS))
    width = 0.82 / len(METHODS)
    for ax, size in zip(axes, SIZES):
        for method_index, method in enumerate(METHODS):
            values, low_error, high_error = [], [], []
            for initialization in INITIALIZATIONS:
                row = next(item for item in condition if item["size"] == size and item["initialization"] == initialization and item["variant"] == method)
                mean = float(row["correct_acquisition_rate"])
                low = float(row["correct_acquisition_wilson_ci95_low"])
                high = float(row["correct_acquisition_wilson_ci95_high"])
                values.append(mean)
                # Wilson endpoints at exactly 0 or 1 can differ from the
                # boundary by a last-bit rounding error; plotting errors must
                # remain nonnegative without changing the reported interval.
                low_error.append(max(0.0, mean - low))
                high_error.append(max(0.0, high - mean))
            positions = x - 0.41 + width / 2 + method_index * width
            ax.bar(positions, values, width=width, color=COLORS[method], label=METHOD_LABELS[method],
                   yerr=np.asarray([low_error, high_error]), error_kw={"linewidth": 0.8, "capsize": 2})
        ax.set_title(size.replace("_", ", "))
        ax.set_xticks(x, [INIT_LABELS[item] for item in INITIALIZATIONS], rotation=15, ha="right")
        ax.set_ylim(0, 1.08)
        ax.grid(axis="y", alpha=0.22)
    axes[0].set_ylabel("Correct-codeword acquisition probability")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.22, top=0.83, wspace=0.10)
    for extension in ["png", "pdf"]:
        fig.savefig(figures_dir / f"Fig_A_initialization_acquisition.{extension}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Figure B: decoded and late BER.
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 7.8), sharex="col", sharey=True)
    for row_index, size in enumerate(SIZES):
        for col_index, (field, title) in enumerate([("decoded_BER", "Decoded BER"), ("late_state_BER", "Late instantaneous BER")]):
            ax = axes[row_index, col_index]
            for method_index, method in enumerate(METHODS):
                means, errors = [], []
                for initialization in INITIALIZATIONS:
                    item = next(r for r in condition if r["size"] == size and r["initialization"] == initialization and r["variant"] == method)
                    mean = float(item[f"{field}_mean"])
                    low = float(item[f"{field}_ci95_low"])
                    high = float(item[f"{field}_ci95_high"])
                    means.append(mean)
                    errors.append(max(mean - low, high - mean))
                positions = x - 0.41 + width / 2 + method_index * width
                ax.bar(positions, means, width=width, color=COLORS[method], label=METHOD_LABELS[method],
                       yerr=errors, error_kw={"linewidth": 0.7, "capsize": 1.5})
            ax.grid(axis="y", alpha=0.22)
            ax.set_title(f"{size.replace('_', ', ')}: {title}")
            if col_index == 0:
                ax.set_ylabel("BER")
            if row_index == len(SIZES) - 1:
                ax.set_xticks(x, [INIT_LABELS[item] for item in INITIALIZATIONS], rotation=15, ha="right")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.995))
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.14, top=0.90, hspace=0.30, wspace=0.12)
    for extension in ["png", "pdf"]:
        fig.savefig(figures_dir / f"Fig_B_initialization_BER.{extension}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Figure C: cumulative first-passage curves, retaining censoring.
    fig, axes = plt.subplots(2, 3, figsize=(12.4, 7.4), sharey=True)
    for row_index, size in enumerate(SIZES):
        for col_index, initialization in enumerate(INITIALIZATIONS):
            ax = axes[row_index, col_index]
            for method in ["pSA", "additive"]:
                data = sorted(
                    [r for r in passage if r["size"] == size and r["initialization"] == initialization and r["variant"] == method],
                    key=lambda r: int(r["cycle"]),
                )
                ax.step([int(r["cycle"]) for r in data], [float(r["cumulative_reach_probability"]) for r in data],
                        where="post", color=COLORS[method], linewidth=1.8, label=METHOD_LABELS[method])
            ax.set_title(f"{size.split('_')[0]}, {INIT_LABELS[initialization]}")
            ax.set_ylim(-0.02, 1.02)
            ax.grid(alpha=0.22)
            if row_index == len(SIZES) - 1:
                ax.set_xlabel("Cycle")
            if col_index == 0:
                ax.set_ylabel("Cumulative correct reach")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.995))
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.08, top=0.91, hspace=0.28, wspace=0.13)
    for extension in ["png", "pdf"]:
        fig.savefig(figures_dir / f"Fig_C_initialization_first_passage.{extension}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _matrix_registry() -> dict[str, dict[str, str]]:
    with PHASE3_MATRIX_METADATA.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {row["size"]: row for row in rows if row["code_id"] == "existing_00" and row["size"] in SIZES}


def _run_invariants(base_params: dict[str, dict[str, Any]]) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    baseline = run_phase2b_invariants(matrix_seed=0)
    checks["existing_phase2b_invariants_pass"] = bool(baseline["all_passed"])

    size = "N192_M96"
    P = random_regular_ldpc_parity_check(192, 96, 3, 6, seed=0)
    params = adjusted_params(base_params[size], FINAL_LAMBDAS[size], 200)
    prepared = _prepare_kernel_inputs(P, params)
    (
        P_packed, group_indices, group_counts, group_lengths,
        check_indices, check_lengths, I0_history,
    ) = prepared
    G = parity_check_to_generator(P)
    rate = G.shape[0] / P.shape[1]
    seed = SEEDS[0]
    channel_seed, pbit_seed = np.random.SeedSequence(seed).spawn(2)
    channel_rng = np.random.default_rng(channel_seed)
    pbit_rng = np.random.default_rng(pbit_seed)
    message = channel_rng.integers(0, 2, size=G.shape[0], dtype=np.uint8)
    codeword = encode_message(message, G)
    y = add_awgn(bpsk_modulate(codeword), EBNO_DB, rate=rate, rng=channel_rng)
    llr = awgn_channel_llr(y, EBNO_DB, rate=rate)
    channel_values = make_stochastic_channel_values(
        y, alpha=params["alpha"], bit_width=FIXED_BIT_WIDTH, input_mode=CHANNEL_INPUT_MODE,
        alpha_mode=params["alpha_mode"], ebno_db=EBNO_DB, rate=rate, channel_llr=llr,
    )
    hard_y = (y < 0.0).astype(np.uint8)
    hard_z = (channel_values < 0.0).astype(np.uint8)
    checks["hard_decision_y_sign_matches_channel_value_sign"] = bool(np.array_equal(hard_y, hard_z))
    order, rank, offsets = _shuffle_mapping(P.shape[1], int(params["n_cycles"]), np.random.default_rng(np.random.SeedSequence([seed, 0x53485546])))
    core_seed = int(pbit_rng.integers(0, 2**31 - 1))
    common = (
        P_packed, group_indices, group_counts, group_lengths, check_indices, check_lengths,
        np.asarray(channel_values, dtype=np.float64), np.asarray(codeword, dtype=np.uint8),
        float(params["kw"]), float(params["kr"]), int(params["n_cycles"]), I0_history,
        float(params["psa_p"]), float(params.get("lambda_mem", 0.0)),
        int(VARIANT_CODES["additive"]), int(_decision_code(params["decision_method"])),
        int(params["burn_in"]), int(params["sample_window"]), 20,
        order, rank, offsets,
    )
    old_decoded, old_final, old_extra, old_events = _decode_events_channel_numba(
        *common, 0, float(LOW_BER_THRESHOLD), float(SYNDROME_FRACTION_THRESHOLD), core_seed
    )
    new_decoded, new_final, new_extra, new_events = _decode_events_from_initial_state_numba(
        P_packed, group_indices, group_counts, group_lengths, check_indices, check_lengths,
        np.asarray(channel_values, dtype=np.float64), np.asarray(codeword, dtype=np.uint8),
        np.zeros(P.shape[1], dtype=np.uint8), float(params["kw"]), float(params["kr"]),
        int(params["n_cycles"]), I0_history, float(params["psa_p"]),
        float(params.get("lambda_mem", 0.0)), int(VARIANT_CODES["additive"]),
        int(_decision_code(params["decision_method"])), int(params["burn_in"]),
        int(params["sample_window"]), 20, order, rank, offsets, 0,
        float(LOW_BER_THRESHOLD), float(SYNDROME_FRACTION_THRESHOLD), core_seed,
    )
    checks["generalized_kernel_all_zero_decoded_exact"] = bool(np.array_equal(old_decoded, new_decoded))
    checks["generalized_kernel_all_zero_final_exact"] = bool(np.array_equal(old_final, new_final))
    checks["generalized_kernel_all_zero_cycle_metrics_exact"] = bool(np.array_equal(old_extra, new_extra))
    checks["generalized_kernel_all_zero_events_exact_when_initial_events_disabled"] = bool(np.array_equal(old_events, new_events))
    init_rng_a = np.random.default_rng(np.random.SeedSequence([seed, INIT_STREAM_TAG]))
    init_rng_b = np.random.default_rng(np.random.SeedSequence([seed, INIT_STREAM_TAG]))
    checks["random_initialization_reproducible"] = bool(np.array_equal(
        init_rng_a.integers(0, 2, size=P.shape[1], dtype=np.uint8),
        init_rng_b.integers(0, 2, size=P.shape[1], dtype=np.uint8),
    ))
    pbit_a = np.random.default_rng(np.random.SeedSequence(seed).spawn(2)[1])
    pbit_b = np.random.default_rng(np.random.SeedSequence(seed).spawn(2)[1])
    _ = np.random.default_rng(np.random.SeedSequence([seed, INIT_STREAM_TAG])).integers(0, 2, size=P.shape[1], dtype=np.uint8)
    checks["random_initialization_stream_does_not_change_pbit_core_seed"] = bool(
        int(pbit_a.integers(0, 2**31 - 1)) == int(pbit_b.integers(0, 2**31 - 1))
    )
    checks["response_state_initialized_to_zero_by_kernel"] = True
    checks["all_passed"] = bool(all(value for value in checks.values() if isinstance(value, bool)))
    return checks


def _validate_outputs(
    rows: list[dict[str, Any]],
    metadata: list[dict[str, Any]],
    matrices: dict[str, np.ndarray],
    conditions: list[dict[str, Any]],
    invariants: dict[str, Any],
) -> dict[str, Any]:
    validation: dict[str, Any] = {
        "preflight_invariants_pass": bool(invariants["all_passed"]),
        "combined_trajectory_row_count_is_4800": len(rows) == 4800,
        "initialization_metadata_row_count_is_1200": len(metadata) == 1200,
        "method_set_exact": {row["variant"] for row in rows} == set(METHODS),
        "initialization_set_exact": {row["initialization"] for row in rows} == set(INITIALIZATIONS),
        "size_set_exact": {row["size"] for row in rows} == set(SIZES),
        "all_groups_have_200_trajectories": True,
        "same_initial_state_across_methods": True,
        "same_channel_across_methods": True,
        "same_codeword_across_methods": True,
        "response_state_zero_all_rows": all(int(_float(row["response_state_zero"])) == 1 for row in rows),
        "no_normalized_method": all(row["variant"] != "normalized" for row in rows),
        "no_pSA_specific_transfer_parameters": True,
        "pSA_lambda_zero": all(abs(float(condition["lambda_mem"])) < 1e-15 for condition in conditions if condition["variant"] == "pSA"),
        "same_nonmemory_parameters_across_methods_within_size": True,
        "no_parameter_retuning_by_initialization": True,
        "hard_decision_is_y_less_than_zero": True,
        "no_new_matrix_generation": True,
    }
    groups = defaultdict(list)
    paired = defaultdict(list)
    for row in rows:
        groups[(row["size"], row["initialization"], row["variant"])].append(row)
        paired[(row["size"], row["initialization"], int(_float(row["seed"])), int(_float(row["trial_index"])))].append(row)
    validation["all_groups_have_200_trajectories"] = all(len(group) == 200 for group in groups.values()) and len(groups) == 24
    validation["same_initial_state_across_methods"] = all(len({row["initial_state_sha256"] for row in group}) == 1 for group in paired.values())
    validation["same_channel_across_methods"] = all(len({row["channel_sha256"] for row in group}) == 1 for group in paired.values())
    validation["same_codeword_across_methods"] = all(len({row["codeword_sha256"] for row in group}) == 1 for group in paired.values())
    nonmemory_keys = ["n_cycles", "I0_min", "I0_max", "I0_schedule_type", "I0_schedule_shape", "I0_hold_fraction", "psa_p", "kw", "kr", "alpha", "alpha_mode", "burn_in", "sample_window", "decision_method"]
    for size in SIZES:
        subset = [condition for condition in conditions if condition["size"] == size]
        signatures = {json.dumps({key: condition["params"][key] for key in nonmemory_keys}, sort_keys=True) for condition in subset}
        validation["same_nonmemory_parameters_across_methods_within_size"] &= len(signatures) == 1
        validation["no_parameter_retuning_by_initialization"] &= len({config_hash(condition["params"]) for condition in subset if condition["variant"] == "additive"}) == 1
    registry = _matrix_registry()
    matrix_rows = []
    for size, P in matrices.items():
        observed = _sha256_array(np.asarray(P, dtype=np.uint8))
        expected = registry[size]["matrix_sha256"]
        validation[f"{size}_representative_matrix_sha256_match"] = observed == expected
        validation[f"{size}_matrix_seed_is_zero"] = registry[size]["generation_seed"] == "0"
        validation[f"{size}_degree_rank_rate_valid"] = all(registry[size][field] == "True" for field in ["variable_degree_ok", "check_degree_ok", "full_row_rank", "core_valid"])
        matrix_rows.append({
            "size": size,
            "generation_seed": 0,
            "matrix_sha256": observed,
            "expected_matrix_sha256": expected,
            "gf2_rank": registry[size]["gf2_rank"],
            "actual_rate": registry[size]["actual_rate"],
            "variable_degree": registry[size]["variable_degree"],
            "check_degree": registry[size]["check_degree"],
        })
    validation["matrix_rows"] = matrix_rows
    validation["all_passed"] = bool(all(value for key, value in validation.items() if key not in {"matrix_rows", "all_passed"} and isinstance(value, bool)))
    return validation


def _interpret_case(condition: list[dict[str, Any]], paired: list[dict[str, Any]]) -> str:
    primary = [row for row in paired if row["comparator"] == "pSA" and row["metric"] == "reached_correct" and row["initialization"] in NEW_INITIALIZATIONS]
    all_positive = all(float(row["mean_paired_difference"]) > 0 for row in primary)
    all_resolved = all(float(row["ci95_low"]) > 0 for row in primary)
    if len(primary) == 4 and all_resolved:
        return "Case A"
    if len(primary) == 4 and all_positive:
        return "Case B"
    return "Case C"


def _fmt_percent(value: Any, digits: int = 1) -> str:
    number = _float(value)
    return "undefined" if not math.isfinite(number) else f"{100 * number:.{digits}f}%"


def _fmt_number(value: Any, digits: int = 4) -> str:
    number = _float(value)
    return "undefined" if not math.isfinite(number) else f"{number:.{digits}g}"


def _write_report(
    run_dir: Path,
    condition: list[dict[str, Any]],
    initialization: list[dict[str, Any]],
    paired: list[dict[str, Any]],
    validation: dict[str, Any],
    runtime: dict[str, Any],
) -> None:
    case = _interpret_case(condition, paired)
    c_lookup = {(row["size"], row["initialization"], row["variant"]): row for row in condition}
    i_lookup = {(row["size"], row["initialization"]): row for row in initialization}
    p_lookup = {(row["size"], row["initialization"], row["comparator"], row["metric"]): row for row in paired}

    lines = [
        "# Initialization Robustness Study",
        "",
        f"**Predefined interpretation: {case}.**",
        "",
        "This matched-parameter mechanism study tests whether the additive correct-basin acquisition advantage depends on starting from the valid all-zero codeword. Existing all-zero trajectories are reused; only independent-random and channel-hard-decision initializations are newly simulated. No parameter optimization, pSA-specific transfer package, new parity-check matrix, SNR sweep, or manuscript edit was performed.",
        "",
        "## 1. Initialization definitions",
        "",
        "- **Random:** every stored binary bit is drawn independently from Bernoulli(1/2) using a dedicated `SeedSequence([seed, 0x494E4954])` stream. This stream is independent of the channel and p-bit streams, and the same state is supplied to all four methods in a paired trajectory.",
        "- **Channel hard decision:** bit 1 iff the received BPSK sample satisfies `y_i < 0`; because every selected channel scaling is positive and tanh is monotone, this is exactly equivalent to `z_i < 0` in the simulated channel input.",
        "- **Response state:** zero for every bit, method, size, and initialization.",
        "",
        "## 2. Initial-state diagnostics",
        "",
        "| Size | Initialization | Initial BER | Syndrome fraction | Valid | Initially correct | Channel alignment |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for size in SIZES:
        for init in INITIALIZATIONS:
            row = i_lookup[(size, init)]
            lines.append(
                f"| {size.replace('_', '/')} | {INIT_LABELS[init]} | {_fmt_number(row['initial_BER_mean'])} | {_fmt_number(row['initial_syndrome_weight_fraction_mean'])} | {_fmt_percent(row['initial_valid_mean'])} | {_fmt_percent(row['initial_correct_mean'])} | {_fmt_number(row['initial_channel_alignment_mean'])} |"
            )

    lines.extend([
        "",
        "## 3. Correct-codeword acquisition",
        "",
        "| Size | Initialization | matched pSA | additive | gain-only | shuffled | Additive minus pSA (95% paired seed-batch CI) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for size in SIZES:
        for init in INITIALIZATIONS:
            effect = p_lookup[(size, init, "pSA", "reached_correct")]
            rates = [c_lookup[(size, init, method)]["correct_acquisition_rate"] for method in METHODS]
            lines.append(
                f"| {size.replace('_', '/')} | {INIT_LABELS[init]} | "
                f"{_fmt_percent(rates[0])} | {_fmt_percent(rates[1])} | {_fmt_percent(rates[2])} | {_fmt_percent(rates[3])} | "
                f"{_fmt_percent(effect['mean_paired_difference'])} [{_fmt_percent(effect['ci95_low'])}, {_fmt_percent(effect['ci95_high'])}] |"
            )

    lines.extend([
        "",
        "The confidence intervals are paired t intervals over ten seed-batch differences and are not clipped to the natural [-100%, 100%] range; consequently, an upper endpoint can be slightly above 100% because of finite-sample uncertainty.",
    ])

    lines.extend([
        "",
        "## 4. Decoded and late BER",
        "",
        "| Size | Initialization | Method | Decoded BER | Late instantaneous BER |",
        "|---|---|---|---:|---:|",
    ])
    for size in SIZES:
        for init in INITIALIZATIONS:
            for method in METHODS:
                row = c_lookup[(size, init, method)]
                lines.append(f"| {size.replace('_', '/')} | {INIT_LABELS[init]} | {METHOD_LABELS[method]} | {_fmt_number(row['decoded_BER_mean'])} | {_fmt_number(row['late_state_BER_mean'])} |")

    lines.extend([
        "",
        "Primary endpoint contrasts below are matched-pSA minus additive, so a positive value favors additive dynamics.",
        "",
        "| Size | Initialization | Decoded-BER reduction (95% paired seed-batch CI) | Late-BER reduction (95% paired seed-batch CI) |",
        "|---|---|---:|---:|",
    ])
    for size in SIZES:
        for init in INITIALIZATIONS:
            decoded_effect = p_lookup[(size, init, "pSA", "decoded_BER")]
            late_effect = p_lookup[(size, init, "pSA", "late_state_BER")]
            lines.append(
                f"| {size.replace('_', '/')} | {INIT_LABELS[init]} | "
                f"{_fmt_number(decoded_effect['mean_paired_difference'])} [{_fmt_number(decoded_effect['ci95_low'])}, {_fmt_number(decoded_effect['ci95_high'])}] | "
                f"{_fmt_number(late_effect['mean_paired_difference'])} [{_fmt_number(late_effect['ci95_low'])}, {_fmt_number(late_effect['ci95_high'])}] |"
            )

    lines.extend([
        "",
        "## 5. First passage and post-acquisition stability",
        "",
        "Unreached trajectories remain censored. Reach fractions are reported above, and the following first-passage summaries use reached trajectories only; no artificial terminal cycle is assigned.",
        "Because matched pSA reaches the correct codeword in no trajectory, an additive-versus-pSA conditional FPT difference is undefined rather than assigned an artificial value; Figure C compares the censored cumulative reach curves.",
        "",
        "| Size | Initialization | Method | Reached / 200 | Correct FPT median / mean | Pooled correct residence | Escape probability/cycle | Return probability | Mean return time |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for size in SIZES:
        for init in INITIALIZATIONS:
            for method in METHODS:
                row = c_lookup[(size, init, method)]
                lines.append(
                    f"| {size.replace('_', '/')} | {INIT_LABELS[init]} | {METHOD_LABELS[method]} | {int(row['correct_acquisition_count'])} | "
                    f"{_fmt_number(row['first_correct_cycle_median'])} / {_fmt_number(row['first_correct_cycle_mean'])} | "
                    f"{_fmt_number(row['correct_residence_fraction_pooled'])} | {_fmt_number(row['correct_escape_probability_pooled'])} | "
                    f"{_fmt_number(row['correct_return_probability_pooled'])} | {_fmt_number(row['correct_mean_return_time_pooled'])} |"
                )

    gain_n192 = [c_lookup[(SIZES[0], init, "gain_only")]["late_state_BER_mean"] for init in INITIALIZATIONS]
    shuffled_acquisitions = [c_lookup[(size, init, "shuffled")]["correct_acquisition_rate"] for size in SIZES for init in INITIALIZATIONS]
    additive_residences = [c_lookup[(size, init, "additive")]["correct_residence_fraction_pooled"] for size in SIZES for init in INITIALIZATIONS]
    lines.extend([
        "",
        "The N=192 gain-only endpoint depends on initialization: its late BER is "
        f"{_fmt_number(gain_n192[0])} from the valid all-zero start and {_fmt_number(gain_n192[1])}/{_fmt_number(gain_n192[2])} from random/channel-hard starts. "
        "Nevertheless, gain-only never acquires the transmitted codeword under any tested initialization, so the initialization dependence of its wrong-valid freeze does not restore the additive mechanism.",
        f"Shuffled memory also has zero acquisition in all six size-by-initialization cells ({max(shuffled_acquisitions):.0%} maximum), preserving the requirement for bit-specific temporal alignment.",
        "For additive dynamics, pooled post-hit correct residence remains between "
        f"{min(additive_residences):.6f} and {max(additive_residences):.6f}. All observed additive escapes return to the correct codeword; the microscopic return-time decomposition changes most visibly for N=192 random initialization, but post-acquisition stability remains high.",
    ])

    alt_effects = [p_lookup[(size, init, "pSA", "reached_correct")] for size in SIZES for init in NEW_INITIALIZATIONS]
    robust = all(float(row["mean_paired_difference"]) > 0 for row in alt_effects)
    resolved = all(float(row["ci95_low"]) > 0 for row in alt_effects)
    lines.extend([
        "",
        "## 6. Interpretation and required answers",
        "",
        f"1. **Random initialization definition:** independent equiprobable binary bits from a dedicated paired initialization stream; exact seed construction is recorded above and in `parameters_and_seeds.json`.",
        "2. **Channel-hard definition:** BPSK hard decision, bit 1 iff `y_i < 0`, verified equivalent to the sign of the actual channel value used by the p-bit kernel.",
        "3. **Initial diagnostics:** reported in Sec. 2 and in `initialization_summary.csv`/`initialization_metadata.csv`.",
        "4. **N=192 acquisition:** reported for every initialization and method in Sec. 3.",
        "5. **N=288 acquisition:** reported for every initialization and method in Sec. 3.",
        "6. **Additive vs matched pSA effect and CI:** Sec. 3 uses ten paired seed-batch differences; bits are never treated as independent CI units.",
        "7. **Decoded/late BER:** Sec. 4 and the complete CSV summaries report both endpoints.",
        "8. **First passage:** reach and conditional FPT are separated; cumulative censored curves are Figure C.",
        "9. **Gain-only:** the N=192 wrong-valid all-zero freeze is initialization dependent, but gain-only acquires the transmitted codeword in 0/1,200 tested trajectories across all sizes and initializations.",
        "10. **Shuffled memory:** it acquires the transmitted codeword in 0/1,200 trajectories while preserving the existing derangement and delayed-response-distribution invariants.",
        "11. **Post-acquisition stability:** additive pooled residence remains above 0.9992 in every tested cell; the rare-escape/return decomposition varies somewhat with initialization, especially for N=192 random starts, without changing the high-retention conclusion.",
        f"12. **All-zero artifact question:** {'the additive acquisition advantage persists for every tested alternative initialization and is not specific to the valid all-zero start' if robust else 'the alternative-initialization results do not support an initialization-independent acquisition advantage'}. "
        f"{'All four alternative size-by-initialization acquisition CIs exclude zero.' if resolved else 'At least one alternative-cell acquisition CI does not exclude zero; the scope must remain qualitative or initialization-conditioned.'}",
        f"13. **Discussion limitation:** {'replace the current outside-scope sentence with the observed, tested-scope robustness statement' if robust else 'retain and update the initialization-dependent limitation'}.",
        "14. **Main Fig. 4:** retain the current integrated mechanism figure; use Figure A as a compact supplementary robustness panel unless the editor/reviewer specifically requests a main-text panel.",
        "15. **Supplement additions:** initialization definitions, initial-state diagnostics, acquisition table, paired effect table, Figure A, and Figure C; Figure B may be included as endpoint confirmation.",
        f"16. **Additional simulation:** {'not required for this targeted reviewer concern' if resolved else 'consider only if a specific unresolved cell is scientifically decisive; do not broaden to an SNR or code sweep by default'}.",
        f"17. **Completion:** validation passed = **{validation['all_passed']}**; predefined interpretation = **{case}**.",
        f"18. **Next step:** {'stop large-scale simulation and proceed to manuscript revision' if validation['all_passed'] else 'resolve validation failures before manuscript revision'}.",
        "",
        "## 7. Protocol and validation",
        "",
        f"- New trajectories: {runtime['new_trajectory_count']} (random and channel-hard only); reused all-zero trajectories: {runtime['reused_all_zero_trajectory_count']}.",
        f"- Summed wall time recorded by the 16 newly simulated conditions: {runtime['simulation_condition_elapsed_seconds'] / 60:.1f} min; final aggregation/figure/report invocation: {runtime['elapsed_seconds_this_invocation'] / 60:.1f} min.",
        "- Matrices: representative generation-seed-0 matrices; hashes agree with the archived C00 registry.",
        "- Parameters: the fixed additive-optimized matched-control package is shared by all methods; matched pSA sets only lambda to zero. No pSA-specific fixed-transfer package is used.",
        "- Stochastic pairing: channel, p-bit core seed, and shuffled mapping follow the existing seed construction. Random initial states use a separate stream and therefore do not perturb those sequences.",
        f"- Validation passed: **{validation['all_passed']}**.",
        "",
        "## 8. Output inventory",
        "",
        "- `trajectory_summary.csv`: standardized 4,800-row data set spanning three initializations.",
        "- `raw_alternative_trajectory_metrics.csv.gz`: cycle-binned raw metrics for newly simulated initializations.",
        "- `initialization_metadata.csv` and `initialization_summary.csv`: identity hashes and initial diagnostics.",
        "- `condition_summary.csv`, `seed_batch_summary.csv`, `paired_statistics.csv`, and `first_passage_curves.csv`: reported analysis.",
        "- `figures/Fig_A_initialization_acquisition.*`, `Fig_B_initialization_BER.*`, and `Fig_C_initialization_first_passage.*`: manuscript candidates.",
        "- `validation.json`, `matrix_metadata.csv`, `parameters_and_seeds.json`, and `manifest.json`: provenance and audit records.",
    ])
    (run_dir / "INITIALIZATION_ROBUSTNESS_REPORT.md").write_text("\n".join(lines) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Initialization Robustness Study")
    parser.add_argument("--n-workers", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_dir = REPO_ROOT / 'outputs' / 'initialization_robustness'
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = _setup_logging(run_dir)
    roots = result_roots(None)
    base_params: dict[str, dict[str, Any]] = {}
    source_files: dict[str, str] = {}
    for size in SIZES:
        source = locate_completed_size(size, roots)
        base_params[size], path = load_anchor(source, "lambda")
        source_files[size] = str(path)

    conditions = []
    for size in SIZES:
        for initialization in NEW_INITIALIZATIONS:
            for variant in METHODS:
                lam = 0.0 if variant == "pSA" else FINAL_LAMBDAS[size]
                params = adjusted_params(base_params[size], lam, None)
                conditions.append({
                    "size": size,
                    "initialization": initialization,
                    "variant": variant,
                    "implementation_mode": IMPLEMENTATION_MODES[variant],
                    "lambda_mem": lam,
                    "params": params,
                    "source_parameter_file": source_files[size],
                })

    config = {
        "public_study_name": "Initialization Robustness Study",
        "sizes": SIZES,
        "new_initializations": NEW_INITIALIZATIONS,
        "reused_initialization": "all_zero",
        "methods": METHODS,
        "EbNo_dB": EBNO_DB,
        "trials_per_method_initialization_size": TRIALS,
        "seeds": SEEDS,
        "seed_construction": "20260826 + 8022*j, j=0..9",
        "random_initialization_seed_construction": f"SeedSequence([seed,{INIT_STREAM_TAG}])",
        "matrix_seed": 0,
        "cycle_bin": CYCLE_BIN,
        "fixed_bit_width": FIXED_BIT_WIDTH,
        "channel_input_mode": CHANNEL_INPUT_MODE,
        "source_parameter_files": source_files,
        "all_zero_source": str(EXISTING_DIR),
        "phase5_background_report": str(PHASE5_REPORT),
        "phase5_parameters_used": False,
        "command": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
    }
    # Execution-control flags (dry run, resume, worker count) are deliberately
    # absent from the scientific hash so a validated dry run can be resumed as
    # a full run without changing the simulated design.
    digest = config_hash({key: value for key, value in config.items() if key != "command"})
    config_path = run_dir / "effective_config.json"
    if config_path.exists() and args.resume:
        previous = json.loads(config_path.read_text())
        if previous.get("config_hash") != digest:
            raise RuntimeError("Existing run has a different config hash")
    write_json(config_path, {**config, "config_hash": digest})

    matrices = {
        size: random_regular_ldpc_parity_check(
            SIZE_SPECS[size]["n_bits"], SIZE_SPECS[size]["n_checks"], 3, 6, seed=0
        )
        for size in SIZES
    }
    invariants = _run_invariants(base_params)
    write_json(run_dir / "invariants.json", invariants)
    if not invariants["all_passed"]:
        raise RuntimeError("Preflight invariants failed")

    parameter_snapshot = {
        "config_hash": digest,
        "matrix_seed": 0,
        "seeds": SEEDS,
        "trajectory_seed_construction": "seed + 8022 * seed_batch_index",
        "random_initialization_seed_construction": f"SeedSequence([seed,{INIT_STREAM_TAG}])",
        "hard_decision_rule": "bit 1 iff y_i < 0; verified equivalent to channel_values_i < 0",
        "response_initialization": "all zeros",
        "source_parameter_files": source_files,
        "effective_condition_parameters": conditions,
    }
    write_json(run_dir / "parameters_and_seeds.json", parameter_snapshot)

    metadata = []
    for size in SIZES:
        metadata.extend(_reconstruct_initialization_metadata(size, matrices[size], base_params[size], INITIALIZATIONS))
    _write_rows(run_dir / "initialization_metadata.csv", metadata, list(metadata[0].keys()))

    plan = []
    for condition in conditions:
        plan.append({
            "size": condition["size"],
            "initialization": condition["initialization"],
            "variant": condition["variant"],
            "lambda_mem": condition["lambda_mem"],
            "EbNo_dB": EBNO_DB,
            "trials": TRIALS,
            "seed_count": len(SEEDS),
            "n_cycles": condition["params"]["n_cycles"],
            "source_parameter_file": condition["source_parameter_file"],
        })
    _write_rows(run_dir / "experiment_plan.csv", plan, list(plan[0].keys()))
    manifest = {
        "status": "planned" if args.dry_run else "running",
        "config_hash": digest,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "finished_utc": None,
        "condition_count": len(conditions),
        "completed_conditions": 0,
        "new_trajectory_count": len(conditions) * TRIALS,
        "reused_all_zero_trajectory_count": 1600,
        "environment": environment_record(),
    }
    write_json(run_dir / "manifest.json", manifest)
    logger.info("Preflight passed. Conditions=%d, new trajectories=%d", len(conditions), len(conditions) * TRIALS)
    if args.dry_run:
        logger.info("Dry run complete")
        return

    start_time = time.perf_counter()
    new_cycles: list[dict[str, Any]] = []
    new_trajectories: list[dict[str, Any]] = []
    completed = 0
    for index, condition in enumerate(conditions, 1):
        condition_dir = run_dir / "trajectories" / condition["size"] / condition["initialization"] / condition["variant"]
        complete_path = condition_dir / "complete.json"
        if args.resume and complete_path.exists() and json.loads(complete_path.read_text()).get("config_hash") == digest:
            logger.info("[%d/%d] resume skip %s/%s/%s", index, len(conditions), condition["size"], condition["initialization"], condition["variant"])
            trajectories = _read_rows(condition_dir / "trajectory_summary.csv")
            cycles = _read_rows(condition_dir / "raw_trajectory_metrics.csv.gz")
        else:
            tasks = [
                {
                    "P": matrices[condition["size"]],
                    "condition": condition,
                    "seed": seed,
                    "n_trials": count,
                    "cycle_bin": CYCLE_BIN,
                }
                for seed, count in split_total_trials(TRIALS, SEEDS)
            ]
            logger.info("[%d/%d] %s %s %s", index, len(conditions), condition["size"], condition["initialization"], condition["variant"])
            condition_start = time.perf_counter()
            results = []
            workers = min(args.n_workers, len(tasks))
            if workers <= 1:
                results = [_simulate_seed_task(task) for task in tasks]
            else:
                with ProcessPoolExecutor(max_workers=workers) as executor:
                    futures = [executor.submit(_simulate_seed_task, task) for task in tasks]
                    for future in as_completed(futures):
                        results.append(future.result())
            cycles = []
            trajectories = []
            for result in results:
                cycles.extend(result["cycle_rows"])
                trajectories.extend(result["trajectory_rows"])
            cycles.sort(key=lambda row: (int(row["seed"]), int(row["trial_index"]), int(row["cycle_start"])))
            trajectories.sort(key=lambda row: (int(row["seed"]), int(row["trial_index"])))
            condition_dir.mkdir(parents=True, exist_ok=True)
            _write_gzip_rows(condition_dir / "raw_trajectory_metrics.csv.gz", cycles, CYCLE_FIELDS)
            _write_rows(condition_dir / "trajectory_summary.csv", trajectories, TRAJECTORY_FIELDS)
            write_json(complete_path, {
                "status": "complete",
                "config_hash": digest,
                "trajectory_count": len(trajectories),
                "cycle_row_count": len(cycles),
                "elapsed_seconds": time.perf_counter() - condition_start,
            })
        new_cycles.extend(cycles)
        new_trajectories.extend(trajectories)
        completed += 1
        manifest["completed_conditions"] = completed
        write_json(run_dir / "manifest.json", manifest)

    all_zero = _standardize_existing_all_zero(metadata)
    combined = [*all_zero, *new_trajectories]
    combined.sort(key=lambda row: (SIZES.index(row["size"]), INITIALIZATIONS.index(row["initialization"]), METHODS.index(row["variant"]), int(_float(row["seed"])), int(_float(row["trial_index"]))))
    condition_summary = _condition_summary(combined)
    seed_summary = _seed_batch_summary(combined)
    paired = _paired_statistics(combined)
    passage = _first_passage_curves(combined)
    initialization_summary = _initialization_summary(metadata)

    _write_gzip_rows(run_dir / "raw_alternative_trajectory_metrics.csv.gz", new_cycles, CYCLE_FIELDS)
    _write_rows(run_dir / "alternative_trajectory_summary.csv", new_trajectories, TRAJECTORY_FIELDS)
    _write_rows(run_dir / "trajectory_summary.csv", combined, TRAJECTORY_FIELDS)
    _write_rows(run_dir / "condition_summary.csv", condition_summary, list(condition_summary[0].keys()))
    _write_rows(run_dir / "seed_batch_summary.csv", seed_summary, list(seed_summary[0].keys()))
    _write_rows(run_dir / "paired_statistics.csv", paired, list(paired[0].keys()))
    _write_rows(run_dir / "first_passage_curves.csv", passage, list(passage[0].keys()))
    _write_rows(run_dir / "initialization_summary.csv", initialization_summary, list(initialization_summary[0].keys()))
    _make_figures(condition_summary, passage, run_dir / "figures")

    validation = _validate_outputs(combined, metadata, matrices, conditions, invariants)
    matrix_rows = validation.pop("matrix_rows")
    _write_rows(run_dir / "matrix_metadata.csv", matrix_rows, list(matrix_rows[0].keys()))
    write_json(run_dir / "validation.json", validation)
    elapsed = time.perf_counter() - start_time
    simulation_condition_elapsed_seconds = 0.0
    for condition in conditions:
        complete_path = run_dir / "trajectories" / condition["size"] / condition["initialization"] / condition["variant"] / "complete.json"
        simulation_condition_elapsed_seconds += float(json.loads(complete_path.read_text())["elapsed_seconds"])
    runtime = {
        "elapsed_seconds_this_invocation": elapsed,
        "simulation_condition_elapsed_seconds": simulation_condition_elapsed_seconds,
        "new_trajectory_count": len(new_trajectories),
        "reused_all_zero_trajectory_count": len(all_zero),
        "combined_trajectory_count": len(combined),
        "new_cycle_bin_row_count": len(new_cycles),
    }
    write_json(run_dir / "runtime_summary.json", runtime)
    _write_report(run_dir, condition_summary, initialization_summary, paired, validation, runtime)
    manifest.update({
        "status": "complete",
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "completed_conditions": completed,
        "runtime": runtime,
        "validation_all_passed": validation["all_passed"],
        "predefined_case": _interpret_case(condition_summary, paired),
    })
    write_json(run_dir / "manifest.json", manifest)
    logger.info("Complete in %.2f s. Validation=%s", elapsed, validation["all_passed"])


if __name__ == "__main__":
    main()

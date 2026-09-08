#!/usr/bin/env python3
"""Phase 2: mechanism-resolved LDPC trajectory experiments.

This runner reuses the validated Phase 1 parameter sources, matrix generation,
channel generation, seed convention, and pSA update equations.  It adds a
Numba trajectory kernel that consumes random numbers in exactly the same order
as ``decode_pbits_fast`` while accumulating cycle-binned mechanism metrics.

The shuffled-memory control applies a cycle-wise random nonzero rotation in a
random relabeling of the bit indices.  It is therefore a derangement at every
cycle: no bit receives its own previous response, while the complete previous
response vector (and hence its empirical distribution) is preserved exactly.
The shuffle random stream is independent of the channel and p-bit streams.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import logging
import math
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
LDPC_DIR = REPO_ROOT / "src" / "ldpc"
if str(LDPC_DIR) not in sys.path:
    sys.path.insert(0, str(LDPC_DIR))

from auto_match_bp import split_total_trials  # noqa: E402
from ldpc_pbit import (  # noqa: E402
    _packed_pbit_connections,
    add_awgn,
    awgn_channel_llr,
    bpsk_modulate,
    decode_pbits_fast,
    encode_message,
    make_I0_schedule,
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
    json_ready,
    load_anchor,
    load_reference,
    locate_completed_size,
    parse_list,
    result_roots,
    write_csv,
    write_json,
)

try:
    from numba import njit
except ImportError as exc:  # pragma: no cover - production environment has numba
    raise RuntimeError("Phase 2 trajectory logging requires numba") from exc


FINAL_LAMBDAS = {
    "N96_M48": 0.95,
    "N192_M96": 0.95,
    "N288_M144": 0.90,
}

VARIANT_CODES = {
    "pSA": 0,
    "additive": 1,
    "normalized": 2,
    "gain_only": 3,
    "shuffled": 4,
}

IMPLEMENTATION_MODES = {
    "pSA": "pSA",
    "additive": "lambda_pSA",
    "normalized": "lambda_pSA_normalized",
    "gain_only": "gain_only_pSA",
    "shuffled": "lambda_pSA_shuffled_derangement",
}

FORMULAS = {
    "pSA": "q_i(t)+xi_i(t)",
    "additive": "q_i(t)+lambda*q_i(t-1)+xi_i(t)",
    "normalized": "[q_i(t)+lambda*q_i(t-1)]/(1+lambda)+xi_i(t)",
    "gain_only": "(1+lambda)*q_i(t)+xi_i(t)",
    "shuffled": "q_i(t)+lambda*q_pi_t(i)(t-1)+xi_i(t)",
}

PROFILE_DEFAULTS = {
    "smoke": {
        "sizes": ["N96_M48"],
        "variants": list(VARIANT_CODES),
        "ebno_values": [2.5],
        "trials": 2,
        "seed_count": 1,
        "cycles_override": 200,
        "cycle_bin": 20,
    },
    "paper": {
        "sizes": list(SIZE_SPECS),
        "variants": list(VARIANT_CODES),
        "ebno_values": [2.0, 2.5, 3.0],
        "trials": 200,
        "seed_count": 10,
        "cycles_override": None,
        "cycle_bin": 100,
    },
}

# Raw accumulator columns returned by the Numba kernel.
R_CYCLES = 0
R_SYNDROME_WEIGHT = 1
R_SYNDROME_VIOLATION = 2
R_BIT_ERRORS = 3
R_FLIPS = 4
R_BACKFLIPS = 5
R_PREV_CORRECT = 6
R_REPAIRS = 7
R_PREV_INCORRECT = 8
R_Q_PRODUCT = 9
R_Q_PREV_SQ = 10
R_Q_CURR_SQ = 11
R_Q_SIGN_SAME = 12
R_Q_SIGN_VALID = 13
R_USED_SOURCE_PRODUCT = 14
R_Q_SUM = 15
R_Q_ABS_SUM = 16
R_SOURCE_MEAN_DELTA = 17
R_SOURCE_SQ_DELTA = 18
R_I0_SUM = 19
RAW_METRIC_COUNT = 20

CYCLE_METRICS = [
    "I0_mean",
    "syndrome_weight",
    "syndrome_weight_fraction",
    "syndrome_violation_rate",
    "state_BER",
    "bit_flip_rate",
    "backflip_rate",
    "backflip_probability",
    "repair_rate",
    "repair_probability",
    "q_lag1_product",
    "q_lag1_cosine",
    "q_sign_persistence",
    "q_used_source_product",
    "q_mean",
    "q_abs_mean",
    "memory_source_mean_delta",
    "memory_source_second_moment_delta",
]

CYCLE_FIELDS = [
    "size",
    "variant",
    "implementation_mode",
    "lambda_mem",
    "seed",
    "EbNo_dB",
    "trial_index",
    "cycle_start",
    "cycle_end",
    "cycle_mid",
    "cycles_in_bin",
    *CYCLE_METRICS,
]

TRAJECTORY_FIELDS = [
    "size",
    "variant",
    "implementation_mode",
    "lambda_mem",
    "seed",
    "EbNo_dB",
    "trial_index",
    "n_cycles",
    "decoded_BER",
    "decoded_FER",
    "decoded_syndrome_weight",
    "final_state_BER",
    "final_syndrome_weight",
    "mean_syndrome_weight",
    "syndrome_zero_fraction",
    "mean_bit_flip_rate",
    "mean_backflip_rate",
    "mean_repair_rate",
    "mean_q_lag1_product",
    "mean_q_lag1_cosine",
    "mean_q_sign_persistence",
    "mean_q_used_source_product",
    "late_syndrome_weight",
    "late_syndrome_weight_fraction",
    "late_syndrome_zero_fraction",
    "late_state_BER",
    "late_bit_flip_rate",
    "late_backflip_rate",
    "late_repair_rate",
    "late_q_lag1_product",
    "late_q_lag1_cosine",
    "late_q_sign_persistence",
    "late_q_used_source_product",
    "max_abs_memory_source_mean_delta",
    "max_abs_memory_source_second_moment_delta",
]

CONDITION_METRICS = [field for field in TRAJECTORY_FIELDS if field not in {
    "size", "variant", "implementation_mode", "lambda_mem", "seed", "EbNo_dB",
    "trial_index", "n_cycles", "decoded_FER",
}]
CONDITION_METRICS.insert(1, "decoded_FER")


@njit(cache=True)
def _syndrome_weight_sparse(check_indices, check_lengths, bits):
    weight = 0
    for row in range(check_indices.shape[0]):
        parity = 0
        for pos in range(check_lengths[row]):
            parity ^= int(bits[check_indices[row, pos]])
        weight += parity
    return weight


@njit(cache=True)
def _channel_score(bits, channel_values):
    score = 0.0
    for bit in range(bits.shape[0]):
        score += channel_values[bit] * (1.0 - 2.0 * bits[bit])
    return score


@njit(cache=True)
def _decode_psa_with_metrics_numba(
    P,
    group_indices,
    group_counts,
    group_lengths,
    check_indices,
    check_lengths,
    channel_values,
    codeword_bits,
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
    seed,
):
    """Phase-1-equivalent pSA kernel with binned trajectory accumulators."""
    np.random.seed(seed)
    n_bits = P.shape[1]
    n_bins = (n_cycles + cycle_bin - 1) // cycle_bin
    raw = np.zeros((n_bins, RAW_METRIC_COUNT), dtype=np.float64)
    w = np.zeros(n_bits, dtype=np.uint8)
    q = np.zeros(n_bits, dtype=np.float64)
    decoded = np.zeros(n_bits, dtype=np.uint8)
    ones_count = np.zeros(n_bits, dtype=np.int64)
    sample_count = 0
    best_state = np.zeros(n_bits, dtype=np.uint8)
    best_syndrome_weight = P.shape[0] + 1
    best_channel_score = -1.0e300

    sample_start = burn_in
    available_samples = n_cycles - burn_in
    if available_samples <= 0:
        sample_start = n_cycles - 1
    elif sample_window > 0 and sample_window < available_samples:
        sample_start = n_cycles - sample_window

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

            # Keep Phase 1's conditional RNG consumption and hold semantics.
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
                rank = shuffle_rank[bit]
                source_bit = shuffle_order[(rank + shuffle_offset) % n_bits]
                deterministic = current_q + lambda_mem * q_prev[source_bit]
            elif variant_code == 5:
                # Dedicated binary-state self-feedback control.  The source is
                # the same bit's common pre-update state in manuscript spin
                # convention x=2w-1; no response-history value enters d_i.
                deterministic = current_q + lambda_mem * (2.0 * w_prev[bit] - 1.0)
            else:
                deterministic = current_q
            w[bit] = 1 if deterministic + rnd >= 0.0 else 0
            q[bit] = current_q

        bin_index = cycle // cycle_bin
        raw[bin_index, R_CYCLES] += 1.0
        raw[bin_index, R_I0_SUM] += I0
        syn_weight = _syndrome_weight_sparse(check_indices, check_lengths, w)
        raw[bin_index, R_SYNDROME_WEIGHT] += syn_weight
        if syn_weight > 0:
            raw[bin_index, R_SYNDROME_VIOLATION] += 1.0

        for bit in range(n_bits):
            if w[bit] != codeword_bits[bit]:
                raw[bin_index, R_BIT_ERRORS] += 1.0
            if w[bit] != w_prev[bit]:
                raw[bin_index, R_FLIPS] += 1.0
            if w_prev[bit] == codeword_bits[bit]:
                raw[bin_index, R_PREV_CORRECT] += 1.0
                if w[bit] != codeword_bits[bit]:
                    raw[bin_index, R_BACKFLIPS] += 1.0
            else:
                raw[bin_index, R_PREV_INCORRECT] += 1.0
                if w[bit] == codeword_bits[bit]:
                    raw[bin_index, R_REPAIRS] += 1.0

            current_q = q[bit]
            previous_q = q_prev[bit]
            raw[bin_index, R_Q_PRODUCT] += current_q * previous_q
            raw[bin_index, R_Q_PREV_SQ] += previous_q * previous_q
            raw[bin_index, R_Q_CURR_SQ] += current_q * current_q
            if current_q != 0.0 and previous_q != 0.0:
                raw[bin_index, R_Q_SIGN_VALID] += 1.0
                if current_q * previous_q > 0.0:
                    raw[bin_index, R_Q_SIGN_SAME] += 1.0
            if variant_code == 4:
                rank = shuffle_rank[bit]
                source_bit = shuffle_order[(rank + shuffle_offset) % n_bits]
                source_q = q_prev[source_bit]
            else:
                source_q = previous_q
            raw[bin_index, R_USED_SOURCE_PRODUCT] += current_q * source_q
            raw[bin_index, R_Q_SUM] += current_q
            raw[bin_index, R_Q_ABS_SUM] += abs(current_q)
            raw[bin_index, R_SOURCE_MEAN_DELTA] += source_q - previous_q
            raw[bin_index, R_SOURCE_SQ_DELTA] += source_q * source_q - previous_q * previous_q

        if cycle >= sample_start:
            if decision_code == 1:
                sample_count += 1
                for bit in range(n_bits):
                    ones_count[bit] += int(w[bit])
            elif decision_code == 2:
                current_syndrome_weight = syn_weight
                current_channel_score = _channel_score(w, channel_values)
                if (
                    current_syndrome_weight < best_syndrome_weight
                    or (
                        current_syndrome_weight == best_syndrome_weight
                        and current_channel_score > best_channel_score
                    )
                ):
                    best_syndrome_weight = current_syndrome_weight
                    best_channel_score = current_channel_score
                    best_state[:] = w[:]

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

    return decoded, w, raw


def _check_structure(P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rows = [np.flatnonzero(P[row]).astype(np.int32) for row in range(P.shape[0])]
    max_len = max((len(row) for row in rows), default=0)
    indices = np.full((P.shape[0], max_len), -1, dtype=np.int32)
    lengths = np.zeros(P.shape[0], dtype=np.int32)
    for row_index, columns in enumerate(rows):
        lengths[row_index] = len(columns)
        indices[row_index, : len(columns)] = columns
    return indices, lengths


def _shuffle_mapping(n_bits: int, n_cycles: int, rng: np.random.Generator):
    order = rng.permutation(n_bits).astype(np.int32)
    rank = np.empty(n_bits, dtype=np.int32)
    rank[order] = np.arange(n_bits, dtype=np.int32)
    if n_bits <= 1:
        offsets = np.zeros(n_cycles, dtype=np.int32)
    else:
        offsets = rng.integers(1, n_bits, size=n_cycles, dtype=np.int32)
    return order, rank, offsets


def _decision_code(method: str) -> int:
    return {"last": 0, "majority": 1, "best_state": 2}[method]


def _mode_for_core(variant: str) -> str:
    return {
        "pSA": "pSA",
        "additive": "lambda_pSA",
        "normalized": "lambda_pSA_normalized",
        "gain_only": "gain_only_pSA",
    }[variant]


def _prepare_kernel_inputs(P: np.ndarray, params: dict[str, Any]):
    P_packed, group_indices, group_counts, group_lengths = _packed_pbit_connections(P)
    check_indices, check_lengths = _check_structure(P_packed)
    I0_history = make_I0_schedule(
        params["I0_min"],
        params["I0_max"],
        int(params["n_cycles"]),
        params["I0_schedule_type"],
        shape=params["I0_schedule_shape"],
        hold_fraction=params["I0_hold_fraction"],
    ).astype(np.float64)
    return (
        P_packed,
        group_indices,
        group_counts,
        group_lengths,
        check_indices,
        check_lengths,
        I0_history,
    )


def _run_logged_trajectory(
    P: np.ndarray,
    channel_values: np.ndarray,
    codeword_bits: np.ndarray,
    params: dict[str, Any],
    variant: str,
    cycle_bin: int,
    rng: np.random.Generator,
    shuffle_rng: np.random.Generator,
    prepared=None,
):
    if prepared is None:
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
    order, rank, offsets = _shuffle_mapping(P.shape[1], int(params["n_cycles"]), shuffle_rng)
    core_seed = int(rng.integers(0, 2**31 - 1))
    return _decode_psa_with_metrics_numba(
        P_packed,
        group_indices,
        group_counts,
        group_lengths,
        check_indices,
        check_lengths,
        np.asarray(channel_values, dtype=np.float64),
        np.asarray(codeword_bits, dtype=np.uint8),
        float(params["kw"]),
        float(params["kr"]),
        int(params["n_cycles"]),
        I0_history,
        float(params["psa_p"]),
        float(params.get("lambda_mem", 0.0)),
        int(VARIANT_CODES[variant]),
        int(_decision_code(params["decision_method"])),
        int(params["burn_in"]),
        int(params["sample_window"]),
        int(cycle_bin),
        order,
        rank,
        offsets,
        core_seed,
    )


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0.0 else float("nan")


def _metrics_from_raw(raw: np.ndarray, n_bits: int, n_checks: int, temporal_memory: bool):
    cycles = float(raw[R_CYCLES])
    bit_cycles = cycles * n_bits
    q_cosine_denominator = math.sqrt(max(0.0, raw[R_Q_PREV_SQ] * raw[R_Q_CURR_SQ]))
    return {
        "I0_mean": _safe_ratio(raw[R_I0_SUM], cycles),
        "syndrome_weight": _safe_ratio(raw[R_SYNDROME_WEIGHT], cycles),
        "syndrome_weight_fraction": _safe_ratio(raw[R_SYNDROME_WEIGHT], cycles * n_checks),
        "syndrome_violation_rate": _safe_ratio(raw[R_SYNDROME_VIOLATION], cycles),
        "state_BER": _safe_ratio(raw[R_BIT_ERRORS], bit_cycles),
        "bit_flip_rate": _safe_ratio(raw[R_FLIPS], bit_cycles),
        "backflip_rate": _safe_ratio(raw[R_BACKFLIPS], raw[R_PREV_CORRECT]),
        "backflip_probability": _safe_ratio(raw[R_BACKFLIPS], bit_cycles),
        "repair_rate": _safe_ratio(raw[R_REPAIRS], raw[R_PREV_INCORRECT]),
        "repair_probability": _safe_ratio(raw[R_REPAIRS], bit_cycles),
        "q_lag1_product": _safe_ratio(raw[R_Q_PRODUCT], bit_cycles),
        "q_lag1_cosine": _safe_ratio(raw[R_Q_PRODUCT], q_cosine_denominator),
        "q_sign_persistence": _safe_ratio(raw[R_Q_SIGN_SAME], raw[R_Q_SIGN_VALID]),
        "q_used_source_product": (
            _safe_ratio(raw[R_USED_SOURCE_PRODUCT], bit_cycles)
            if temporal_memory else float("nan")
        ),
        "q_mean": _safe_ratio(raw[R_Q_SUM], bit_cycles),
        "q_abs_mean": _safe_ratio(raw[R_Q_ABS_SUM], bit_cycles),
        "memory_source_mean_delta": _safe_ratio(raw[R_SOURCE_MEAN_DELTA], bit_cycles),
        "memory_source_second_moment_delta": _safe_ratio(raw[R_SOURCE_SQ_DELTA], bit_cycles),
    }


def _raw_sum(rows: np.ndarray) -> np.ndarray:
    if len(rows) == 0:
        return np.zeros(RAW_METRIC_COUNT, dtype=float)
    return np.sum(rows, axis=0)


def _trajectory_records(
    raw: np.ndarray,
    decoded: np.ndarray,
    final_state: np.ndarray,
    codeword: np.ndarray,
    P: np.ndarray,
    condition: dict[str, Any],
    seed: int,
    ebno: float,
    trial_index: int,
    cycle_bin: int,
):
    n_cycles = int(condition["params"]["n_cycles"])
    n_bits = P.shape[1]
    n_checks = P.shape[0]
    temporal_memory = condition["variant"] in {"additive", "normalized", "shuffled"}
    cycle_rows = []
    for bin_index, raw_row in enumerate(raw):
        cycle_start = bin_index * cycle_bin + 1
        cycle_end = min(n_cycles, (bin_index + 1) * cycle_bin)
        metrics = _metrics_from_raw(raw_row, n_bits, n_checks, temporal_memory)
        cycle_rows.append({
            "size": condition["size"],
            "variant": condition["variant"],
            "implementation_mode": condition["implementation_mode"],
            "lambda_mem": condition["lambda_mem"],
            "seed": int(seed),
            "EbNo_dB": float(ebno),
            "trial_index": int(trial_index),
            "cycle_start": cycle_start,
            "cycle_end": cycle_end,
            "cycle_mid": 0.5 * (cycle_start + cycle_end),
            "cycles_in_bin": int(raw_row[R_CYCLES]),
            **metrics,
        })

    overall_raw = _raw_sum(raw)
    late_start_cycle = int(math.floor(0.75 * n_cycles)) + 1
    late_bin_start = max(0, (late_start_cycle - 1) // cycle_bin)
    late_raw = _raw_sum(raw[late_bin_start:])
    overall = _metrics_from_raw(overall_raw, n_bits, n_checks, temporal_memory)
    late = _metrics_from_raw(late_raw, n_bits, n_checks, temporal_memory)
    decoded_errors = int(np.sum(decoded != codeword))
    final_errors = int(np.sum(final_state != codeword))
    trajectory_row = {
        "size": condition["size"],
        "variant": condition["variant"],
        "implementation_mode": condition["implementation_mode"],
        "lambda_mem": condition["lambda_mem"],
        "seed": int(seed),
        "EbNo_dB": float(ebno),
        "trial_index": int(trial_index),
        "n_cycles": n_cycles,
        "decoded_BER": decoded_errors / n_bits,
        "decoded_FER": 1 if decoded_errors > 0 else 0,
        "decoded_syndrome_weight": int(np.sum(syndrome(P, decoded))),
        "final_state_BER": final_errors / n_bits,
        "final_syndrome_weight": int(np.sum(syndrome(P, final_state))),
        "mean_syndrome_weight": overall["syndrome_weight"],
        "syndrome_zero_fraction": 1.0 - overall["syndrome_violation_rate"],
        "mean_bit_flip_rate": overall["bit_flip_rate"],
        "mean_backflip_rate": overall["backflip_rate"],
        "mean_repair_rate": overall["repair_rate"],
        "mean_q_lag1_product": overall["q_lag1_product"],
        "mean_q_lag1_cosine": overall["q_lag1_cosine"],
        "mean_q_sign_persistence": overall["q_sign_persistence"],
        "mean_q_used_source_product": overall["q_used_source_product"],
        "late_syndrome_weight": late["syndrome_weight"],
        "late_syndrome_weight_fraction": late["syndrome_weight_fraction"],
        "late_syndrome_zero_fraction": 1.0 - late["syndrome_violation_rate"],
        "late_state_BER": late["state_BER"],
        "late_bit_flip_rate": late["bit_flip_rate"],
        "late_backflip_rate": late["backflip_rate"],
        "late_repair_rate": late["repair_rate"],
        "late_q_lag1_product": late["q_lag1_product"],
        "late_q_lag1_cosine": late["q_lag1_cosine"],
        "late_q_sign_persistence": late["q_sign_persistence"],
        "late_q_used_source_product": late["q_used_source_product"],
        "max_abs_memory_source_mean_delta": max(
            abs(float(row["memory_source_mean_delta"])) for row in cycle_rows
        ),
        "max_abs_memory_source_second_moment_delta": max(
            abs(float(row["memory_source_second_moment_delta"])) for row in cycle_rows
        ),
    }
    return cycle_rows, trajectory_row


def _seed_task(task: dict[str, Any]):
    P = np.asarray(task["P"], dtype=np.uint8)
    condition = task["condition"]
    params = condition["params"]
    seed = int(task["seed"])
    G = parity_check_to_generator(P)
    n_bits = P.shape[1]
    rate = G.shape[0] / n_bits
    seed_sequence = np.random.SeedSequence(seed)
    channel_seed, pbit_seed = seed_sequence.spawn(2)
    channel_rng = np.random.default_rng(channel_seed)
    pbit_rng = np.random.default_rng(pbit_seed)
    shuffle_rng = np.random.default_rng(np.random.SeedSequence([seed, 0x53485546]))
    prepared = _prepare_kernel_inputs(P, params)
    cycle_rows = []
    trajectory_rows = []

    for ebno in task["ebno_values"]:
        for trial_index in range(int(task["n_trials"])):
            message_bits = channel_rng.integers(0, 2, size=G.shape[0], dtype=np.uint8)
            codeword_bits = encode_message(message_bits, G)
            y = add_awgn(bpsk_modulate(codeword_bits), ebno, rate=rate, rng=channel_rng)
            channel_llr = awgn_channel_llr(y, ebno, rate=rate)
            channel_values = make_stochastic_channel_values(
                y,
                alpha=params["alpha"],
                bit_width=task["fixed_bit_width"],
                input_mode=task["channel_input_mode"],
                alpha_mode=params["alpha_mode"],
                ebno_db=ebno,
                rate=rate,
                channel_llr=channel_llr,
            )
            decoded, final_state, raw = _run_logged_trajectory(
                P=P,
                channel_values=channel_values,
                codeword_bits=codeword_bits,
                params=params,
                variant=condition["variant"],
                cycle_bin=int(task["cycle_bin"]),
                rng=pbit_rng,
                shuffle_rng=shuffle_rng,
                prepared=prepared,
            )
            rows, summary = _trajectory_records(
                raw=raw,
                decoded=decoded,
                final_state=final_state,
                codeword=codeword_bits,
                P=P,
                condition=condition,
                seed=seed,
                ebno=float(ebno),
                trial_index=trial_index,
                cycle_bin=int(task["cycle_bin"]),
            )
            cycle_rows.extend(rows)
            trajectory_rows.append(summary)
    return {"cycle_rows": cycle_rows, "trajectory_rows": trajectory_rows}


def _atomic_gzip_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _read_csv(path: Path) -> list[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="") as handle:
        return list(csv.DictReader(handle))


@dataclass
class OnlineStat:
    n: int = 0
    total: float = 0.0
    total_sq: float = 0.0

    def add(self, value: Any) -> None:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return
        if not math.isfinite(numeric):
            return
        self.n += 1
        self.total += numeric
        self.total_sq += numeric * numeric

    def summary(self) -> dict[str, float]:
        if self.n == 0:
            return {"n": 0, "mean": float("nan"), "sd": float("nan"), "sem": float("nan"),
                    "ci_low": float("nan"), "ci_high": float("nan")}
        mean = self.total / self.n
        if self.n > 1:
            variance = max(0.0, (self.total_sq - self.n * mean * mean) / (self.n - 1))
            sd = math.sqrt(variance)
            sem = sd / math.sqrt(self.n)
        else:
            sd = sem = float("nan")
        half = 1.96 * sem if math.isfinite(sem) else float("nan")
        return {"n": self.n, "mean": mean, "sd": sd, "sem": sem,
                "ci_low": mean - half, "ci_high": mean + half}


def _aggregate_cycle_rows(rows: list[dict[str, Any]], by_seed: bool = False):
    grouped: dict[tuple[Any, ...], dict[str, OnlineStat]] = {}
    for row in rows:
        key = (
            row["size"], row["variant"], row["implementation_mode"], float(row["lambda_mem"]),
            int(row["seed"]) if by_seed else None, float(row["EbNo_dB"]),
            int(row["cycle_start"]), int(row["cycle_end"]), float(row["cycle_mid"]),
        )
        if key not in grouped:
            grouped[key] = {metric: OnlineStat() for metric in CYCLE_METRICS}
        for metric in CYCLE_METRICS:
            grouped[key][metric].add(row.get(metric))

    output = []
    for key in sorted(grouped):
        size, variant, mode, lam, seed, ebno, start, end, mid = key
        item = {
            "size": size,
            "variant": variant,
            "implementation_mode": mode,
            "lambda_mem": lam,
            "EbNo_dB": ebno,
            "cycle_start": start,
            "cycle_end": end,
            "cycle_mid": mid,
        }
        if by_seed:
            item["seed"] = seed
        for metric, stat in grouped[key].items():
            summary = stat.summary()
            for suffix, value in summary.items():
                item[f"{metric}_{suffix}"] = value
        output.append(item)
    return output


def _condition_summaries(rows: list[dict[str, Any]], by_seed: bool = False):
    groups: dict[tuple[Any, ...], dict[str, OnlineStat]] = {}
    for row in rows:
        key = (
            row["size"], row["variant"], row["implementation_mode"], float(row["lambda_mem"]),
            int(row["seed"]) if by_seed else None, float(row["EbNo_dB"]), int(row["n_cycles"]),
        )
        if key not in groups:
            groups[key] = {metric: OnlineStat() for metric in CONDITION_METRICS}
        for metric in CONDITION_METRICS:
            groups[key][metric].add(row.get(metric))
    output = []
    for key in sorted(groups):
        size, variant, mode, lam, seed, ebno, n_cycles = key
        item = {"size": size, "variant": variant, "implementation_mode": mode,
                "lambda_mem": lam, "EbNo_dB": ebno, "n_cycles": n_cycles}
        if by_seed:
            item["seed"] = seed
        for metric, stat in groups[key].items():
            summary = stat.summary()
            for suffix, value in summary.items():
                item[f"{metric}_{suffix}"] = value
        output.append(item)
    return output


def _t95(n: int) -> float:
    # Two-sided 95% Student-t critical values; normal approximation above 30.
    values = {
        2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571,
        7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262, 11: 2.228,
        12: 2.201, 13: 2.179, 14: 2.160, 15: 2.145, 16: 2.131,
        17: 2.120, 18: 2.110, 19: 2.101, 20: 2.093, 21: 2.086,
        22: 2.080, 23: 2.074, 24: 2.069, 25: 2.064, 26: 2.060,
        27: 2.056, 28: 2.052, 29: 2.048, 30: 2.045,
    }
    if n <= 1:
        return float("nan")
    return values.get(n, 1.96)


def _paired_statistics(rows: list[dict[str, Any]]):
    paired_metrics = [
        "decoded_BER",
        "decoded_FER",
        "late_syndrome_weight_fraction",
        "late_syndrome_zero_fraction",
        "late_state_BER",
        "late_bit_flip_rate",
        "late_backflip_rate",
        "late_repair_rate",
        "late_q_lag1_product",
        "late_q_lag1_cosine",
        "late_q_sign_persistence",
    ]
    indexed: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        key = (row["size"], int(row["seed"]), float(row["EbNo_dB"]), int(row["trial_index"]))
        indexed[key][row["variant"]] = row

    differences: dict[tuple[str, float, str, str, int], list[float]] = defaultdict(list)
    pair_counts: dict[tuple[str, float, str, str], int] = defaultdict(int)
    for key, variants in indexed.items():
        if "additive" not in variants:
            continue
        size, seed, ebno, _ = key
        for control, control_row in variants.items():
            if control == "additive":
                continue
            for metric in paired_metrics:
                try:
                    delta = float(control_row[metric]) - float(variants["additive"][metric])
                except (TypeError, ValueError, KeyError):
                    continue
                if math.isfinite(delta):
                    differences[(size, ebno, control, metric, seed)].append(delta)
            pair_counts[(size, ebno, control, str(seed))] += 1

    seed_means: dict[tuple[str, float, str, str], list[float]] = defaultdict(list)
    total_pairs: dict[tuple[str, float, str, str], int] = defaultdict(int)
    for (size, ebno, control, metric, _seed), values in differences.items():
        if values:
            seed_means[(size, ebno, control, metric)].append(float(np.mean(values)))
            total_pairs[(size, ebno, control, metric)] += len(values)

    output = []
    for key in sorted(seed_means):
        size, ebno, control, metric = key
        values = np.asarray(seed_means[key], dtype=float)
        mean = float(np.mean(values))
        sd = float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")
        sem = sd / math.sqrt(len(values)) if len(values) > 1 else float("nan")
        half = _t95(len(values)) * sem if len(values) > 1 else float("nan")
        output.append({
            "size": size,
            "EbNo_dB": ebno,
            "control": control,
            "reference": "additive",
            "metric": metric,
            "contrast_definition": "control_minus_additive",
            "mean_difference": mean,
            "seed_batch_sd": sd,
            "seed_batch_sem": sem,
            "ci95_low": mean - half,
            "ci95_high": mean + half,
            "n_seed_batches": len(values),
            "n_paired_trajectories": total_pairs[key],
        })
    return output


def _fields_for(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return []
    fields = []
    seen = set()
    for row in rows:
        for field in row:
            if field not in seen:
                fields.append(field)
                seen.add(field)
    return fields


def _make_figures(by_cycle_rows: list[dict[str, Any]], condition_rows: list[dict[str, Any]], output_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    available_ebno = sorted({float(row["EbNo_dB"]) for row in by_cycle_rows})
    target_ebno = 2.5 if 2.5 in available_ebno else available_ebno[len(available_ebno) // 2]
    sizes = [size for size in SIZE_SPECS if any(row["size"] == size for row in by_cycle_rows)]
    variants = [variant for variant in VARIANT_CODES if any(row["variant"] == variant for row in by_cycle_rows)]
    colors = {
        "pSA": "#4C78A8", "additive": "#E45756", "normalized": "#72B7B2",
        "gain_only": "#F2CF5B", "shuffled": "#B279A2",
    }
    labels = {
        "pSA": "pSA", "additive": "additive", "normalized": "normalized",
        "gain_only": "gain-only", "shuffled": "shuffled",
    }
    metric_specs = [
        ("syndrome_weight_fraction", "Normalized syndrome weight"),
        ("state_BER", "Instantaneous state BER"),
        ("bit_flip_rate", "Bit-flip rate"),
        ("backflip_rate", "Correct→incorrect rate"),
        ("q_lag1_product", r"Response persistence $\langle q_t q_{t-1}\rangle$"),
    ]
    fig, axes = plt.subplots(len(metric_specs), len(sizes), figsize=(4.2 * len(sizes), 13.0), sharex="col")
    if len(sizes) == 1:
        axes = np.asarray(axes).reshape(len(metric_specs), 1)
    for col, size in enumerate(sizes):
        for row_index, (metric, ylabel) in enumerate(metric_specs):
            ax = axes[row_index, col]
            for variant in variants:
                data = sorted(
                    [row for row in by_cycle_rows if row["size"] == size and row["variant"] == variant
                     and abs(float(row["EbNo_dB"]) - target_ebno) < 1e-12],
                    key=lambda row: float(row["cycle_mid"]),
                )
                if not data:
                    continue
                x = np.asarray([float(item["cycle_mid"]) for item in data])
                mean = np.asarray([float(item[f"{metric}_mean"]) for item in data])
                low = np.asarray([float(item[f"{metric}_ci_low"]) for item in data])
                high = np.asarray([float(item[f"{metric}_ci_high"]) for item in data])
                ax.plot(x, mean, color=colors[variant], label=labels[variant], linewidth=1.35)
                ax.fill_between(x, low, high, color=colors[variant], alpha=0.12, linewidth=0)
            ax.grid(alpha=0.22)
            if col == 0:
                ax.set_ylabel(ylabel)
            if row_index == 0:
                ax.set_title(size.replace("_", "/"))
            if row_index == len(metric_specs) - 1:
                ax.set_xlabel("Cycle")
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle(f"LDPC dynamics at Eb/N0 = {target_ebno:g} dB", y=0.992)
    fig.legend(handles, legend_labels, loc="upper center", bbox_to_anchor=(0.5, 0.968),
               ncol=len(variants), frameon=False)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.055, top=0.925, hspace=0.13, wspace=0.20)
    for extension in ["png", "pdf"]:
        fig.savefig(output_dir / f"Fig_phase2_dynamics_by_size.{extension}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    bar_specs = [
        ("decoded_BER", "Decoded BER", True),
        ("late_syndrome_weight_fraction", "Late syndrome weight / M", False),
        ("late_backflip_rate", "Late correct→incorrect rate", False),
        ("late_q_used_source_product", r"Late $\langle q_t q^{\rm used}_{t-1}\rangle$", False),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.6))
    x = np.arange(len(sizes), dtype=float)
    width = 0.82 / max(1, len(variants))
    for ax, (metric, ylabel, log_scale) in zip(axes.flat, bar_specs):
        for variant_index, variant in enumerate(variants):
            means, errors = [], []
            for size in sizes:
                matches = [row for row in condition_rows if row["size"] == size and row["variant"] == variant
                           and abs(float(row["EbNo_dB"]) - target_ebno) < 1e-12]
                if not matches:
                    means.append(float("nan")); errors.append(float("nan")); continue
                row = matches[0]
                mean = float(row[f"{metric}_mean"])
                lo = float(row[f"{metric}_ci_low"])
                hi = float(row[f"{metric}_ci_high"])
                means.append(mean)
                errors.append(max(mean - lo, hi - mean))
            positions = x - 0.41 + width / 2 + variant_index * width
            ax.bar(positions, means, width=width, color=colors[variant], label=labels[variant],
                   yerr=errors, error_kw={"linewidth": 0.8, "capsize": 2})
        ax.set_xticks(x, [size.split("_")[0] for size in sizes])
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.22)
        if log_scale:
            ax.set_yscale("log")
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle(f"Final and late-time mechanism metrics at Eb/N0 = {target_ebno:g} dB", y=0.985)
    fig.legend(handles, legend_labels, loc="upper center", bbox_to_anchor=(0.5, 0.935),
               ncol=len(variants), frameon=False)
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.085, top=0.865, hspace=0.32, wspace=0.20)
    for extension in ["png", "pdf"]:
        fig.savefig(output_dir / f"Fig_phase2_mechanism_summary.{extension}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_invariants(matrix_seed: int = 0) -> dict[str, Any]:
    P = random_regular_ldpc_parity_check(48, 24, 3, 6, seed=matrix_seed)
    channel_values = np.linspace(-0.8, 0.8, P.shape[1])
    codeword = np.zeros(P.shape[1], dtype=np.uint8)
    params = {
        "alpha": 1.0,
        "alpha_mode": "fixed",
        "kw": 1.2,
        "kr": 1.5,
        "n_cycles": 24,
        "I0_min": 0.1,
        "I0_max": 0.8,
        "psa_p": 0.25,
        "lambda_mem": 0.7,
        "nrnd": 0.0,
        "response_lambda": 0.0,
        "lambda_out": 0.0,
        "I0_schedule_type": "linear",
        "I0_schedule_shape": 1.0,
        "I0_hold_fraction": 0.0,
        "decision_method": "best_state",
        "burn_in": 5,
        "sample_window": 12,
    }
    checks: dict[str, Any] = {}
    for variant in ["pSA", "additive", "normalized", "gain_only"]:
        core_rng = np.random.default_rng(123456)
        logger_rng = np.random.default_rng(123456)
        core = decode_pbits_fast(
            P=P,
            channel_values=channel_values,
            kw=params["kw"], kr=params["kr"], n_cycles=params["n_cycles"],
            mode=_mode_for_core(variant), I0_min=params["I0_min"], I0_max=params["I0_max"],
            psa_p=params["psa_p"], lambda_mem=0.0 if variant == "pSA" else params["lambda_mem"],
            I0_schedule_type=params["I0_schedule_type"], I0_schedule_shape=params["I0_schedule_shape"],
            I0_hold_fraction=params["I0_hold_fraction"], decision_method=params["decision_method"],
            burn_in=params["burn_in"], sample_window=params["sample_window"], rng=core_rng,
        )
        logger_params = dict(params)
        logger_params["lambda_mem"] = 0.0 if variant == "pSA" else params["lambda_mem"]
        logged, _, _ = _run_logged_trajectory(
            P, channel_values, codeword, logger_params, variant, 6, logger_rng,
            np.random.default_rng(999),
        )
        checks[f"{variant}_decoded_matches_phase1_fast"] = bool(np.array_equal(core, logged))

    zero_outputs = {}
    for variant in VARIANT_CODES:
        zero_params = dict(params)
        zero_params["lambda_mem"] = 0.0
        decoded, final_state, raw = _run_logged_trajectory(
            P, channel_values, codeword, zero_params, variant, 6,
            np.random.default_rng(54321), np.random.default_rng(777),
        )
        zero_outputs[variant] = (decoded, final_state, raw)
    for variant in VARIANT_CODES:
        checks[f"{variant}_lambda_zero_decoded_matches_pSA"] = bool(
            np.array_equal(zero_outputs["pSA"][0], zero_outputs[variant][0])
        )
        # Shuffling changes only the diagnostic columns that explicitly refer
        # to the supplied (permuted) source response.  State dynamics and all
        # self-trajectory diagnostics must still match pSA when lambda is zero.
        comparable_columns = [
            index for index in range(RAW_METRIC_COUNT)
            if index not in {R_USED_SOURCE_PRODUCT, R_SOURCE_MEAN_DELTA, R_SOURCE_SQ_DELTA}
        ]
        checks[f"{variant}_lambda_zero_metrics_match_pSA"] = bool(
            np.array_equal(
                zero_outputs["pSA"][2][:, comparable_columns],
                zero_outputs[variant][2][:, comparable_columns],
            )
        )

    order, rank, offsets = _shuffle_mapping(48, 24, np.random.default_rng(2468))
    no_self = True
    max_mean_delta = 0.0
    max_second_delta = 0.0
    q = np.random.default_rng(1357).normal(size=48)
    for offset in offsets:
        source = np.empty_like(q)
        for bit in range(len(q)):
            source_bit = order[(rank[bit] + int(offset)) % len(q)]
            no_self = no_self and source_bit != bit
            source[bit] = q[source_bit]
        max_mean_delta = max(max_mean_delta, abs(float(np.mean(source) - np.mean(q))))
        max_second_delta = max(max_second_delta, abs(float(np.mean(source * source) - np.mean(q * q))))
    checks["shuffled_mapping_has_no_self_links"] = bool(no_self)
    checks["shuffled_mapping_preserves_response_mean"] = bool(max_mean_delta < 1e-15)
    checks["shuffled_mapping_preserves_response_second_moment"] = bool(max_second_delta < 1e-15)
    checks["shuffle_max_abs_mean_delta"] = max_mean_delta
    checks["shuffle_max_abs_second_moment_delta"] = max_second_delta
    boolean_checks = [value for key, value in checks.items() if key not in {
        "shuffle_max_abs_mean_delta", "shuffle_max_abs_second_moment_delta"
    }]
    checks["all_passed"] = bool(all(boolean_checks))
    return checks


def _setup_logging(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger("phase2_ldpc_dynamics")
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


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Phase 2 LDPC dynamics mechanism experiments.")
    parser.add_argument("--profile", choices=sorted(PROFILE_DEFAULTS), default="smoke")
    parser.add_argument("--sizes", default=None, help="Comma-separated N96_M48,N192_M96,N288_M144 or all")
    parser.add_argument("--variants", default=None, help="Comma-separated pSA,additive,normalized,gain_only,shuffled")
    parser.add_argument("--ebno-values", default=None, help="Comma-separated Eb/N0 values in dB")
    parser.add_argument("--trials", type=int, default=None, help="Total trajectories per Eb/N0 and condition")
    parser.add_argument("--seed-count", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--matrix-seed", type=int, default=0)
    parser.add_argument("--n-workers", type=int, default=8)
    parser.add_argument("--cycles-override", type=int, default=None)
    parser.add_argument("--cycle-bin", type=int, default=None)
    parser.add_argument("--fixed-bit-width", type=int, default=8)
    parser.add_argument("--channel-input-mode", choices=["float", "fixed"], default="float")
    parser.add_argument("--source-root", action="append", default=None)
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "phase2_results"))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    defaults = PROFILE_DEFAULTS[args.profile]
    if args.sizes is None:
        sizes = list(defaults["sizes"])
    elif args.sizes == "all":
        sizes = list(SIZE_SPECS)
    else:
        sizes = parse_list(args.sizes)
    variants = parse_list(args.variants) if args.variants else list(defaults["variants"])
    ebno_values = parse_list(args.ebno_values, float) if args.ebno_values else list(defaults["ebno_values"])
    trials = int(args.trials if args.trials is not None else defaults["trials"])
    seed_count = int(args.seed_count if args.seed_count is not None else defaults["seed_count"])
    cycles_override = args.cycles_override if args.cycles_override is not None else defaults["cycles_override"]
    cycle_bin = int(args.cycle_bin if args.cycle_bin is not None else defaults["cycle_bin"])
    if set(sizes) - set(SIZE_SPECS):
        raise ValueError(f"Unknown sizes: {sorted(set(sizes) - set(SIZE_SPECS))}")
    if set(variants) - set(VARIANT_CODES):
        raise ValueError(f"Unknown variants: {sorted(set(variants) - set(VARIANT_CODES))}")
    if trials <= 0 or seed_count <= 0 or cycle_bin <= 0 or not ebno_values:
        raise ValueError("trials, seed-count, cycle-bin, and Eb/N0 list must be positive/nonempty")
    seeds = [int(args.seed) + 8022 * index for index in range(seed_count)]

    roots = result_roots(args.source_root)
    sources = {size: locate_completed_size(size, roots) for size in sizes}
    base_params = {}
    source_files = {}
    references = {}
    for size in sizes:
        params, path = load_anchor(sources[size], "lambda")
        base_params[size] = params
        source_files[size] = str(path)
        references[size] = load_reference(sources[size])
        available = {float(row["EbNo_dB"]) for row in references[size]}
        missing = sorted(set(ebno_values) - available)
        if missing:
            raise ValueError(f"{size} source reference lacks Eb/N0 values: {missing}")

    run_name = args.run_name or f"{args.profile}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(args.output_dir).expanduser().resolve() / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = _setup_logging(run_dir)
    config = {
        "profile": args.profile,
        "sizes": sizes,
        "variants": variants,
        "ebno_values": ebno_values,
        "trials_per_ebno": trials,
        "seed_count": seed_count,
        "seeds": seeds,
        "matrix_seed": int(args.matrix_seed),
        "n_workers": int(args.n_workers),
        "cycles_override": cycles_override,
        "cycle_bin": cycle_bin,
        "fixed_bit_width": int(args.fixed_bit_width),
        "channel_input_mode": args.channel_input_mode,
        "source_parameter_files": source_files,
        "final_confirmation_lambdas": {size: FINAL_LAMBDAS[size] for size in sizes},
        "shuffle_definition": (
            "cycle-wise nonzero rotation in an independently random bit relabeling; "
            "a derangement preserving the full previous-response empirical distribution"
        ),
        "command": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
    }
    digest = config_hash(config)
    config_path = run_dir / "effective_config.json"
    if config_path.exists() and args.resume:
        previous = json.loads(config_path.read_text())
        if previous.get("config_hash") != digest:
            raise RuntimeError(f"Refusing to resume changed configuration in {run_dir}")
    write_json(config_path, {**config, "config_hash": digest})

    invariants = run_invariants(int(args.matrix_seed))
    write_json(run_dir / "invariants.json", invariants)
    if not invariants["all_passed"]:
        raise RuntimeError("Phase 2 implementation invariant failed; see invariants.json")

    conditions = []
    plan_rows = []
    total_updates = 0
    condition_id = 0
    for size in sizes:
        for variant in variants:
            condition_id += 1
            lam = 0.0 if variant == "pSA" else FINAL_LAMBDAS[size]
            params = adjusted_params(base_params[size], lam, cycles_override)
            condition = {
                "condition_id": condition_id,
                "size": size,
                "variant": variant,
                "implementation_mode": IMPLEMENTATION_MODES[variant],
                "formula": FORMULAS[variant],
                "lambda_mem": lam,
                "params": params,
                "source_parameter_file": source_files[size],
            }
            conditions.append(condition)
            updates = trials * len(ebno_values) * int(params["n_cycles"]) * SIZE_SPECS[size]["n_bits"]
            total_updates += updates
            plan_rows.append({
                "condition_id": condition_id,
                "size": size,
                "variant": variant,
                "implementation_mode": IMPLEMENTATION_MODES[variant],
                "lambda_mem": lam,
                "EbNo_dB_values": ";".join(f"{value:g}" for value in ebno_values),
                "trials_per_ebno": trials,
                "seed_count": seed_count,
                "n_cycles": params["n_cycles"],
                "cycle_bin": cycle_bin,
                "estimated_bit_updates": updates,
                "source_parameter_file": source_files[size],
            })
    write_csv(run_dir / "experiment_plan.csv", plan_rows, _fields_for(plan_rows))
    manifest = {
        "status": "planned" if args.dry_run else "running",
        "started_utc": _utc_now(),
        "finished_utc": None,
        "run_dir": str(run_dir),
        "config_hash": digest,
        "environment": environment_record(),
        "condition_count": len(conditions),
        "completed_conditions": 0,
        "estimated_bit_updates": total_updates,
    }
    write_json(run_dir / "manifest.json", manifest)
    logger.info("Run directory: %s", run_dir)
    logger.info("Conditions: %d; estimated bit updates: %.6g", len(conditions), total_updates)
    logger.info("Common Phase 1 seed series: %s", seeds)
    if args.dry_run:
        logger.info("Dry run complete")
        return

    start_time = time.perf_counter()
    all_cycle_summaries = []
    all_seed_cycle_summaries = []
    all_trajectory_rows = []
    completed = 0
    for index, condition in enumerate(conditions, start=1):
        stem = f"{condition['size']}__{condition['variant']}"
        condition_dir = run_dir / "trajectories" / condition["size"] / condition["variant"]
        cycle_path = condition_dir / "trajectory_by_cycle.csv.gz"
        trajectory_path = condition_dir / "trajectory_summary.csv"
        cycle_summary_path = condition_dir / "by_cycle_summary.csv"
        seed_cycle_path = condition_dir / "by_seed_cycle.csv"
        done_path = condition_dir / "complete.json"
        if args.resume and done_path.exists():
            done = json.loads(done_path.read_text())
            if done.get("config_hash") == digest:
                logger.info("[%d/%d] resume skip %s", index, len(conditions), stem)
                trajectory_rows = _read_csv(trajectory_path)
                cycle_summaries = _read_csv(cycle_summary_path)
                seed_cycle_summaries = _read_csv(seed_cycle_path)
                all_trajectory_rows.extend(trajectory_rows)
                all_cycle_summaries.extend(cycle_summaries)
                all_seed_cycle_summaries.extend(seed_cycle_summaries)
                completed += 1
                continue

        condition_start = time.perf_counter()
        P = random_regular_ldpc_parity_check(
            SIZE_SPECS[condition["size"]]["n_bits"],
            SIZE_SPECS[condition["size"]]["n_checks"],
            3, 6, seed=int(args.matrix_seed),
        )
        seed_trial_pairs = split_total_trials(trials, seeds)
        tasks = [{
            "P": P,
            "condition": condition,
            "seed": seed,
            "n_trials": count,
            "ebno_values": ebno_values,
            "cycle_bin": cycle_bin,
            "fixed_bit_width": int(args.fixed_bit_width),
            "channel_input_mode": args.channel_input_mode,
        } for seed, count in seed_trial_pairs]
        logger.info(
            "[%d/%d] %s variant=%s lambda=%.3g trials/EbNo=%d cycles=%d",
            index, len(conditions), condition["size"], condition["variant"],
            condition["lambda_mem"], trials, condition["params"]["n_cycles"],
        )
        cycle_rows = []
        trajectory_rows = []
        max_workers = min(int(args.n_workers), len(tasks))
        if max_workers <= 1:
            results = [_seed_task(task) for task in tasks]
        else:
            results = []
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_map = {executor.submit(_seed_task, task): task["seed"] for task in tasks}
                for future in as_completed(future_map):
                    results.append(future.result())
        for result in results:
            cycle_rows.extend(result["cycle_rows"])
            trajectory_rows.extend(result["trajectory_rows"])
        cycle_rows.sort(key=lambda row: (float(row["EbNo_dB"]), int(row["seed"]),
                                         int(row["trial_index"]), int(row["cycle_start"])))
        trajectory_rows.sort(key=lambda row: (float(row["EbNo_dB"]), int(row["seed"]),
                                              int(row["trial_index"])))
        cycle_summaries = _aggregate_cycle_rows(cycle_rows, by_seed=False)
        seed_cycle_summaries = _aggregate_cycle_rows(cycle_rows, by_seed=True)
        condition_dir.mkdir(parents=True, exist_ok=True)
        _atomic_gzip_csv(cycle_path, cycle_rows, CYCLE_FIELDS)
        write_csv(trajectory_path, trajectory_rows, TRAJECTORY_FIELDS)
        write_csv(cycle_summary_path, cycle_summaries, _fields_for(cycle_summaries))
        write_csv(seed_cycle_path, seed_cycle_summaries, _fields_for(seed_cycle_summaries))
        elapsed = time.perf_counter() - condition_start
        write_json(done_path, {
            "status": "complete", "config_hash": digest, "finished_utc": _utc_now(),
            "elapsed_seconds": elapsed, "trajectory_count": len(trajectory_rows),
            "cycle_row_count": len(cycle_rows),
        })
        all_trajectory_rows.extend(trajectory_rows)
        all_cycle_summaries.extend(cycle_summaries)
        all_seed_cycle_summaries.extend(seed_cycle_summaries)
        completed += 1
        manifest["completed_conditions"] = completed
        write_json(run_dir / "manifest.json", manifest)
        logger.info("[%d/%d] complete in %.2f s; trajectories=%d", index, len(conditions), elapsed,
                    len(trajectory_rows))

    # Coerce resumed CSV rows through float-capable aggregation functions.
    condition_summaries = _condition_summaries(all_trajectory_rows, by_seed=False)
    seed_condition_summaries = _condition_summaries(all_trajectory_rows, by_seed=True)
    paired_statistics = _paired_statistics(all_trajectory_rows)
    write_csv(run_dir / "trajectory_summary.csv", all_trajectory_rows, TRAJECTORY_FIELDS)
    write_csv(run_dir / "by_cycle_summary.csv", all_cycle_summaries, _fields_for(all_cycle_summaries))
    write_csv(run_dir / "by_seed_cycle.csv", all_seed_cycle_summaries, _fields_for(all_seed_cycle_summaries))
    write_csv(run_dir / "condition_summary.csv", condition_summaries, _fields_for(condition_summaries))
    write_csv(run_dir / "by_seed_condition.csv", seed_condition_summaries, _fields_for(seed_condition_summaries))
    write_csv(run_dir / "paired_statistics.csv", paired_statistics, _fields_for(paired_statistics))
    _make_figures(all_cycle_summaries, condition_summaries, run_dir / "figures")

    total_elapsed = time.perf_counter() - start_time
    runtime = {
        "elapsed_seconds_this_invocation": total_elapsed,
        "completed_conditions": completed,
        "estimated_bit_updates": total_updates,
        "bit_updates_per_second_including_metrics_and_io": total_updates / max(total_elapsed, 1e-12),
        "trajectory_count": len(all_trajectory_rows),
    }
    write_json(run_dir / "runtime_summary.json", runtime)
    manifest.update({"status": "complete", "finished_utc": _utc_now(),
                     "completed_conditions": completed, "runtime_summary": runtime})
    write_json(run_dir / "manifest.json", manifest)
    logger.info("Done in %.2f s. Outputs: %s", total_elapsed, run_dir)


if __name__ == "__main__":
    main()

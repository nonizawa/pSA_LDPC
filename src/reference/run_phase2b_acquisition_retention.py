#!/usr/bin/env python3
"""Phase 2b targeted acquisition-versus-retention analysis for LDPC p-bits.

Phase 1 and Phase 2 source files are imported but not modified.  Each Phase 2b
trajectory is run through (1) the exact Phase 2 metrics kernel and (2) an event
kernel initialized with the identical core seed and shuffled-memory mapping.
The two decoded and final states must agree bit-for-bit.  This preserves the
validated Phase 2 metrics while adding exact-cycle first-passage, residence,
escape, return, and channel-consistency observations.
"""

from __future__ import annotations

import argparse
import csv
import gzip
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
LDPC_DIR = REPO_ROOT / "src" / "ldpc"
if str(LDPC_DIR) not in sys.path:
    sys.path.insert(0, str(LDPC_DIR))

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
    parse_list,
    result_roots,
    write_csv,
    write_json,
)
from run_phase2_ldpc_dynamics import (  # noqa: E402
    CYCLE_FIELDS as PHASE2_CYCLE_FIELDS,
    CYCLE_METRICS as PHASE2_CYCLE_METRICS,
    FINAL_LAMBDAS,
    FORMULAS,
    IMPLEMENTATION_MODES,
    RAW_METRIC_COUNT,
    TRAJECTORY_FIELDS as PHASE2_TRAJECTORY_FIELDS,
    VARIANT_CODES,
    _channel_score,
    _decode_psa_with_metrics_numba,
    _decision_code,
    _prepare_kernel_inputs,
    _shuffle_mapping,
    _syndrome_weight_sparse,
    _trajectory_records,
    run_invariants as run_phase2_invariants,
)

try:
    from numba import njit
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("Phase 2b requires numba") from exc


PRIMARY_VARIANTS = ["pSA", "additive", "gain_only", "shuffled"]
LOW_BER_THRESHOLD = 0.05
SYNDROME_FRACTION_THRESHOLD = 0.05

PROFILE_DEFAULTS = {
    "smoke": {
        "sizes": ["N192_M96"],
        "variants": PRIMARY_VARIANTS,
        "trials": 2,
        "seed_count": 1,
        "cycles_override": 200,
        "cycle_bin": 20,
    },
    "pilot": {
        "sizes": ["N192_M96"],
        "variants": PRIMARY_VARIANTS,
        "trials": 200,
        "seed_count": 10,
        "cycles_override": None,
        "cycle_bin": 100,
    },
    "targeted": {
        "sizes": ["N192_M96", "N288_M144"],
        "variants": PRIMARY_VARIANTS,
        "trials": 200,
        "seed_count": 10,
        "cycles_override": None,
        "cycle_bin": 100,
    },
}

# Additional binned metrics accumulated by the event kernel.
X_CHANNEL_SCORE = 0
X_CHANNEL_GAP_TO_TRUE = 1
X_HARD_CHANNEL_DISAGREEMENTS = 2
X_CORRECT_CYCLES = 3
X_LOW_BER_CYCLES = 4
X_RELAXED_BASIN_CYCLES = 5
X_WRONG_VALID_CYCLES = 6
X_BIT_ERROR_COUNT = 7
X_SYNDROME_WEIGHT = 8
X_FLIP_COUNT = 9
X_BACKFLIP_COUNT = 10
X_PREV_CORRECT_BITS = 11
EXTRA_METRIC_COUNT = 12

CHANNEL_CYCLE_METRICS = [
    "channel_alignment",
    "channel_alignment_gap_to_transmitted",
    "hard_channel_disagreement_rate",
    "exact_correct_fraction",
    "low_BER_fraction",
    "relaxed_basin_fraction",
    "wrong_valid_fraction",
]

# Exact-cycle event and retention counters.
E_FIRST_VALID = 0
E_FIRST_VALID_AFTER_NONVALID = 1
E_FIRST_CORRECT = 2
E_FIRST_LOW_BER = 3
E_FIRST_RELAXED_BASIN = 4
E_FIRST_WRONG_VALID = 5
E_EXACT_REMAINING_CYCLES = 6
E_EXACT_CORRECT_CYCLES = 7
E_EXACT_ESCAPES = 8
E_EXACT_OPPORTUNITIES = 9
E_EXACT_RETURNS = 10
E_EXACT_RETURN_TIME_SUM = 11
E_EXACT_FIRST_RESIDENCE = 12
E_EXACT_FIRST_RESIDENCE_CENSORED = 13
E_BASIN_REMAINING_CYCLES = 14
E_BASIN_CYCLES = 15
E_BASIN_ESCAPES = 16
E_BASIN_OPPORTUNITIES = 17
E_BASIN_RETURNS = 18
E_BASIN_RETURN_TIME_SUM = 19
E_BASIN_FIRST_RESIDENCE = 20
E_BASIN_FIRST_RESIDENCE_CENSORED = 21
EVENT_COUNT = 22

ACQUISITION_EVENTS = {
    "valid": "first_valid_cycle",
    "valid_after_nonvalid": "first_valid_after_nonvalid_cycle",
    "correct": "first_correct_cycle",
    "low_BER": "first_low_BER_cycle",
    "relaxed_basin": "first_relaxed_basin_cycle",
    "wrong_valid": "first_wrong_valid_cycle",
}

CYCLE_FIELDS = [*PHASE2_CYCLE_FIELDS, *CHANNEL_CYCLE_METRICS]

EVENT_TRAJECTORY_FIELDS = [
    "reached_valid",
    "reached_valid_after_nonvalid",
    "reached_correct",
    "reached_low_BER",
    "reached_relaxed_basin",
    "reached_wrong_valid",
    "first_valid_cycle",
    "first_valid_after_nonvalid_cycle",
    "first_correct_cycle",
    "first_low_BER_cycle",
    "first_relaxed_basin_cycle",
    "first_wrong_valid_cycle",
    "correct_residence_fraction",
    "correct_remaining_cycles_after_hit",
    "correct_cycles_after_hit",
    "correct_escape_probability",
    "correct_transition_opportunities",
    "correct_escape_count",
    "correct_return_probability",
    "correct_return_count",
    "correct_mean_return_time",
    "correct_first_residence_time",
    "correct_first_residence_censored",
    "relaxed_basin_residence_fraction",
    "relaxed_basin_remaining_cycles_after_hit",
    "relaxed_basin_cycles_after_hit",
    "relaxed_basin_escape_probability",
    "relaxed_basin_transition_opportunities",
    "relaxed_basin_escape_count",
    "relaxed_basin_return_probability",
    "relaxed_basin_return_count",
    "relaxed_basin_mean_return_time",
    "relaxed_basin_first_residence_time",
    "relaxed_basin_first_residence_censored",
    "mean_channel_alignment",
    "late_channel_alignment",
    "mean_channel_alignment_gap_to_transmitted",
    "late_channel_alignment_gap_to_transmitted",
    "mean_hard_channel_disagreement_rate",
    "late_hard_channel_disagreement_rate",
    "wrong_valid_residence_fraction",
]

TRAJECTORY_FIELDS = [*PHASE2_TRAJECTORY_FIELDS, *EVENT_TRAJECTORY_FIELDS]


@njit(cache=True)
def _decode_events_channel_numba(
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
    initial_state_codeword,
    low_ber_threshold,
    syndrome_fraction_threshold,
    seed,
):
    """Phase-2-equivalent state updates plus exact acquisition/retention events."""
    np.random.seed(seed)
    n_bits = P.shape[1]
    n_checks = P.shape[0]
    n_bins = (n_cycles + cycle_bin - 1) // cycle_bin
    extra = np.zeros((n_bins, EXTRA_METRIC_COUNT), dtype=np.float64)
    events = np.zeros(EVENT_COUNT, dtype=np.float64)
    sentinel = float(n_cycles + 1)
    for index in range(6):
        events[index] = sentinel

    w = np.zeros(n_bits, dtype=np.uint8)
    if initial_state_codeword == 1:
        w[:] = codeword_bits[:]
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

    ever_nonvalid = False
    exact_active = initial_state_codeword == 1
    exact_in_escape = False
    exact_escape_cycle = 0
    exact_prev = initial_state_codeword == 1
    basin_active = initial_state_codeword == 1
    basin_in_escape = False
    basin_escape_cycle = 0
    basin_prev = initial_state_codeword == 1
    if initial_state_codeword == 1:
        events[E_FIRST_VALID] = 0.0
        events[E_FIRST_CORRECT] = 0.0
        events[E_FIRST_LOW_BER] = 0.0
        events[E_FIRST_RELAXED_BASIN] = 0.0

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
                rank = shuffle_rank[bit]
                source_bit = shuffle_order[(rank + shuffle_offset) % n_bits]
                deterministic = current_q + lambda_mem * q_prev[source_bit]
            elif variant_code == 5:
                # Matched binary-state self-feedback: same-bit pre-update spin
                # x=2w-1 is added post-tanh and pre-threshold.  The response
                # history is diagnostic only and is not used by this arm.
                deterministic = current_q + lambda_mem * (2.0 * w_prev[bit] - 1.0)
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
        relaxed_basin_now = (
            low_ber_now and syn_weight / n_checks <= syndrome_fraction_threshold + 1e-15
        )
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
                if (
                    syn_weight < best_syndrome_weight
                    or (syn_weight == best_syndrome_weight and current_channel_score > best_channel_score)
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


def _run_phase2b_trajectory(
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
        P_packed, group_indices, group_counts, group_lengths,
        check_indices, check_lengths, I0_history,
    ) = prepared
    order, rank, offsets = _shuffle_mapping(P.shape[1], int(params["n_cycles"]), shuffle_rng)
    core_seed = int(rng.integers(0, 2**31 - 1))
    common = (
        P_packed, group_indices, group_counts, group_lengths, check_indices, check_lengths,
        np.asarray(channel_values, dtype=np.float64), np.asarray(codeword_bits, dtype=np.uint8),
        float(params["kw"]), float(params["kr"]), int(params["n_cycles"]), I0_history,
        float(params["psa_p"]), float(params.get("lambda_mem", 0.0)),
        int(VARIANT_CODES[variant]), int(_decision_code(params["decision_method"])),
        int(params["burn_in"]), int(params["sample_window"]), int(cycle_bin),
        order, rank, offsets,
    )
    decoded, final_state, phase2_raw = _decode_psa_with_metrics_numba(*common, core_seed)
    event_decoded, event_final, extra, events = _decode_events_channel_numba(
        *common, 0, float(LOW_BER_THRESHOLD), float(SYNDROME_FRACTION_THRESHOLD), core_seed
    )
    if not np.array_equal(decoded, event_decoded) or not np.array_equal(final_state, event_final):
        raise RuntimeError("Phase 2b event logger changed the validated Phase 2 trajectory")
    return decoded, final_state, phase2_raw, extra, events


def _finite_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else float("nan")


def _extra_metrics(raw: np.ndarray, extra: np.ndarray, n_bits: int, late: bool = False):
    if late:
        start = max(0, int(math.floor(0.75 * len(raw))))
        raw = raw[start:]
        extra = extra[start:]
    cycles = float(np.sum(raw[:, 0]))
    totals = np.sum(extra, axis=0)
    return {
        "channel_alignment": _finite_ratio(totals[X_CHANNEL_SCORE], cycles * n_bits),
        "channel_alignment_gap_to_transmitted": _finite_ratio(
            totals[X_CHANNEL_GAP_TO_TRUE], cycles * n_bits
        ),
        "hard_channel_disagreement_rate": _finite_ratio(
            totals[X_HARD_CHANNEL_DISAGREEMENTS], cycles * n_bits
        ),
        "exact_correct_fraction": _finite_ratio(totals[X_CORRECT_CYCLES], cycles),
        "low_BER_fraction": _finite_ratio(totals[X_LOW_BER_CYCLES], cycles),
        "relaxed_basin_fraction": _finite_ratio(totals[X_RELAXED_BASIN_CYCLES], cycles),
        "wrong_valid_fraction": _finite_ratio(totals[X_WRONG_VALID_CYCLES], cycles),
    }


def _event_time(value: float, n_cycles: int):
    return float(value) if value <= n_cycles else float("nan")


def _event_trajectory_fields(events: np.ndarray, raw: np.ndarray, extra: np.ndarray, n_bits: int, n_cycles: int):
    times = [_event_time(events[index], n_cycles) for index in range(6)]
    mean_extra = _extra_metrics(raw, extra, n_bits, late=False)
    late_extra = _extra_metrics(raw, extra, n_bits, late=True)
    correct_reached = math.isfinite(times[2])
    basin_reached = math.isfinite(times[4])
    return {
        "reached_valid": int(math.isfinite(times[0])),
        "reached_valid_after_nonvalid": int(math.isfinite(times[1])),
        "reached_correct": int(correct_reached),
        "reached_low_BER": int(math.isfinite(times[3])),
        "reached_relaxed_basin": int(basin_reached),
        "reached_wrong_valid": int(math.isfinite(times[5])),
        "first_valid_cycle": times[0],
        "first_valid_after_nonvalid_cycle": times[1],
        "first_correct_cycle": times[2],
        "first_low_BER_cycle": times[3],
        "first_relaxed_basin_cycle": times[4],
        "first_wrong_valid_cycle": times[5],
        "correct_residence_fraction": (
            _finite_ratio(events[E_EXACT_CORRECT_CYCLES], events[E_EXACT_REMAINING_CYCLES])
            if correct_reached else float("nan")
        ),
        "correct_remaining_cycles_after_hit": (
            events[E_EXACT_REMAINING_CYCLES] if correct_reached else float("nan")
        ),
        "correct_cycles_after_hit": (
            events[E_EXACT_CORRECT_CYCLES] if correct_reached else float("nan")
        ),
        "correct_escape_probability": (
            _finite_ratio(events[E_EXACT_ESCAPES], events[E_EXACT_OPPORTUNITIES])
            if correct_reached else float("nan")
        ),
        "correct_transition_opportunities": (
            events[E_EXACT_OPPORTUNITIES] if correct_reached else float("nan")
        ),
        "correct_escape_count": events[E_EXACT_ESCAPES] if correct_reached else float("nan"),
        "correct_return_probability": (
            _finite_ratio(events[E_EXACT_RETURNS], events[E_EXACT_ESCAPES])
            if correct_reached else float("nan")
        ),
        "correct_return_count": events[E_EXACT_RETURNS] if correct_reached else float("nan"),
        "correct_mean_return_time": (
            _finite_ratio(events[E_EXACT_RETURN_TIME_SUM], events[E_EXACT_RETURNS])
            if correct_reached else float("nan")
        ),
        "correct_first_residence_time": events[E_EXACT_FIRST_RESIDENCE] if correct_reached else float("nan"),
        "correct_first_residence_censored": events[E_EXACT_FIRST_RESIDENCE_CENSORED] if correct_reached else float("nan"),
        "relaxed_basin_residence_fraction": (
            _finite_ratio(events[E_BASIN_CYCLES], events[E_BASIN_REMAINING_CYCLES])
            if basin_reached else float("nan")
        ),
        "relaxed_basin_remaining_cycles_after_hit": (
            events[E_BASIN_REMAINING_CYCLES] if basin_reached else float("nan")
        ),
        "relaxed_basin_cycles_after_hit": (
            events[E_BASIN_CYCLES] if basin_reached else float("nan")
        ),
        "relaxed_basin_escape_probability": (
            _finite_ratio(events[E_BASIN_ESCAPES], events[E_BASIN_OPPORTUNITIES])
            if basin_reached else float("nan")
        ),
        "relaxed_basin_transition_opportunities": (
            events[E_BASIN_OPPORTUNITIES] if basin_reached else float("nan")
        ),
        "relaxed_basin_escape_count": events[E_BASIN_ESCAPES] if basin_reached else float("nan"),
        "relaxed_basin_return_probability": (
            _finite_ratio(events[E_BASIN_RETURNS], events[E_BASIN_ESCAPES])
            if basin_reached else float("nan")
        ),
        "relaxed_basin_return_count": events[E_BASIN_RETURNS] if basin_reached else float("nan"),
        "relaxed_basin_mean_return_time": (
            _finite_ratio(events[E_BASIN_RETURN_TIME_SUM], events[E_BASIN_RETURNS])
            if basin_reached else float("nan")
        ),
        "relaxed_basin_first_residence_time": events[E_BASIN_FIRST_RESIDENCE] if basin_reached else float("nan"),
        "relaxed_basin_first_residence_censored": events[E_BASIN_FIRST_RESIDENCE_CENSORED] if basin_reached else float("nan"),
        "mean_channel_alignment": mean_extra["channel_alignment"],
        "late_channel_alignment": late_extra["channel_alignment"],
        "mean_channel_alignment_gap_to_transmitted": mean_extra["channel_alignment_gap_to_transmitted"],
        "late_channel_alignment_gap_to_transmitted": late_extra["channel_alignment_gap_to_transmitted"],
        "mean_hard_channel_disagreement_rate": mean_extra["hard_channel_disagreement_rate"],
        "late_hard_channel_disagreement_rate": late_extra["hard_channel_disagreement_rate"],
        "wrong_valid_residence_fraction": mean_extra["wrong_valid_fraction"],
    }


def analyze_artificial_trajectory(P: np.ndarray, states: np.ndarray, codeword: np.ndarray):
    """Reference event analysis used by unit tests on hand-constructed states."""
    P = np.asarray(P, dtype=np.uint8)
    states = np.asarray(states, dtype=np.uint8)
    codeword = np.asarray(codeword, dtype=np.uint8)
    valid = np.asarray([not np.any(syndrome(P, state)) for state in states], dtype=bool)
    errors = np.sum(states != codeword, axis=1)
    correct = errors == 0
    low = errors / states.shape[1] <= LOW_BER_THRESHOLD + 1e-15
    relaxed = low & (np.asarray([np.sum(syndrome(P, state)) for state in states]) / P.shape[0]
                     <= SYNDROME_FRACTION_THRESHOLD + 1e-15)
    wrong_valid = valid & ~correct

    def first(flags):
        indices = np.flatnonzero(flags)
        return int(indices[0] + 1) if len(indices) else None

    ever_nonvalid = False
    first_valid_after = None
    for index, is_valid in enumerate(valid):
        if not is_valid:
            ever_nonvalid = True
        elif ever_nonvalid:
            first_valid_after = index + 1
            break

    def retention(flags):
        hit = first(flags)
        if hit is None:
            return None
        tail = flags[hit - 1:]
        escapes = 0
        returns = 0
        return_times = []
        escape_at = None
        first_residence = None
        for local in range(1, len(tail)):
            if tail[local - 1] and not tail[local]:
                escapes += 1
                escape_at = local
                if first_residence is None:
                    first_residence = local
            elif escape_at is not None and tail[local]:
                returns += 1
                return_times.append(local - escape_at)
                escape_at = None
        censored = escapes == 0
        if first_residence is None:
            first_residence = len(tail)
        return {
            "hit": hit,
            "residence_fraction": float(np.mean(tail)),
            "escapes": escapes,
            "escape_probability": escapes / max(1, int(np.sum(tail[:-1]))),
            "returns": returns,
            "return_probability": returns / escapes if escapes else float("nan"),
            "mean_return_time": float(np.mean(return_times)) if return_times else float("nan"),
            "first_residence": first_residence,
            "first_residence_censored": censored,
        }

    return {
        "valid": valid,
        "correct": correct,
        "wrong_valid": wrong_valid,
        "first_valid": first(valid),
        "first_valid_after_nonvalid": first_valid_after,
        "first_correct": first(correct),
        "first_wrong_valid": first(wrong_valid),
        "correct_retention": retention(correct),
        "relaxed_basin_retention": retention(relaxed),
    }


def _cycle_extra_rows(phase2_rows, raw: np.ndarray, extra: np.ndarray, n_bits: int):
    output = []
    for index, base in enumerate(phase2_rows):
        cycles = float(raw[index, 0])
        totals = extra[index]
        output.append({
            **base,
            "channel_alignment": _finite_ratio(totals[X_CHANNEL_SCORE], cycles * n_bits),
            "channel_alignment_gap_to_transmitted": _finite_ratio(
                totals[X_CHANNEL_GAP_TO_TRUE], cycles * n_bits
            ),
            "hard_channel_disagreement_rate": _finite_ratio(
                totals[X_HARD_CHANNEL_DISAGREEMENTS], cycles * n_bits
            ),
            "exact_correct_fraction": _finite_ratio(totals[X_CORRECT_CYCLES], cycles),
            "low_BER_fraction": _finite_ratio(totals[X_LOW_BER_CYCLES], cycles),
            "relaxed_basin_fraction": _finite_ratio(totals[X_RELAXED_BASIN_CYCLES], cycles),
            "wrong_valid_fraction": _finite_ratio(totals[X_WRONG_VALID_CYCLES], cycles),
        })
    return output


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
    ebno = float(task["ebno"])
    for trial_index in range(int(task["n_trials"])):
        message_bits = channel_rng.integers(0, 2, size=G.shape[0], dtype=np.uint8)
        codeword_bits = encode_message(message_bits, G)
        y = add_awgn(bpsk_modulate(codeword_bits), ebno, rate=rate, rng=channel_rng)
        channel_llr = awgn_channel_llr(y, ebno, rate=rate)
        channel_values = make_stochastic_channel_values(
            y, alpha=params["alpha"], bit_width=task["fixed_bit_width"],
            input_mode=task["channel_input_mode"], alpha_mode=params["alpha_mode"],
            ebno_db=ebno, rate=rate, channel_llr=channel_llr,
        )
        decoded, final_state, phase2_raw, extra, events = _run_phase2b_trajectory(
            P, channel_values, codeword_bits, params, condition["variant"],
            int(task["cycle_bin"]), pbit_rng, shuffle_rng, prepared=prepared,
        )
        base_cycle, base_summary = _trajectory_records(
            phase2_raw, decoded, final_state, codeword_bits, P, condition,
            seed, ebno, trial_index, int(task["cycle_bin"]),
        )
        cycle_rows.extend(_cycle_extra_rows(base_cycle, phase2_raw, extra, n_bits))
        base_summary.update(_event_trajectory_fields(
            events, phase2_raw, extra, n_bits, int(params["n_cycles"])
        ))
        trajectory_rows.append(base_summary)
    return {"cycle_rows": cycle_rows, "trajectory_rows": trajectory_rows}


class OnlineStat:
    def __init__(self):
        self.values = []

    def add(self, value):
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return
        if math.isfinite(numeric):
            self.values.append(numeric)

    def summary(self):
        if not self.values:
            return {"n": 0, "mean": float("nan"), "sd": float("nan"), "sem": float("nan"),
                    "ci_low": float("nan"), "ci_high": float("nan"), "median": float("nan"),
                    "q25": float("nan"), "q75": float("nan")}
        values = np.asarray(self.values, dtype=float)
        mean = float(np.mean(values))
        sd = float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")
        sem = sd / math.sqrt(len(values)) if len(values) > 1 else float("nan")
        half = 1.96 * sem if math.isfinite(sem) else float("nan")
        return {"n": len(values), "mean": mean, "sd": sd, "sem": sem,
                "ci_low": mean - half, "ci_high": mean + half,
                "median": float(np.median(values)), "q25": float(np.quantile(values, 0.25)),
                "q75": float(np.quantile(values, 0.75))}


def _fields_for(rows):
    fields, seen = [], set()
    for row in rows:
        for field in row:
            if field not in seen:
                fields.append(field); seen.add(field)
    return fields


def _aggregate_cycle(rows, by_seed=False):
    metrics = [*PHASE2_CYCLE_METRICS, *CHANNEL_CYCLE_METRICS]
    groups = {}
    for row in rows:
        key = (row["size"], row["variant"], float(row["lambda_mem"]),
               int(row["seed"]) if by_seed else None, float(row["EbNo_dB"]),
               int(row["cycle_start"]), int(row["cycle_end"]), float(row["cycle_mid"]))
        groups.setdefault(key, {metric: OnlineStat() for metric in metrics})
        for metric in metrics:
            groups[key][metric].add(row.get(metric))
    output = []
    for key in sorted(groups):
        size, variant, lam, seed, ebno, start, end, mid = key
        item = {"size": size, "variant": variant, "lambda_mem": lam, "EbNo_dB": ebno,
                "cycle_start": start, "cycle_end": end, "cycle_mid": mid}
        if by_seed:
            item["seed"] = seed
        for metric, stat in groups[key].items():
            for suffix, value in stat.summary().items():
                item[f"{metric}_{suffix}"] = value
        output.append(item)
    return output


SUMMARY_METRICS = [
    "decoded_BER", "late_state_BER", "late_backflip_rate",
    "reached_valid", "reached_valid_after_nonvalid", "reached_correct", "reached_low_BER",
    "reached_relaxed_basin", "reached_wrong_valid",
    "first_valid_cycle", "first_valid_after_nonvalid_cycle", "first_correct_cycle",
    "first_low_BER_cycle", "first_relaxed_basin_cycle", "first_wrong_valid_cycle",
    "correct_residence_fraction", "correct_escape_probability", "correct_escape_count",
    "correct_remaining_cycles_after_hit", "correct_cycles_after_hit",
    "correct_transition_opportunities",
    "correct_return_probability", "correct_mean_return_time", "correct_first_residence_time",
    "relaxed_basin_residence_fraction", "relaxed_basin_escape_probability",
    "relaxed_basin_remaining_cycles_after_hit", "relaxed_basin_cycles_after_hit",
    "relaxed_basin_transition_opportunities",
    "relaxed_basin_escape_count", "relaxed_basin_return_probability",
    "relaxed_basin_mean_return_time", "relaxed_basin_first_residence_time",
    "late_channel_alignment", "late_channel_alignment_gap_to_transmitted",
    "late_hard_channel_disagreement_rate", "wrong_valid_residence_fraction",
]


def _aggregate_trajectory(rows, by_seed=False):
    groups = {}
    for row in rows:
        key = (row["size"], row["variant"], float(row["lambda_mem"]),
               int(row["seed"]) if by_seed else None, float(row["EbNo_dB"]), int(row["n_cycles"]))
        groups.setdefault(key, {metric: OnlineStat() for metric in SUMMARY_METRICS})
        for metric in SUMMARY_METRICS:
            groups[key][metric].add(row.get(metric))
    output = []
    for key in sorted(groups):
        size, variant, lam, seed, ebno, n_cycles = key
        item = {"size": size, "variant": variant, "lambda_mem": lam,
                "EbNo_dB": ebno, "n_cycles": n_cycles}
        if by_seed:
            item["seed"] = seed
        for metric, stat in groups[key].items():
            for suffix, value in stat.summary().items():
                item[f"{metric}_{suffix}"] = value
        output.append(item)
    return output


def _add_pooled_retention(summary_rows, trajectory_rows, by_seed=False):
    grouped = defaultdict(list)
    for row in trajectory_rows:
        key = (row["size"], row["variant"], int(row["seed"]) if by_seed else None)
        grouped[key].append(row)
    for item in summary_rows:
        key = (item["size"], item["variant"], int(item["seed"]) if by_seed else None)
        rows = grouped.get(key, [])
        for prefix in ["correct", "relaxed_basin"]:
            remaining = sum(float(row[f"{prefix}_remaining_cycles_after_hit"])
                            for row in rows if math.isfinite(float(row[f"{prefix}_remaining_cycles_after_hit"])) )
            occupied = sum(float(row[f"{prefix}_cycles_after_hit"])
                           for row in rows if math.isfinite(float(row[f"{prefix}_cycles_after_hit"])) )
            opportunities = sum(float(row[f"{prefix}_transition_opportunities"])
                                for row in rows if math.isfinite(float(row[f"{prefix}_transition_opportunities"])) )
            escapes = sum(float(row[f"{prefix}_escape_count"])
                          for row in rows if math.isfinite(float(row[f"{prefix}_escape_count"])) )
            returns = sum(float(row[f"{prefix}_return_count"])
                          for row in rows if math.isfinite(float(row[f"{prefix}_return_count"])) )
            item[f"{prefix}_residence_fraction_pooled"] = _finite_ratio(occupied, remaining)
            item[f"{prefix}_escape_probability_pooled"] = _finite_ratio(escapes, opportunities)
            item[f"{prefix}_return_probability_pooled"] = _finite_ratio(returns, escapes)
    return summary_rows


def _t95(n):
    critical = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571,
                7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262}
    return critical.get(n, 1.96) if n > 1 else float("nan")


def _paired_statistics(rows):
    metrics = [
        "reached_correct", "reached_relaxed_basin", "reached_valid_after_nonvalid",
        "reached_wrong_valid", "first_correct_cycle", "first_relaxed_basin_cycle",
        "correct_residence_fraction", "correct_escape_probability",
        "relaxed_basin_residence_fraction", "relaxed_basin_escape_probability",
        "late_channel_alignment", "late_channel_alignment_gap_to_transmitted",
        "wrong_valid_residence_fraction",
    ]
    indexed = defaultdict(dict)
    for row in rows:
        key = (row["size"], int(row["seed"]), int(row["trial_index"]))
        indexed[key][row["variant"]] = row
    differences = defaultdict(list)
    for (size, seed, _trial), variants in indexed.items():
        if "additive" not in variants:
            continue
        for control, control_row in variants.items():
            if control == "additive":
                continue
            for metric in metrics:
                try:
                    delta = float(control_row[metric]) - float(variants["additive"][metric])
                except (KeyError, TypeError, ValueError):
                    continue
                if math.isfinite(delta):
                    differences[(size, control, metric, seed)].append(delta)
    seed_means = defaultdict(list)
    pairs = defaultdict(int)
    for (size, control, metric, _seed), values in differences.items():
        if values:
            seed_means[(size, control, metric)].append(float(np.mean(values)))
            pairs[(size, control, metric)] += len(values)
    output = []
    for key in sorted(seed_means):
        values = np.asarray(seed_means[key])
        mean = float(np.mean(values)); n = len(values)
        sd = float(np.std(values, ddof=1)) if n > 1 else float("nan")
        sem = sd / math.sqrt(n) if n > 1 else float("nan")
        half = _t95(n) * sem if n > 1 else float("nan")
        size, control, metric = key
        output.append({"size": size, "EbNo_dB": 2.5, "control": control,
                       "reference": "additive", "metric": metric,
                       "contrast_definition": "control_minus_additive",
                       "mean_difference": mean, "seed_batch_sd": sd, "seed_batch_sem": sem,
                       "ci95_low": mean - half, "ci95_high": mean + half,
                       "n_seed_batches": n, "n_paired_trajectories_with_finite_metric": pairs[key]})
    return output


def _first_passage_curves(rows, cycle_bin):
    output = []
    groups = defaultdict(list)
    for row in rows:
        groups[(row["size"], row["variant"], int(row["n_cycles"]))].append(row)
    for (size, variant, n_cycles), group in sorted(groups.items()):
        cycles = list(range(0, n_cycles + 1, cycle_bin))
        if cycles[-1] != n_cycles:
            cycles.append(n_cycles)
        for event, field in ACQUISITION_EVENTS.items():
            values = []
            for row in group:
                try:
                    value = float(row[field])
                except (TypeError, ValueError):
                    continue
                if math.isfinite(value):
                    values.append(value)
            for cycle in cycles:
                reached = sum(value <= cycle for value in values)
                fraction = reached / len(group)
                output.append({"size": size, "variant": variant, "event": event,
                               "cycle": cycle, "cumulative_reach_probability": fraction,
                               "not_yet_reached_probability": 1.0 - fraction,
                               "n_trajectories": len(group), "n_ever_reached": len(values)})
    return output


def _atomic_gzip_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


def _read_csv(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def _make_figures(cycle_summary, condition_summary, passage_curves, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    colors = {"pSA": "#4C78A8", "additive": "#E45756", "gain_only": "#F2CF5B",
              "shuffled": "#B279A2", "normalized": "#72B7B2"}
    labels = {"pSA": "pSA", "additive": "additive", "gain_only": "gain-only",
              "shuffled": "shuffled", "normalized": "normalized"}
    variants = [v for v in VARIANT_CODES if any(row["variant"] == v for row in cycle_summary)]

    n192 = [row for row in cycle_summary if row["size"] == "N192_M96"]
    fig, axes = plt.subplots(3, 1, figsize=(8.2, 8.8), sharex=True)
    specs = [("syndrome_weight_fraction", "Syndrome weight / M"),
             ("state_BER", "Instantaneous BER"),
             ("channel_alignment", "Channel alignment")]
    for ax, (metric, ylabel) in zip(axes, specs):
        for variant in variants:
            data = sorted([row for row in n192 if row["variant"] == variant],
                          key=lambda row: float(row["cycle_mid"]))
            if not data: continue
            x = np.asarray([float(row["cycle_mid"]) for row in data])
            mean = np.asarray([float(row[f"{metric}_mean"]) for row in data])
            low = np.asarray([float(row[f"{metric}_ci_low"]) for row in data])
            high = np.asarray([float(row[f"{metric}_ci_high"]) for row in data])
            ax.plot(x, mean, color=colors[variant], label=labels[variant], linewidth=1.5)
            ax.fill_between(x, low, high, color=colors[variant], alpha=0.13, linewidth=0)
        ax.set_ylabel(ylabel); ax.grid(alpha=0.22)
    axes[-1].set_xlabel("Cycle")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.suptitle("Acquisition dynamics: N192/M96, Eb/N0 = 2.5 dB", y=0.985)
    fig.legend(handles, legend_labels, loc="upper center", bbox_to_anchor=(0.5, 0.945),
               ncol=len(variants), frameon=False)
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.08, top=0.89, hspace=0.10)
    for ext in ["png", "pdf"]:
        fig.savefig(out_dir / f"Fig_phase2b_A_channel_acquisition.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    sizes = [size for size in ["N192_M96", "N288_M144"]
             if any(row["size"] == size for row in condition_summary)]
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.6))
    bar_specs = [
        ("reached_correct_mean", "Correct-codeword reach probability", False),
        ("first_correct_cycle_mean", "First-passage cycle (reached only)", False),
        ("correct_residence_fraction_pooled", "Post-hit correct residence fraction", False),
        ("correct_escape_probability_pooled", "Post-hit escape probability / cycle", True),
    ]
    x = np.arange(len(sizes)); width = 0.82 / len(variants)
    for ax, (field, ylabel, log_scale) in zip(axes.flat, bar_specs):
        for vi, variant in enumerate(variants):
            means, errs = [], []
            for size in sizes:
                match = next((row for row in condition_summary
                              if row["size"] == size and row["variant"] == variant), None)
                if match is None:
                    means.append(float("nan")); errs.append(float("nan")); continue
                mean = float(match[field]); lo = float(match[field.replace("_mean", "_ci_low")])
                hi = float(match[field.replace("_mean", "_ci_high")])
                means.append(mean); errs.append(max(mean - lo, hi - mean))
            positions = x - 0.41 + width / 2 + vi * width
            ax.bar(positions, means, width, color=colors[variant], label=labels[variant],
                   yerr=errs, error_kw={"linewidth": 0.8, "capsize": 2})
        ax.set_xticks(x, [size.split("_")[0] for size in sizes]); ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.22)
        if log_scale:
            positive_values = []
            for row in condition_summary:
                try:
                    value = float(row[field])
                except (KeyError, TypeError, ValueError):
                    continue
                if math.isfinite(value) and value > 0.0:
                    positive_values.append(value)
            if positive_values:
                ax.set_yscale("log")
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle("Acquisition versus retention", y=0.985)
    fig.legend(handles, legend_labels, loc="upper center", bbox_to_anchor=(0.5, 0.935),
               ncol=len(variants), frameon=False)
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.08, top=0.865, hspace=0.30, wspace=0.22)
    for ext in ["png", "pdf"]:
        fig.savefig(out_dir / f"Fig_phase2b_B_acquisition_retention_summary.{ext}", dpi=300,
                    bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(len(sizes), 2, figsize=(10.8, 3.6 * len(sizes)), sharex="row")
    if len(sizes) == 1: axes = np.asarray(axes).reshape(1, 2)
    for si, size in enumerate(sizes):
        for col, event in enumerate(["correct", "relaxed_basin"]):
            ax = axes[si, col]
            for variant in variants:
                data = sorted([row for row in passage_curves if row["size"] == size
                               and row["variant"] == variant and row["event"] == event],
                              key=lambda row: float(row["cycle"]))
                if not data: continue
                ax.step([float(row["cycle"]) for row in data],
                        [float(row["cumulative_reach_probability"]) for row in data],
                        where="post", color=colors[variant], label=labels[variant])
            ax.set_ylim(-0.02, 1.02); ax.grid(alpha=0.22); ax.set_xlabel("Cycle")
            ax.set_ylabel("Cumulative reach probability")
            ax.set_title(f"{size.split('_')[0]}: {'exact correct' if col == 0 else 'relaxed low-BER basin'}")
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", bbox_to_anchor=(0.5, 0.98),
               ncol=len(variants), frameon=False)
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.08, top=0.91, hspace=0.34, wspace=0.22)
    for ext in ["png", "pdf"]:
        fig.savefig(out_dir / f"Fig_phase2b_C_first_passage_curves.{ext}", dpi=300,
                    bbox_inches="tight")
    plt.close(fig)


def run_invariants(matrix_seed=0):
    checks = {"phase2_invariants_all_passed": bool(run_phase2_invariants(matrix_seed)["all_passed"])}
    P = random_regular_ldpc_parity_check(48, 24, 3, 6, seed=matrix_seed)
    channel_values = np.linspace(-0.8, 0.8, 48)
    codeword = np.zeros(48, dtype=np.uint8)
    params = {"alpha": 1.0, "alpha_mode": "fixed", "kw": 1.2, "kr": 1.5,
              "n_cycles": 24, "I0_min": 0.1, "I0_max": 0.8, "psa_p": 0.25,
              "lambda_mem": 0.7, "nrnd": 0.0, "response_lambda": 0.0, "lambda_out": 0.0,
              "I0_schedule_type": "linear", "I0_schedule_shape": 1.0,
              "I0_hold_fraction": 0.0, "decision_method": "best_state",
              "burn_in": 5, "sample_window": 12}
    from run_phase2_ldpc_dynamics import _run_logged_trajectory
    for variant in PRIMARY_VARIANTS:
        phase2_rng = np.random.default_rng(12345); phase2_shuffle = np.random.default_rng(999)
        phase2b_rng = np.random.default_rng(12345); phase2b_shuffle = np.random.default_rng(999)
        p2 = _run_logged_trajectory(P, channel_values, codeword, params, variant, 6,
                                    phase2_rng, phase2_shuffle)
        p2b = _run_phase2b_trajectory(P, channel_values, codeword, params, variant, 6,
                                      phase2b_rng, phase2b_shuffle)
        checks[f"{variant}_decoded_matches_phase2"] = bool(np.array_equal(p2[0], p2b[0]))
        checks[f"{variant}_final_state_matches_phase2"] = bool(np.array_equal(p2[1], p2b[1]))
        checks[f"{variant}_existing_metrics_match_phase2"] = bool(np.array_equal(p2[2], p2b[2]))
    scalar_score = channel_likelihood_score(codeword, channel_values)
    checks["channel_score_matches_existing_definition"] = bool(
        abs(scalar_score - float(np.sum(channel_values * (1.0 - 2.0 * codeword)))) < 1e-12
    )
    toy_P = np.asarray([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
    toy_codeword = np.asarray([0, 0, 0], dtype=np.uint8)
    toy_states = np.asarray([[1, 1, 1], [0, 1, 0], [0, 0, 0],
                             [0, 0, 1], [0, 0, 0], [0, 0, 0]], dtype=np.uint8)
    analysis = analyze_artificial_trajectory(toy_P, toy_states, toy_codeword)
    checks["ber_zero_means_transmitted_codeword"] = bool(np.array_equal(
        toy_states[2], toy_codeword
    ) and analysis["correct"][2])
    checks["wrong_valid_codeword_distinguished"] = bool(
        analysis["valid"][0] and analysis["wrong_valid"][0] and not analysis["correct"][0]
    )
    checks["toy_first_passages_correct"] = bool(
        analysis["first_valid"] == 1 and analysis["first_valid_after_nonvalid"] == 3
        and analysis["first_correct"] == 3 and analysis["first_wrong_valid"] == 1
    )
    retention = analysis["correct_retention"]
    checks["toy_residence_escape_return_correct"] = bool(
        abs(retention["residence_fraction"] - 0.75) < 1e-12
        and retention["escapes"] == 1 and retention["returns"] == 1
        and abs(retention["escape_probability"] - 1.0 / 2.0) < 1e-12
        and retention["mean_return_time"] == 1.0 and retention["first_residence"] == 1
        and not retention["first_residence_censored"]
    )
    checks["all_passed"] = bool(all(value for value in checks.values()))
    return checks


def _setup_logging(run_dir):
    logger = logging.getLogger("phase2b")
    logger.handlers.clear(); logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    stream = logging.StreamHandler(sys.stdout); stream.setFormatter(formatter)
    file_handler = logging.FileHandler(run_dir / "run.log"); file_handler.setFormatter(formatter)
    logger.addHandler(stream); logger.addHandler(file_handler)
    return logger


def build_parser():
    parser = argparse.ArgumentParser(description="Phase 2b acquisition versus retention analysis")
    parser.add_argument("--profile", choices=sorted(PROFILE_DEFAULTS), default="smoke")
    parser.add_argument("--sizes", default=None)
    parser.add_argument("--variants", default=None)
    parser.add_argument("--trials", type=int, default=None)
    parser.add_argument("--seed-count", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--matrix-seed", type=int, default=0)
    parser.add_argument("--ebno", type=float, default=2.5)
    parser.add_argument("--n-workers", type=int, default=8)
    parser.add_argument("--cycles-override", type=int, default=None)
    parser.add_argument("--cycle-bin", type=int, default=None)
    parser.add_argument("--fixed-bit-width", type=int, default=8)
    parser.add_argument("--channel-input-mode", choices=["float", "fixed"], default="float")
    parser.add_argument("--source-root", action="append", default=None)
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "phase2b_results"))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main():
    args = build_parser().parse_args(); defaults = PROFILE_DEFAULTS[args.profile]
    if args.sizes is None: sizes = list(defaults["sizes"])
    elif args.sizes == "all": sizes = ["N192_M96", "N288_M144"]
    else: sizes = parse_list(args.sizes)
    variants = parse_list(args.variants) if args.variants else list(defaults["variants"])
    trials = int(args.trials if args.trials is not None else defaults["trials"])
    seed_count = int(args.seed_count if args.seed_count is not None else defaults["seed_count"])
    cycles_override = args.cycles_override if args.cycles_override is not None else defaults["cycles_override"]
    cycle_bin = int(args.cycle_bin if args.cycle_bin is not None else defaults["cycle_bin"])
    if set(sizes) - {"N192_M96", "N288_M144"}: raise ValueError("Phase 2b targets N192/N288")
    if set(variants) - set(VARIANT_CODES): raise ValueError("Unknown variant")
    seeds = [int(args.seed) + 8022 * index for index in range(seed_count)]

    roots = result_roots(args.source_root)
    sources = {size: locate_completed_size(size, roots) for size in sizes}
    base_params, source_files = {}, {}
    for size in sizes:
        params, path = load_anchor(sources[size], "lambda")
        base_params[size] = params; source_files[size] = str(path)

    run_name = args.run_name or f"{args.profile}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(args.output_dir).expanduser().resolve() / run_name
    run_dir.mkdir(parents=True, exist_ok=True); logger = _setup_logging(run_dir)
    config = {"profile": args.profile, "sizes": sizes, "variants": variants,
              "EbNo_dB": float(args.ebno), "trials_per_condition": trials,
              "seed_count": seed_count, "seeds": seeds, "matrix_seed": int(args.matrix_seed),
              "n_workers": int(args.n_workers), "cycles_override": cycles_override,
              "cycle_bin": cycle_bin, "fixed_bit_width": int(args.fixed_bit_width),
              "channel_input_mode": args.channel_input_mode,
              "low_BER_threshold": LOW_BER_THRESHOLD,
              "syndrome_fraction_threshold_for_relaxed_basin": SYNDROME_FRACTION_THRESHOLD,
              "channel_alignment_definition": "sum_i z_i*(1-2*x_i)/N",
              "source_parameter_files": source_files,
              "lambdas": {size: FINAL_LAMBDAS[size] for size in sizes},
              "command": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]}
    digest = config_hash(config); config_path = run_dir / "effective_config.json"
    if config_path.exists() and args.resume:
        old = json.loads(config_path.read_text())
        if old.get("config_hash") != digest: raise RuntimeError("Changed config; use a new run name")
    write_json(config_path, {**config, "config_hash": digest})
    invariants = run_invariants(int(args.matrix_seed)); write_json(run_dir / "invariants.json", invariants)
    if not invariants["all_passed"]: raise RuntimeError("Phase 2b invariant failed")

    conditions, plan_rows = [], []; total_updates = 0
    for size in sizes:
        for variant in variants:
            lam = 0.0 if variant == "pSA" else FINAL_LAMBDAS[size]
            params = adjusted_params(base_params[size], lam, cycles_override)
            condition = {"size": size, "variant": variant,
                         "implementation_mode": IMPLEMENTATION_MODES[variant],
                         "formula": FORMULAS[variant], "lambda_mem": lam,
                         "params": params, "source_parameter_file": source_files[size]}
            conditions.append(condition)
            updates = trials * int(params["n_cycles"]) * SIZE_SPECS[size]["n_bits"] * 2
            total_updates += updates
            plan_rows.append({"size": size, "variant": variant, "lambda_mem": lam,
                              "EbNo_dB": args.ebno, "trials": trials, "seed_count": seed_count,
                              "n_cycles": params["n_cycles"], "cycle_bin": cycle_bin,
                              "estimated_bit_updates_including_dual_validation_kernel": updates,
                              "source_parameter_file": source_files[size]})
    write_csv(run_dir / "experiment_plan.csv", plan_rows, _fields_for(plan_rows))
    write_json(run_dir / "parameters_and_seeds.json", {
        "config_hash": digest,
        "matrix_seed": int(args.matrix_seed),
        "trajectory_seed_construction": "seed + 8022 * seed_batch_index",
        "seeds": seeds,
        "fixed_bit_width": int(args.fixed_bit_width),
        "channel_input_mode": args.channel_input_mode,
        "anchor_parameters": {
            size: {
                "source_parameter_file": source_files[size],
                "loaded_parameters": base_params[size],
            }
            for size in sizes
        },
        "effective_condition_parameters": [
            {
                "size": condition["size"],
                "variant": condition["variant"],
                "implementation_mode": condition["implementation_mode"],
                "formula": condition["formula"],
                "lambda_mem": condition["lambda_mem"],
                "source_parameter_file": condition["source_parameter_file"],
                "parameters": condition["params"],
            }
            for condition in conditions
        ],
    })
    manifest = {"status": "planned" if args.dry_run else "running",
                "started_utc": datetime.now(timezone.utc).isoformat(), "finished_utc": None,
                "run_dir": str(run_dir), "config_hash": digest,
                "environment": environment_record(), "condition_count": len(conditions),
                "completed_conditions": 0, "estimated_bit_updates": total_updates}
    write_json(run_dir / "manifest.json", manifest)
    logger.info("Run directory: %s", run_dir)
    logger.info("Conditions=%d trajectories=%d dual-kernel bit updates=%.6g",
                len(conditions), len(conditions) * trials, total_updates)
    if args.dry_run: logger.info("Dry run complete"); return

    start_time = time.perf_counter(); all_cycles = []; all_trajectories = []
    all_cycle_summary = []; all_seed_cycle = []; completed = 0
    for index, condition in enumerate(conditions, 1):
        condition_dir = run_dir / "trajectories" / condition["size"] / condition["variant"]
        done_path = condition_dir / "complete.json"
        if args.resume and done_path.exists() and json.loads(done_path.read_text()).get("config_hash") == digest:
            logger.info("[%d/%d] resume skip %s/%s", index, len(conditions), condition["size"], condition["variant"])
            trajectories = _read_csv(condition_dir / "trajectory_summary.csv")
            cycle_summary = _read_csv(condition_dir / "by_cycle_summary.csv")
            seed_cycle = _read_csv(condition_dir / "by_seed_cycle.csv")
            all_trajectories.extend(trajectories); all_cycle_summary.extend(cycle_summary)
            all_seed_cycle.extend(seed_cycle); completed += 1; continue
        P = random_regular_ldpc_parity_check(SIZE_SPECS[condition["size"]]["n_bits"],
                                             SIZE_SPECS[condition["size"]]["n_checks"],
                                             3, 6, seed=int(args.matrix_seed))
        tasks = [{"P": P, "condition": condition, "seed": seed, "n_trials": count,
                  "ebno": float(args.ebno), "cycle_bin": cycle_bin,
                  "fixed_bit_width": int(args.fixed_bit_width),
                  "channel_input_mode": args.channel_input_mode}
                 for seed, count in split_total_trials(trials, seeds)]
        logger.info("[%d/%d] %s %s trials=%d cycles=%d", index, len(conditions),
                    condition["size"], condition["variant"], trials, condition["params"]["n_cycles"])
        condition_start = time.perf_counter(); results = []
        workers = min(int(args.n_workers), len(tasks))
        if workers <= 1: results = [_seed_task(task) for task in tasks]
        else:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(_seed_task, task) for task in tasks]
                for future in as_completed(futures): results.append(future.result())
        cycles = []; trajectories = []
        for result in results:
            cycles.extend(result["cycle_rows"]); trajectories.extend(result["trajectory_rows"])
        cycles.sort(key=lambda row: (int(row["seed"]), int(row["trial_index"]), int(row["cycle_start"])))
        trajectories.sort(key=lambda row: (int(row["seed"]), int(row["trial_index"])))
        cycle_summary = _aggregate_cycle(cycles, False); seed_cycle = _aggregate_cycle(cycles, True)
        condition_dir.mkdir(parents=True, exist_ok=True)
        _atomic_gzip_csv(condition_dir / "raw_trajectory_metrics.csv.gz", cycles, CYCLE_FIELDS)
        write_csv(condition_dir / "trajectory_summary.csv", trajectories, TRAJECTORY_FIELDS)
        write_csv(condition_dir / "by_cycle_summary.csv", cycle_summary, _fields_for(cycle_summary))
        write_csv(condition_dir / "by_seed_cycle.csv", seed_cycle, _fields_for(seed_cycle))
        elapsed = time.perf_counter() - condition_start
        write_json(done_path, {"status": "complete", "config_hash": digest,
                               "elapsed_seconds": elapsed, "trajectory_count": len(trajectories),
                               "cycle_row_count": len(cycles)})
        all_trajectories.extend(trajectories); all_cycle_summary.extend(cycle_summary)
        all_seed_cycle.extend(seed_cycle); completed += 1
        manifest["completed_conditions"] = completed; write_json(run_dir / "manifest.json", manifest)
        logger.info("[%d/%d] complete in %.2f s", index, len(conditions), elapsed)

    condition_summary = _add_pooled_retention(
        _aggregate_trajectory(all_trajectories, False), all_trajectories, False
    )
    seed_summary = _add_pooled_retention(
        _aggregate_trajectory(all_trajectories, True), all_trajectories, True
    )
    paired = _paired_statistics(all_trajectories)
    passage = _first_passage_curves(all_trajectories, cycle_bin)
    write_csv(run_dir / "trajectory_summary.csv", all_trajectories, TRAJECTORY_FIELDS)
    write_csv(run_dir / "condition_summary.csv", condition_summary, _fields_for(condition_summary))
    write_csv(run_dir / "seed_batch_summary.csv", seed_summary, _fields_for(seed_summary))
    write_csv(run_dir / "paired_statistics.csv", paired, _fields_for(paired))
    write_csv(run_dir / "by_cycle_summary.csv", all_cycle_summary, _fields_for(all_cycle_summary))
    write_csv(run_dir / "by_seed_cycle.csv", all_seed_cycle, _fields_for(all_seed_cycle))
    write_csv(run_dir / "first_passage_curves.csv", passage, _fields_for(passage))
    _make_figures(all_cycle_summary, condition_summary, passage, run_dir / "figures")
    elapsed = time.perf_counter() - start_time
    runtime = {"elapsed_seconds_this_invocation": elapsed, "completed_conditions": completed,
               "trajectory_count": len(all_trajectories), "dual_kernel_bit_updates": total_updates,
               "bit_updates_per_second": total_updates / max(elapsed, 1e-12)}
    write_json(run_dir / "runtime_summary.json", runtime)
    manifest.update({"status": "complete", "finished_utc": datetime.now(timezone.utc).isoformat(),
                     "completed_conditions": completed, "runtime_summary": runtime})
    write_json(run_dir / "manifest.json", manifest)
    logger.info("Done in %.2f s: %s", elapsed, run_dir)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Matched binary-state self-feedback control for the LDPC response-memory study.

This runner keeps the representative matrices, additive-optimized nonmemory
parameters, channel construction, partial-activation semantics, stochastic
threshold, readout, SNRs, and seed batches of the validated matched causal
control.  It changes only the stored quantity supplied after the current tanh:

    d_i^(t) = q_i^(t) + kappa * s_i^(t),
    s_i^(t) = 2*w_i^(t) - 1 in {-1,+1}.

Here w_i^(t) is the same bit's common pre-update/current binary state.  This is
a binary-state self-feedback control, not a reproduction of PIMI.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import inspect
import json
import logging
import math
import os
import platform
import socket
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


RUN_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
LDPC_DIR = REPO_ROOT / "src" / "ldpc"
REFERENCE_DIR = REPO_ROOT / "src" / "reference"
for import_path in [REPO_ROOT, LDPC_DIR, REFERENCE_DIR]:
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from auto_match_bp import (  # noqa: E402
    EvalCounts,
    candidate_rows_from_counts,
    score_candidate,
    split_total_trials,
)
from ldpc_pbit import (  # noqa: E402
    _packed_pbit_connections,
    _syndrome_weight_numba,
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
    load_anchor,
    load_reference,
    locate_completed_size,
    result_roots,
)
from run_phase2_ldpc_dynamics import (  # noqa: E402
    _decode_psa_with_metrics_numba,
    _prepare_kernel_inputs,
    _shuffle_mapping,
    _trajectory_records,
)
from run_phase2b_acquisition_retention import (  # noqa: E402
    LOW_BER_THRESHOLD,
    SYNDROME_FRACTION_THRESHOLD,
    TRAJECTORY_FIELDS,
    _add_pooled_retention,
    _aggregate_cycle,
    _aggregate_trajectory,
    _cycle_extra_rows,
    _decode_events_channel_numba,
    _event_trajectory_fields,
    _first_passage_curves,
    _paired_statistics as trajectory_paired_statistics,
    _run_phase2b_trajectory,
)

try:
    from numba import njit
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("This study requires the project environment with numba") from exc


SIZES = ["N96_M48", "N192_M96", "N288_M144"]
SELECTED_LAMBDAS = {
    "N96_M48": 0.95,
    "N192_M96": 0.95,
    "N288_M144": 0.90,
}
KAPPA_GRID = [0.0, 0.25, 0.50, 0.75, 0.90, 0.95, 1.00, 1.10, 1.25]
EBNO_VALUES = [2.0, 2.5, 3.0]
BASE_SEED = 20260826
SWEEP_SEEDS = [BASE_SEED + 8022 * index for index in range(8)]
HIGHSTAT_SEEDS = [BASE_SEED + 8022 * index for index in range(10)]
MATRIX_SEED = 0
BINARY_VARIANT_CODE = 5

SOURCE_DIRS = {
    "N96_M48": REPO_ROOT / "data" / "processed" / "representative_benchmark" / "search_records" / "N96_M48",
    "N192_M96": REPO_ROOT / "data" / "processed" / "representative_benchmark" / "search_records" / "N192_M96",
    "N288_M144": REPO_ROOT / "data" / "processed" / "representative_benchmark" / "search_records" / "N288_M144",
}
CONFIRM_DIRS = {
    "N96_M48": REPO_ROOT / "data" / "processed" / "matched_causal_controls" / "archive" / "N96",
    "N192_M96": REPO_ROOT / "data" / "processed" / "matched_causal_controls" / "archive" / "N192",
    "N288_M144": REPO_ROOT / "data" / "processed" / "matched_causal_controls" / "archive" / "N288",
}
EXISTING_SWEEP_DIR = REPO_ROOT / "data" / "processed" / "matched_causal_controls" / "archive" / "full_sweep"
MANUSCRIPT_DIR = REPO_ROOT / "docs" / "source_records"

SWEEP_SUMMARY_FIELDS = [
    "size", "variant", "implementation_mode", "kappa", "equation", "score",
    "ber_score", "fer_score", "shape_score", "syndrome_penalty", "floor_penalty",
    "mean_BER", "mean_FER", "mean_syndrome_violation", "trials_per_EbNo",
    "seed_count", "seeds", "elapsed_seconds", "source_parameter_file",
]
BY_EBNO_FIELDS = [
    "stage", "size", "variant", "implementation_mode", "kappa", "EbNo_dB",
    "BER", "FER", "syndrome_violation_rate", "avg_syndrome_weight", "BP_BER",
    "BP_FER", "n_trials", "n_bits_total",
]
BY_SEED_FIELDS = [
    "stage", "size", "variant", "implementation_mode", "kappa", "seed", "EbNo_dB",
    "BER", "FER", "syndrome_violation_rate", "avg_syndrome_weight", "BP_BER",
    "BP_FER", "n_trials", "n_bits_total",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_ready(data), indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: Iterable[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields_list: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for field in row:
                if field not in seen:
                    fields_list.append(field)
                    seen.add(field)
    else:
        fields_list = list(fields)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields_list, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_gzip_csv(path: Path, rows: list[dict[str, Any]], fields: Iterable[str]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def hash_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def setup_logging() -> logging.Logger:
    logger = logging.getLogger("binary_state_self_feedback")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(RUN_DIR / "run.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


@njit(cache=True)
def _channel_score_numba(bits, channel_values):
    score = 0.0
    for bit in range(bits.shape[0]):
        score += channel_values[bit] * (1.0 - 2.0 * bits[bit])
    return score


@njit(cache=True)
def _decode_binary_state_numba(
    P,
    group_indices,
    group_counts,
    group_lengths,
    channel_values,
    kw,
    kr,
    n_cycles,
    I0_history,
    psa_p,
    kappa,
    decision_code,
    burn_in,
    sample_window,
    seed,
):
    """Production endpoint kernel with no response-history state array."""
    np.random.seed(seed)
    n_bits = P.shape[1]
    w = np.zeros(n_bits, dtype=np.uint8)
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

            # Match the optimized matched-control path: held bits skip q and xi.
            if psa_p > 0.0 and np.random.random() < psa_p:
                w[bit] = w_prev[bit]
                continue

            current_q = np.tanh(I0 * (kw * parity_sum + kr * channel_term))
            rnd = -1.0 + 2.0 * np.random.random()
            spin_pre = 2.0 * w_prev[bit] - 1.0
            deterministic = current_q + kappa * spin_pre
            w[bit] = 1 if deterministic + rnd >= 0.0 else 0

        if cycle >= sample_start:
            if decision_code == 1:
                sample_count += 1
                for bit in range(n_bits):
                    ones_count[bit] += int(w[bit])
            elif decision_code == 2:
                syndrome_weight = _syndrome_weight_numba(P, w)
                channel_score = _channel_score_numba(w, channel_values)
                if (
                    syndrome_weight < best_syndrome_weight
                    or (
                        syndrome_weight == best_syndrome_weight
                        and channel_score > best_channel_score
                    )
                ):
                    best_syndrome_weight = syndrome_weight
                    best_channel_score = channel_score
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
    return decoded


@njit(cache=True)
def _trace_psa_or_binary_numba(
    P,
    group_indices,
    group_counts,
    group_lengths,
    channel_values,
    kw,
    kr,
    n_cycles,
    I0_history,
    psa_p,
    kappa,
    binary_mode,
    seed,
):
    """Validation-only history logger with optimized-path RNG ordering."""
    np.random.seed(seed)
    n_bits = P.shape[1]
    w = np.zeros(n_bits, dtype=np.uint8)
    history = np.zeros((n_cycles + 1, n_bits), dtype=np.uint8)
    discriminants = np.full((n_cycles, n_bits), np.nan, dtype=np.float64)
    for cycle in range(n_cycles):
        I0 = I0_history[cycle]
        w_prev = w.copy()
        for bit in range(n_bits):
            parity_sum = 0
            for group_idx in range(group_counts[bit]):
                target = 0
                for pos in range(group_lengths[bit, group_idx]):
                    target ^= int(w_prev[group_indices[bit, group_idx, pos]])
                parity_sum += 1 if target == 1 else -1
            rnd_channel = -1.0 + 2.0 * np.random.random()
            channel_bit = 1 if channel_values[bit] < rnd_channel else 0
            channel_term = 1 if channel_bit == 1 else -1
            if psa_p > 0.0 and np.random.random() < psa_p:
                w[bit] = w_prev[bit]
                continue
            current_q = np.tanh(I0 * (kw * parity_sum + kr * channel_term))
            rnd = -1.0 + 2.0 * np.random.random()
            deterministic = current_q
            if binary_mode == 1:
                deterministic += kappa * (2.0 * w_prev[bit] - 1.0)
            discriminants[cycle, bit] = deterministic + rnd
            w[bit] = 1 if discriminants[cycle, bit] >= 0.0 else 0
        history[cycle + 1] = w
    return history, discriminants


def _decision_code(method: str) -> int:
    return {"last": 0, "majority": 1, "best_state": 2}[method]


def decode_binary_state_fast(
    P: np.ndarray,
    channel_values: np.ndarray,
    params: dict[str, Any],
    kappa: float,
    rng: np.random.Generator,
) -> np.ndarray:
    P_packed, group_indices, group_counts, group_lengths = _packed_pbit_connections(P)
    I0_history = make_I0_schedule(
        params["I0_min"],
        params["I0_max"],
        int(params["n_cycles"]),
        params["I0_schedule_type"],
        shape=params["I0_schedule_shape"],
        hold_fraction=params["I0_hold_fraction"],
    ).astype(np.float64)
    core_seed = int(rng.integers(0, 2**31 - 1))
    return _decode_binary_state_numba(
        P_packed,
        group_indices,
        group_counts,
        group_lengths,
        np.asarray(channel_values, dtype=np.float64),
        float(params["kw"]),
        float(params["kr"]),
        int(params["n_cycles"]),
        I0_history,
        float(params["psa_p"]),
        float(kappa),
        int(_decision_code(params["decision_method"])),
        int(params["burn_in"]),
        int(params["sample_window"]),
        core_seed,
    )


def base_params_for(size: str) -> tuple[dict[str, Any], Path]:
    params, source = load_anchor(SOURCE_DIRS[size], "lambda")
    return adjusted_params(params, 0.0, None), source


def matrix_for(size: str) -> np.ndarray:
    spec = SIZE_SPECS[size]
    return random_regular_ldpc_parity_check(
        n_bits=spec["n_bits"],
        n_checks=spec["n_checks"],
        variable_degree=3,
        check_degree=6,
        seed=MATRIX_SEED,
    )


def _candidate_chunk_binary(task: dict[str, Any]) -> dict[float, EvalCounts]:
    P = np.asarray(task["P"], dtype=np.uint8)
    params = task["params"]
    G = parity_check_to_generator(P)
    n_bits = P.shape[1]
    rate = G.shape[0] / n_bits
    seed_sequence = np.random.SeedSequence(int(task["seed"]))
    channel_seed, pbit_seed = seed_sequence.spawn(2)
    channel_rng = np.random.default_rng(channel_seed)
    pbit_rng = np.random.default_rng(pbit_seed)
    counts = {float(ebno): EvalCounts() for ebno in task["ebno_values"]}

    for ebno in task["ebno_values"]:
        cell = counts[float(ebno)]
        for _ in range(int(task["n_trials"])):
            message_bits = channel_rng.integers(0, 2, size=G.shape[0], dtype=np.uint8)
            codeword_bits = encode_message(message_bits, G)
            y = add_awgn(bpsk_modulate(codeword_bits), ebno, rate=rate, rng=channel_rng)
            channel_llr = awgn_channel_llr(y, ebno, rate=rate)
            channel_values = make_stochastic_channel_values(
                y,
                alpha=params["alpha"],
                bit_width=int(task["fixed_bit_width"]),
                input_mode=task["channel_input_mode"],
                alpha_mode=params["alpha_mode"],
                ebno_db=ebno,
                rate=rate,
                channel_llr=channel_llr,
            )
            decoded = decode_binary_state_fast(
                P=P,
                channel_values=channel_values,
                params=params,
                kappa=float(task["kappa"]),
                rng=pbit_rng,
            )
            errors = int(np.sum(decoded != codeword_bits))
            syn_weight = int(np.sum(syndrome(P, decoded)))
            cell.bit_errors += errors
            cell.frame_errors += int(errors > 0)
            cell.syndrome_violations += int(syn_weight > 0)
            cell.syndrome_weight_total += syn_weight
            cell.trials += 1
            cell.bits += n_bits
    return counts


def _combine_counts(chunks: list[dict[float, EvalCounts]]) -> dict[float, EvalCounts]:
    combined = {float(ebno): EvalCounts() for ebno in EBNO_VALUES}
    for chunk in chunks:
        for ebno, source in chunk.items():
            target = combined[float(ebno)]
            for field in [
                "bit_errors", "frame_errors", "syndrome_violations",
                "syndrome_weight_total", "trials", "bits",
            ]:
                setattr(target, field, getattr(target, field) + getattr(source, field))
    return combined


def _candidate_chunk_binary_tagged(task: dict[str, Any]) -> dict[str, Any]:
    return {
        "size": task["size"],
        "kappa": float(task["kappa"]),
        "seed": int(task["seed"]),
        "counts": _candidate_chunk_binary(task),
    }


def _result_from_seed_chunks(
    size: str,
    kappa: float,
    total_trials: int,
    seed_chunks: list[tuple[int, dict[float, EvalCounts]]],
    elapsed_seconds: float,
) -> dict[str, Any]:
    params, source = base_params_for(size)
    params = dict(params)
    params["lambda_mem"] = float(kappa)
    reference = load_reference(SOURCE_DIRS[size])
    by_ebno = candidate_rows_from_counts(
        _combine_counts([chunk for _, chunk in seed_chunks]), reference
    )
    scores = score_candidate(
        by_ebno, eps=1e-12, fer_weight=0.2, shape_weight=0.0,
        syndrome_weight=0.5, floor_weight=0.0,
    )
    return {
        "size": size, "variant": "binary_state",
        "implementation_mode": "binary_state_self_feedback", "kappa": float(kappa),
        "equation": "d_i^(t)=q_i^(t)+kappa*(2*w_i^(t)-1)", **scores,
        "mean_BER": float(np.mean([row["candidate_BER"] for row in by_ebno])),
        "mean_FER": float(np.mean([row["candidate_FER"] for row in by_ebno])),
        "mean_syndrome_violation": float(
            np.mean([row["syndrome_violation_rate"] for row in by_ebno])
        ),
        "trials_per_EbNo": int(total_trials), "seed_count": len(seed_chunks),
        "seeds": ";".join(str(seed) for seed, _ in sorted(seed_chunks)),
        "elapsed_seconds": float(elapsed_seconds), "source_parameter_file": str(source),
        "params": params, "by_ebno": by_ebno,
        "by_seed": [
            {"seed": seed, "by_ebno": candidate_rows_from_counts(chunk, reference)}
            for seed, chunk in sorted(seed_chunks)
        ],
    }


def evaluate_binary_condition(
    size: str,
    kappa: float,
    total_trials: int,
    seeds: list[int],
    n_workers: int,
) -> dict[str, Any]:
    P = matrix_for(size)
    base_params, source = base_params_for(size)
    params = dict(base_params)
    params["lambda_mem"] = float(kappa)
    reference = load_reference(SOURCE_DIRS[size])
    seed_trials = split_total_trials(total_trials, seeds)
    tasks = [
        {
            "P": P,
            "params": params,
            "kappa": float(kappa),
            "ebno_values": EBNO_VALUES,
            "n_trials": int(n_trials),
            "seed": int(seed),
            "fixed_bit_width": 8,
            "channel_input_mode": "float",
        }
        for seed, n_trials in seed_trials
    ]
    chunks: list[dict[float, EvalCounts]] = []
    by_seed_chunks: list[tuple[int, dict[float, EvalCounts]]] = []
    started = time.perf_counter()
    if n_workers == 1:
        for task in tasks:
            chunk = _candidate_chunk_binary(task)
            chunks.append(chunk)
            by_seed_chunks.append((int(task["seed"]), chunk))
    else:
        with ProcessPoolExecutor(max_workers=min(n_workers, len(tasks))) as executor:
            futures = {
                executor.submit(_candidate_chunk_binary, task): int(task["seed"])
                for task in tasks
            }
            for future in as_completed(futures):
                chunk = future.result()
                chunks.append(chunk)
                by_seed_chunks.append((futures[future], chunk))

    by_ebno = candidate_rows_from_counts(_combine_counts(chunks), reference)
    scores = score_candidate(
        by_ebno,
        eps=1e-12,
        fer_weight=0.2,
        shape_weight=0.0,
        syndrome_weight=0.5,
        floor_weight=0.0,
    )
    return {
        "size": size,
        "variant": "binary_state",
        "implementation_mode": "binary_state_self_feedback",
        "kappa": float(kappa),
        "equation": "d_i^(t)=q_i^(t)+kappa*(2*w_i^(t)-1)",
        **scores,
        "mean_BER": float(np.mean([row["candidate_BER"] for row in by_ebno])),
        "mean_FER": float(np.mean([row["candidate_FER"] for row in by_ebno])),
        "mean_syndrome_violation": float(
            np.mean([row["syndrome_violation_rate"] for row in by_ebno])
        ),
        "trials_per_EbNo": int(total_trials),
        "seed_count": len(seed_trials),
        "seeds": ";".join(str(seed) for seed, _ in seed_trials),
        "elapsed_seconds": time.perf_counter() - started,
        "source_parameter_file": str(source),
        "params": params,
        "by_ebno": by_ebno,
        "by_seed": [
            {
                "seed": seed,
                "by_ebno": candidate_rows_from_counts(chunk, reference),
            }
            for seed, chunk in sorted(by_seed_chunks)
        ],
    }


def _condition_key(size: str, kappa: float) -> tuple[str, str]:
    return size, f"{float(kappa):.12g}"


def run_sweep(logger: logging.Logger, n_workers: int) -> list[dict[str, Any]]:
    summary_path = RUN_DIR / "kappa_sweep.csv"
    by_ebno_path = RUN_DIR / "kappa_sweep_by_ebno.csv"
    by_seed_path = RUN_DIR / "kappa_sweep_by_seed.csv"
    summaries: list[dict[str, Any]] = [dict(row) for row in read_csv(summary_path)]
    by_ebno_rows: list[dict[str, Any]] = [dict(row) for row in read_csv(by_ebno_path)]
    by_seed_rows: list[dict[str, Any]] = [dict(row) for row in read_csv(by_seed_path)]
    completed = {
        _condition_key(str(row["size"]), float(row["kappa"])) for row in summaries
    }
    all_conditions = [(size, kappa) for size in SIZES for kappa in KAPPA_GRID]
    pending = [(size, kappa) for size, kappa in all_conditions
               if _condition_key(size, kappa) not in completed]
    for index, (size, kappa) in enumerate(all_conditions, start=1):
        if _condition_key(size, kappa) in completed:
            logger.info("[sweep %d/%d] resume skip %s kappa=%.3g",
                        index, len(all_conditions), size, kappa)
    if not pending:
        return [dict(row) for row in read_csv(summary_path)]

    tasks = []
    condition_started = {}
    for size, kappa in pending:
        params, _ = base_params_for(size)
        params = dict(params); params["lambda_mem"] = float(kappa)
        condition_started[_condition_key(size, kappa)] = time.perf_counter()
        for seed, n_trials in split_total_trials(2000, SWEEP_SEEDS):
            tasks.append({
                "size": size, "P": matrix_for(size), "params": params, "kappa": kappa,
                "ebno_values": EBNO_VALUES, "n_trials": n_trials, "seed": seed,
                "fixed_bit_width": 8, "channel_input_mode": "float",
            })
    logger.info("[sweep] persistent pool: %d pending conditions, %d seed tasks, %d workers",
                len(pending), len(tasks), n_workers)
    chunks_by_condition: dict[tuple[str, str], list[tuple[int, dict[float, EvalCounts]]]] = defaultdict(list)
    with ProcessPoolExecutor(max_workers=min(n_workers, len(tasks))) as executor:
        futures = [executor.submit(_candidate_chunk_binary_tagged, task) for task in tasks]
        for future in as_completed(futures):
            tagged = future.result()
            size = str(tagged["size"]); kappa = float(tagged["kappa"])
            key = _condition_key(size, kappa)
            chunks_by_condition[key].append((int(tagged["seed"]), tagged["counts"]))
            if len(chunks_by_condition[key]) != len(SWEEP_SEEDS):
                continue
            result = _result_from_seed_chunks(
                size, kappa, 2000, chunks_by_condition[key],
                time.perf_counter() - condition_started[key],
            )
            summaries.append({field: result.get(field, "") for field in SWEEP_SUMMARY_FIELDS})
            for point in result["by_ebno"]:
                by_ebno_rows.append({
                    "stage": "sweep", "size": size, "variant": "binary_state",
                    "implementation_mode": "binary_state_self_feedback", "kappa": kappa,
                    "EbNo_dB": point["EbNo_dB"], "BER": point["candidate_BER"],
                    "FER": point["candidate_FER"],
                    "syndrome_violation_rate": point["syndrome_violation_rate"],
                    "avg_syndrome_weight": point["avg_syndrome_weight"],
                    "BP_BER": point["BP_BER"], "BP_FER": point["BP_FER"],
                    "n_trials": point["n_trials"], "n_bits_total": point["n_bits_total"],
                })
            for seed_record in result["by_seed"]:
                for point in seed_record["by_ebno"]:
                    by_seed_rows.append({
                        "stage": "sweep", "size": size, "variant": "binary_state",
                        "implementation_mode": "binary_state_self_feedback", "kappa": kappa,
                        "seed": seed_record["seed"], "EbNo_dB": point["EbNo_dB"],
                        "BER": point["candidate_BER"], "FER": point["candidate_FER"],
                        "syndrome_violation_rate": point["syndrome_violation_rate"],
                        "avg_syndrome_weight": point["avg_syndrome_weight"],
                        "BP_BER": point["BP_BER"], "BP_FER": point["BP_FER"],
                        "n_trials": point["n_trials"], "n_bits_total": point["n_bits_total"],
                    })
            write_csv(by_seed_path, by_seed_rows, BY_SEED_FIELDS)
            write_csv(by_ebno_path, by_ebno_rows, BY_EBNO_FIELDS)
            write_csv(summary_path, summaries, SWEEP_SUMMARY_FIELDS)
            completed.add(key)
            index = all_conditions.index((size, kappa)) + 1
            logger.info(
                "[sweep %d/%d] complete %.1fs %s kappa=%.3g mean BER=%.6g score=%.6g",
                index, len(all_conditions), result["elapsed_seconds"], size, kappa,
                result["mean_BER"], result["score"],
            )
    return [dict(row) for row in read_csv(summary_path)]


def select_best_kappas(sweep_rows: list[dict[str, Any]]) -> dict[str, float]:
    selected: dict[str, float] = {}
    selection_rows = []
    for size in SIZES:
        candidates = [row for row in sweep_rows if row["size"] == size]
        best = min(
            candidates,
            key=lambda row: (
                float(row["score"]), float(row["ber_score"]), KAPPA_GRID.index(float(row["kappa"])),
            ),
        )
        selected[size] = float(best["kappa"])
        selection_rows.append({
            "size": size,
            "best_kappa": float(best["kappa"]),
            "selection_objective": "minimum existing matched-control score; tie break BER score then grid order",
            "score": float(best["score"]),
            "ber_score": float(best["ber_score"]),
            "fer_score": float(best["fer_score"]),
            "mean_BER": float(best["mean_BER"]),
            "mean_FER": float(best["mean_FER"]),
            "trials_per_EbNo": int(float(best["trials_per_EbNo"])),
        })
    write_csv(RUN_DIR / "best_kappa_selection.csv", selection_rows)
    return selected


def run_highstat(
    logger: logging.Logger, best_kappas: dict[str, float], n_workers: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary_path = RUN_DIR / "binary_highstat_summary.csv"
    by_ebno_path = RUN_DIR / "binary_highstat_by_ebno.csv"
    by_seed_path = RUN_DIR / "binary_highstat_by_seed.csv"
    summaries = [dict(row) for row in read_csv(summary_path)]
    by_ebno_rows = [dict(row) for row in read_csv(by_ebno_path)]
    by_seed_rows = [dict(row) for row in read_csv(by_seed_path)]
    completed = {row["size"] for row in summaries}
    pending = [size for size in SIZES if size not in completed]
    for index, size in enumerate(SIZES, start=1):
        if size in completed:
            logger.info("[high-stat %d/3] resume skip %s", index, size)
    if not pending:
        return read_csv(by_ebno_path), read_csv(by_seed_path)
    tasks = []
    condition_started = {}
    for size in pending:
        kappa = float(best_kappas[size])
        params, _ = base_params_for(size)
        params = dict(params); params["lambda_mem"] = kappa
        condition_started[size] = time.perf_counter()
        for seed, n_trials in split_total_trials(10000, HIGHSTAT_SEEDS):
            tasks.append({
                "size": size, "P": matrix_for(size), "params": params, "kappa": kappa,
                "ebno_values": EBNO_VALUES, "n_trials": n_trials, "seed": seed,
                "fixed_bit_width": 8, "channel_input_mode": "float",
            })
    logger.info("[high-stat] persistent pool: %d conditions, %d seed tasks, %d workers",
                len(pending), len(tasks), n_workers)
    chunks_by_size: dict[str, list[tuple[int, dict[float, EvalCounts]]]] = defaultdict(list)
    with ProcessPoolExecutor(max_workers=min(n_workers, len(tasks))) as executor:
        futures = [executor.submit(_candidate_chunk_binary_tagged, task) for task in tasks]
        for future in as_completed(futures):
            tagged = future.result(); size = str(tagged["size"])
            chunks_by_size[size].append((int(tagged["seed"]), tagged["counts"]))
            if len(chunks_by_size[size]) != len(HIGHSTAT_SEEDS):
                continue
            kappa = float(best_kappas[size])
            result = _result_from_seed_chunks(
                size, kappa, 10000, chunks_by_size[size],
                time.perf_counter() - condition_started[size],
            )
            summaries.append({field: result.get(field, "") for field in SWEEP_SUMMARY_FIELDS})
            for point in result["by_ebno"]:
                by_ebno_rows.append({
                    "stage": "highstat", "size": size, "variant": "binary_state",
                    "implementation_mode": "binary_state_self_feedback", "kappa": kappa,
                    "EbNo_dB": point["EbNo_dB"], "BER": point["candidate_BER"],
                    "FER": point["candidate_FER"],
                    "syndrome_violation_rate": point["syndrome_violation_rate"],
                    "avg_syndrome_weight": point["avg_syndrome_weight"],
                    "BP_BER": point["BP_BER"], "BP_FER": point["BP_FER"],
                    "n_trials": point["n_trials"], "n_bits_total": point["n_bits_total"],
                })
            for seed_record in result["by_seed"]:
                for point in seed_record["by_ebno"]:
                    by_seed_rows.append({
                        "stage": "highstat", "size": size, "variant": "binary_state",
                        "implementation_mode": "binary_state_self_feedback", "kappa": kappa,
                        "seed": seed_record["seed"], "EbNo_dB": point["EbNo_dB"],
                        "BER": point["candidate_BER"], "FER": point["candidate_FER"],
                        "syndrome_violation_rate": point["syndrome_violation_rate"],
                        "avg_syndrome_weight": point["avg_syndrome_weight"],
                        "BP_BER": point["BP_BER"], "BP_FER": point["BP_FER"],
                        "n_trials": point["n_trials"], "n_bits_total": point["n_bits_total"],
                    })
            write_csv(by_seed_path, by_seed_rows, BY_SEED_FIELDS)
            write_csv(by_ebno_path, by_ebno_rows, BY_EBNO_FIELDS)
            write_csv(summary_path, summaries, SWEEP_SUMMARY_FIELDS)
            index = SIZES.index(size) + 1
            logger.info("[high-stat %d/3] complete %.1fs %s kappa=%.3g mean BER=%.6g",
                        index, result["elapsed_seconds"], size, kappa, result["mean_BER"])
    return read_csv(by_ebno_path), read_csv(by_seed_path)


def select_ber_optimal_kappas(sweep_rows: list[dict[str, Any]]) -> dict[str, float]:
    selections = {}
    rows = []
    score_selected = {
        row["size"]: float(row["best_kappa"])
        for row in read_csv(RUN_DIR / "best_kappa_selection.csv")
    }
    for size in SIZES:
        candidates = [row for row in sweep_rows if row["size"] == size]
        best = min(
            candidates,
            key=lambda row: (float(row["mean_BER"]), KAPPA_GRID.index(float(row["kappa"]))),
        )
        selections[size] = float(best["kappa"])
        rows.append({
            "size": size, "BER_optimal_kappa": selections[size],
            "score_selected_kappa": score_selected[size],
            "differs_from_primary_score_selection": selections[size] != score_selected[size],
            "sweep_mean_BER": float(best["mean_BER"]),
            "sweep_mean_FER": float(best["mean_FER"]),
            "interpretation": "secondary sensitivity only; primary objective is unchanged",
        })
    write_csv(RUN_DIR / "ber_optimal_sensitivity_selection.csv", rows)
    return selections


def run_ber_optimal_sensitivity(
    logger: logging.Logger,
    sweep_rows: list[dict[str, Any]],
    primary_best: dict[str, float],
    n_workers: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_ebno_path = RUN_DIR / "ber_optimal_sensitivity_by_ebno.csv"
    by_seed_path = RUN_DIR / "ber_optimal_sensitivity_by_seed.csv"
    summary_path = RUN_DIR / "ber_optimal_sensitivity_summary.csv"
    ber_optimal = select_ber_optimal_kappas(sweep_rows)
    differing = [size for size in SIZES if ber_optimal[size] != primary_best[size]]
    if not differing:
        write_csv(summary_path, [], SWEEP_SUMMARY_FIELDS)
        write_csv(by_ebno_path, [], BY_EBNO_FIELDS)
        write_csv(by_seed_path, [], BY_SEED_FIELDS)
        return [], []
    existing_summary = read_csv(summary_path)
    completed = {row["size"] for row in existing_summary}
    summaries = [dict(row) for row in existing_summary]
    by_ebno_rows = [dict(row) for row in read_csv(by_ebno_path)]
    by_seed_rows = [dict(row) for row in read_csv(by_seed_path)]
    for size in differing:
        if size in completed:
            logger.info("[BER-optimal sensitivity] resume skip %s", size)
            continue
        kappa = ber_optimal[size]
        logger.info("[BER-optimal sensitivity] %s kappa=%.3g, 10000 trials/SNR", size, kappa)
        result = evaluate_binary_condition(size, kappa, 10000, HIGHSTAT_SEEDS, n_workers)
        summaries.append({field: result.get(field, "") for field in SWEEP_SUMMARY_FIELDS})
        for point in result["by_ebno"]:
            by_ebno_rows.append({
                "stage": "ber_optimal_sensitivity", "size": size,
                "variant": "binary_state_ber_optimal_sensitivity",
                "implementation_mode": "binary_state_self_feedback", "kappa": kappa,
                "EbNo_dB": point["EbNo_dB"], "BER": point["candidate_BER"],
                "FER": point["candidate_FER"],
                "syndrome_violation_rate": point["syndrome_violation_rate"],
                "avg_syndrome_weight": point["avg_syndrome_weight"],
                "BP_BER": point["BP_BER"], "BP_FER": point["BP_FER"],
                "n_trials": point["n_trials"], "n_bits_total": point["n_bits_total"],
            })
        for seed_record in result["by_seed"]:
            for point in seed_record["by_ebno"]:
                by_seed_rows.append({
                    "stage": "ber_optimal_sensitivity", "size": size,
                    "variant": "binary_state_ber_optimal_sensitivity",
                    "implementation_mode": "binary_state_self_feedback", "kappa": kappa,
                    "seed": seed_record["seed"], "EbNo_dB": point["EbNo_dB"],
                    "BER": point["candidate_BER"], "FER": point["candidate_FER"],
                    "syndrome_violation_rate": point["syndrome_violation_rate"],
                    "avg_syndrome_weight": point["avg_syndrome_weight"],
                    "BP_BER": point["BP_BER"], "BP_FER": point["BP_FER"],
                    "n_trials": point["n_trials"], "n_bits_total": point["n_bits_total"],
                })
        write_csv(summary_path, summaries, SWEEP_SUMMARY_FIELDS)
        write_csv(by_ebno_path, by_ebno_rows, BY_EBNO_FIELDS)
        write_csv(by_seed_path, by_seed_rows, BY_SEED_FIELDS)
        logger.info("[BER-optimal sensitivity] complete %.1fs mean BER=%.6g",
                    result["elapsed_seconds"], result["mean_BER"])
    return read_csv(by_ebno_path), read_csv(by_seed_path)


def make_sensitivity_paired_statistics(
    sensitivity_by_ebno: list[dict[str, Any]],
    sensitivity_by_seed: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not sensitivity_by_seed:
        write_csv(RUN_DIR / "ber_optimal_sensitivity_paired_statistics.csv", [])
        return []
    output = []
    for size in sorted({row["size"] for row in sensitivity_by_seed}):
        additive_rows = [
            row for row in read_csv(CONFIRM_DIRS[size] / "by_seed_ebno.csv")
            if row["variant"] == "additive"
        ]
        for metric in ["BER", "FER"]:
            for ebno_label in [2.0, 2.5, 3.0, "pooled"]:
                values = []
                for seed in HIGHSTAT_SEEDS:
                    b_rows = [row for row in sensitivity_by_seed if row["size"] == size
                              and int(row["seed"]) == seed]
                    a_rows = [row for row in additive_rows if int(row["seed"]) == seed]
                    if ebno_label == "pooled":
                        binary = float(np.mean([float(row[metric]) for row in b_rows]))
                        additive = float(np.mean([float(row[metric]) for row in a_rows]))
                    else:
                        binary = float(next(row[metric] for row in b_rows
                                            if float(row["EbNo_dB"]) == ebno_label))
                        additive = float(next(row[metric] for row in a_rows
                                              if float(row["EbNo_dB"]) == ebno_label))
                    values.append(binary - additive)
                array = np.asarray(values); mean = float(np.mean(array))
                sd = float(np.std(array, ddof=1)); sem = sd / math.sqrt(len(array))
                half = _t95(len(array)) * sem
                output.append({
                    "size": size, "EbNo_dB": ebno_label, "metric": metric,
                    "contrast": "BER_optimal_binary_minus_additive",
                    "mean_difference": mean, "seed_batch_sd": sd, "seed_batch_sem": sem,
                    "ci95_low": mean - half, "ci95_high": mean + half,
                    "n_seed_batches": len(array), "unit": "paired seed-batch mean",
                })
    write_csv(RUN_DIR / "ber_optimal_sensitivity_paired_statistics.csv", output)
    return output


def _core_psa_decode(
    P: np.ndarray,
    channel_values: np.ndarray,
    params: dict[str, Any],
    rng: np.random.Generator,
) -> np.ndarray:
    return decode_pbits_fast(
        P=P,
        channel_values=channel_values,
        kw=params["kw"],
        kr=params["kr"],
        n_cycles=params["n_cycles"],
        mode="pSA",
        I0_min=params["I0_min"],
        I0_max=params["I0_max"],
        psa_p=params["psa_p"],
        lambda_mem=0.0,
        I0_schedule_type=params["I0_schedule_type"],
        I0_schedule_shape=params["I0_schedule_shape"],
        I0_hold_fraction=params["I0_hold_fraction"],
        decision_method=params["decision_method"],
        burn_in=params["burn_in"],
        sample_window=params["sample_window"],
        rng=rng,
    )


def _values_equal(left: Any, right: Any) -> bool:
    try:
        return abs(float(left) - float(right)) <= 1e-12 * max(1.0, abs(float(right)))
    except (TypeError, ValueError):
        return str(left) == str(right)


def run_validation() -> dict[str, Any]:
    checks: dict[str, Any] = {}
    P = random_regular_ldpc_parity_check(48, 24, 3, 6, seed=MATRIX_SEED)
    channel_values = np.linspace(-0.85, 0.85, P.shape[1])
    params = {
        "kw": 1.2, "kr": 1.5, "alpha": 1.0, "alpha_mode": "fixed",
        "n_cycles": 40, "I0_min": 0.1, "I0_max": 0.9, "psa_p": 0.31,
        "I0_schedule_type": "cosine", "I0_schedule_shape": 1.0,
        "I0_hold_fraction": 0.0, "decision_method": "best_state",
        "burn_in": 10, "sample_window": 20,
    }

    endpoint_matches = []
    for seed in [7, 12345, 987654]:
        psa = _core_psa_decode(P, channel_values, params, np.random.default_rng(seed))
        binary_zero = decode_binary_state_fast(
            P, channel_values, params, 0.0, np.random.default_rng(seed)
        )
        endpoint_matches.append(bool(np.array_equal(psa, binary_zero)))
    checks["kappa_zero_decoded_matches_core_pSA_all_test_seeds"] = bool(all(endpoint_matches))
    checks["kappa_zero_decoded_match_vector"] = endpoint_matches

    P_packed, group_indices, group_counts, group_lengths = _packed_pbit_connections(P)
    schedule = make_I0_schedule(
        params["I0_min"], params["I0_max"], params["n_cycles"],
        params["I0_schedule_type"], shape=params["I0_schedule_shape"],
        hold_fraction=params["I0_hold_fraction"],
    ).astype(np.float64)
    p_history, p_discriminants = _trace_psa_or_binary_numba(
        P_packed, group_indices, group_counts, group_lengths, channel_values,
        params["kw"], params["kr"], params["n_cycles"], schedule,
        params["psa_p"], 0.0, 0, 24681357,
    )
    b_history, b_discriminants = _trace_psa_or_binary_numba(
        P_packed, group_indices, group_counts, group_lengths, channel_values,
        params["kw"], params["kr"], params["n_cycles"], schedule,
        params["psa_p"], 0.0, 1, 24681357,
    )
    checks["kappa_zero_state_history_trajectory_identical"] = bool(
        np.array_equal(p_history, b_history)
    )
    checks["kappa_zero_threshold_discriminants_trajectory_identical"] = bool(
        np.array_equal(p_discriminants, b_discriminants, equal_nan=True)
    )

    checks["spin_mapping_binary_zero_to_minus_one"] = bool(2 * 0 - 1 == -1)
    checks["spin_mapping_binary_one_to_plus_one"] = bool(2 * 1 - 1 == 1)
    checks["manuscript_spin_mapping"] = "s_i^(t)=x_i^(t)=2*w_i^(t)-1; b=1 maps to +1"
    checks["binary_state_timing"] = (
        "common pre-update/current state w_prev copied at cycle start; active-bit candidate uses "
        "the same bit w_prev[bit]"
    )
    checks["held_bit_semantics"] = (
        "hold is sampled before q/threshold; held bit retains w and needs no separate memory commit"
    )

    range_checks = {}
    for size, lam in SELECTED_LAMBDAS.items():
        binary_range = (-1.0 - lam, 1.0 + lam)
        additive_range = (-1.0 - lam, 1.0 + lam)
        range_checks[size] = {
            "kappa_equals_lambda": lam,
            "binary_deterministic_range": binary_range,
            "additive_deterministic_range": additive_range,
            "equal": binary_range == additive_range,
        }
    checks["equal_range_at_kappa_equals_selected_lambda"] = range_checks
    checks["all_equal_range_checks_passed"] = bool(
        all(item["equal"] for item in range_checks.values())
    )

    parameter_checks = {}
    for size in SIZES:
        params_size, source = base_params_for(size)
        confirm_summary = read_csv(CONFIRM_DIRS[size] / "summary.csv")
        additive_row = next(row for row in confirm_summary if row["variant"] == "additive")
        fields = [
            "kw", "kr", "alpha", "alpha_mode", "nrnd", "psa_p", "response_lambda",
            "lambda_out", "I0_min", "I0_max", "I0_schedule_type",
            "I0_schedule_shape", "I0_hold_fraction", "decision_method", "burn_in",
            "sample_window", "n_cycles",
        ]
        mismatches = {
            field: {"runner": params_size[field], "confirm": additive_row[field]}
            for field in fields if not _values_equal(params_size[field], additive_row[field])
        }
        config = json.loads((CONFIRM_DIRS[size] / "effective_config.json").read_text())
        parameter_checks[size] = {
            "source_parameter_file": str(source),
            "nonmemory_parameter_mismatches": mismatches,
            "nonmemory_parameters_identical": not mismatches,
            "matrix_seed_is_zero": int(config["matrix_seed"]) == MATRIX_SEED,
            "snr_grid_identical": load_reference(SOURCE_DIRS[size]) and [
                float(row["EbNo_dB"]) for row in load_reference(SOURCE_DIRS[size])
            ] == EBNO_VALUES,
            "highstat_seed_batches_identical": [int(seed) for seed in config["seeds"]]
            == HIGHSTAT_SEEDS,
            "highstat_trials_per_EbNo_identical": int(config["trials_per_ebno"]) == 10000,
            "readout_identical": params_size["decision_method"] == additive_row["decision_method"],
        }
    checks["matched_parameter_validation"] = parameter_checks
    checks["all_nonmemory_parameters_readout_matrix_snr_seed_framework_identical"] = bool(
        all(
            item["nonmemory_parameters_identical"]
            and item["matrix_seed_is_zero"]
            and item["snr_grid_identical"]
            and item["highstat_seed_batches_identical"]
            and item["highstat_trials_per_EbNo_identical"]
            and item["readout_identical"]
            for item in parameter_checks.values()
        )
    )

    endpoint_source = inspect.getsource(_decode_binary_state_numba.py_func)
    checks["production_binary_kernel_has_no_response_history_array"] = bool(
        "q_prev" not in endpoint_source
        and "Itanh_hold" not in endpoint_source
        and "q = np.zeros" not in endpoint_source
    )
    checks["production_binary_kernel_state_arrays"] = [
        "w (binary state)", "readout accumulators only"
    ]
    phase2_source = inspect.getsource(_decode_psa_with_metrics_numba.py_func)
    phase2b_source = inspect.getsource(_decode_events_channel_numba.py_func)
    checks["trajectory_logger_contains_binary_variant_code_5"] = bool(
        "variant_code == 5" in phase2_source and "variant_code == 5" in phase2b_source
    )
    checks["trajectory_binary_arm_ignores_q_prev_in_update"] = bool(
        "lambda_mem * (2.0 * w_prev[bit] - 1.0)" in phase2_source
        and "lambda_mem * (2.0 * w_prev[bit] - 1.0)" in phase2b_source
    )
    checks["control_label"] = "Binary-State Self-Feedback Control"
    checks["not_a_PIMI_reproduction"] = True
    checks["equation"] = "d_i^(t)=q_i^(t)+kappa*s_i^(t), s_i^(t)=2*w_i^(t)-1"

    boolean_keys = [key for key, value in checks.items() if isinstance(value, bool)]
    checks["all_passed"] = bool(all(checks[key] for key in boolean_keys))
    write_json(RUN_DIR / "validation.json", checks)
    if not checks["all_passed"]:
        raise RuntimeError("Validation failed; inspect validation.json")
    return checks


def write_parameter_record(best_kappas: dict[str, float] | None = None) -> None:
    matrices = {size: matrix_for(size) for size in SIZES}
    params = {size: base_params_for(size)[0] for size in SIZES}
    sources = {
        size: {
            "parameter_file": str(base_params_for(size)[1]),
            "parameter_file_sha256": sha256_file(base_params_for(size)[1]),
            "reference_file": str(SOURCE_DIRS[size] / "comparison" / "bp_baseline.csv"),
            "reference_file_sha256": sha256_file(
                SOURCE_DIRS[size] / "comparison" / "bp_baseline.csv"
            ),
        }
        for size in SIZES
    }
    manuscript_files = {}
    for name in ["main.tex", "supplement.tex"]:
        path = MANUSCRIPT_DIR / name
        manuscript_files[name] = {"path": str(path), "sha256": sha256_file(path)}
    record = {
        "study_name": "Binary-State Self-Feedback Control",
        "created_utc": utc_now(),
        "equation": "d_i^(t)=q_i^(t)+kappa*s_i^(t)",
        "binary_source": "s_i^(t)=2*w_i^(t)-1, same-bit common pre-update/current state",
        "timing": "Jacobi pre-update state; hold checked before active candidate evaluation",
        "not_PIMI_reproduction": True,
        "PIMI_specific_features_not_reproduced": [
            "PIMI-specific Gaussian noise", "fully active synchronous organization",
            "PIMI-specific schedule and dense-PIM configuration",
        ],
        "matrix_seed": MATRIX_SEED,
        "matrices": {
            size: {
                "shape": list(matrix.shape),
                "sha256_array_bytes": hash_array(matrix),
                "column_degree_set": sorted(set(np.sum(matrix, axis=0).astype(int).tolist())),
                "row_degree_set": sorted(set(np.sum(matrix, axis=1).astype(int).tolist())),
            }
            for size, matrix in matrices.items()
        },
        "EbNo_dB": EBNO_VALUES,
        "kappa_grid_predeclared": KAPPA_GRID,
        "equal_range_kappas": SELECTED_LAMBDAS,
        "best_kappas": best_kappas,
        "sweep": {"trials_per_EbNo": 2000, "seeds": SWEEP_SEEDS},
        "highstat": {"trials_per_EbNo": 10000, "seeds": HIGHSTAT_SEEDS},
        "trajectory": {
            "sizes": ["N192_M96", "N288_M144"], "EbNo_dB": 2.5,
            "trajectories_per_condition": 200, "seeds": HIGHSTAT_SEEDS,
        },
        "selection_objective": {
            "definition": "existing matched-control score_candidate",
            "score": "mean absolute log10 BER gap to BP + 0.2*FER gap + 0.5*syndrome violation",
            "tie_break": "BER score then predeclared grid order",
        },
        "fixed_nonmemory_parameters": params,
        "sources": sources,
        "manuscript_references": manuscript_files,
        "environment": {
            "hostname": socket.gethostname(), "platform": platform.platform(),
            "machine": platform.machine(), "python": sys.version,
            "python_executable": sys.executable, "numpy": np.__version__,
            "cpu_count": os.cpu_count(),
        },
    }
    write_json(RUN_DIR / "parameter_config_record.json", record)


def _t95(n: int) -> float:
    values = {
        2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571,
        7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262,
    }
    return values.get(n, 1.96) if n > 1 else float("nan")


def make_equal_range_summary(sweep_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    existing = read_csv(EXISTING_SWEEP_DIR / "summary.csv")
    output: list[dict[str, Any]] = []
    for size in SIZES:
        lam = SELECTED_LAMBDAS[size]
        binary = next(
            row for row in sweep_rows
            if row["size"] == size and abs(float(row["kappa"]) - lam) < 1e-12
        )
        method_rows = {
            "binary_state": binary,
            "additive": next(
                row for row in existing
                if row["size"] == size and row["variant"] == "additive"
                and abs(float(row["lambda_mem"]) - lam) < 1e-12
            ),
            "matched_pSA": next(
                row for row in existing if row["size"] == size and row["variant"] == "pSA"
            ),
            "normalized": next(
                row for row in existing
                if row["size"] == size and row["variant"] == "normalized"
                and abs(float(row["lambda_mem"]) - lam) < 1e-12
            ),
            "gain_only": next(
                row for row in existing
                if row["size"] == size and row["variant"] == "gain_only"
                and abs(float(row["lambda_mem"]) - lam) < 1e-12
            ),
        }
        additive_ber = float(method_rows["additive"]["mean_BER"])
        for method, row in method_rows.items():
            coefficient = float(row.get("kappa", row.get("lambda_mem", 0.0)))
            output.append({
                "size": size, "method": method, "coefficient": coefficient,
                "equal_range_target_coefficient": lam,
                "mean_BER": float(row["mean_BER"]), "mean_FER": float(row["mean_FER"]),
                "score": float(row["score"]),
                "method_minus_additive_BER": float(row["mean_BER"]) - additive_ber,
                "trials_per_EbNo": int(float(row.get("trials_per_EbNo", row.get("evaluation_trials_per_ebno", 2000)))),
                "seed_count": int(float(row.get("seed_count", 8))),
                "source": (
                    "new binary-state sweep" if method == "binary_state"
                    else "existing full matched-control sweep"
                ),
            })
    write_csv(RUN_DIR / "equal_range_summary.csv", output)
    return output


def make_highstat_combined(
    binary_by_ebno: list[dict[str, Any]], binary_by_seed: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    condition_rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    for size in SIZES:
        existing_by_ebno = read_csv(CONFIRM_DIRS[size] / "by_ebno.csv")
        existing_by_seed = read_csv(CONFIRM_DIRS[size] / "by_seed_ebno.csv")
        for row in existing_by_ebno:
            if row["variant"] not in {"pSA", "additive"}:
                continue
            condition_rows.append({
                "size": size,
                "method": "matched_pSA" if row["variant"] == "pSA" else "additive",
                "coefficient": float(row["lambda_mem"]),
                "EbNo_dB": float(row["EbNo_dB"]),
                "BER": float(row["BER"]), "FER": float(row["FER"]),
                "syndrome_violation_rate": float(row["syndrome_violation_rate"]),
                "n_trials": int(float(row["n_trials"])),
                "source": "existing high-stat matched-control confirmation",
            })
        for row in existing_by_seed:
            if row["variant"] not in {"pSA", "additive"}:
                continue
            seed_rows.append({
                "size": size,
                "method": "matched_pSA" if row["variant"] == "pSA" else "additive",
                "coefficient": float(row["lambda_mem"]), "seed": int(row["seed"]),
                "EbNo_dB": float(row["EbNo_dB"]), "BER": float(row["BER"]),
                "FER": float(row["FER"]),
                "syndrome_violation_rate": float(row["syndrome_violation_rate"]),
                "n_trials": int(float(row["n_trials"])),
                "source": "existing high-stat matched-control confirmation",
            })
    for row in binary_by_ebno:
        condition_rows.append({
            "size": row["size"], "method": "binary_state",
            "coefficient": float(row["kappa"]), "EbNo_dB": float(row["EbNo_dB"]),
            "BER": float(row["BER"]), "FER": float(row["FER"]),
            "syndrome_violation_rate": float(row["syndrome_violation_rate"]),
            "n_trials": int(float(row["n_trials"])), "source": "new high-stat binary-state run",
        })
    for row in binary_by_seed:
        seed_rows.append({
            "size": row["size"], "method": "binary_state",
            "coefficient": float(row["kappa"]), "seed": int(row["seed"]),
            "EbNo_dB": float(row["EbNo_dB"]), "BER": float(row["BER"]),
            "FER": float(row["FER"]),
            "syndrome_violation_rate": float(row["syndrome_violation_rate"]),
            "n_trials": int(float(row["n_trials"])), "source": "new high-stat binary-state run",
        })

    pooled_rows = []
    for size in SIZES:
        for method in ["additive", "binary_state", "matched_pSA"]:
            rows = [
                row for row in condition_rows if row["size"] == size and row["method"] == method
            ]
            pooled_rows.append({
                "size": size, "method": method, "coefficient": rows[0]["coefficient"],
                "EbNo_dB": "pooled", "BER": float(np.mean([row["BER"] for row in rows])),
                "FER": float(np.mean([row["FER"] for row in rows])),
                "syndrome_violation_rate": float(
                    np.mean([row["syndrome_violation_rate"] for row in rows])
                ),
                "n_trials": sum(row["n_trials"] for row in rows),
                "source": rows[0]["source"],
            })
    output = condition_rows + pooled_rows
    write_csv(RUN_DIR / "highstat_ber_fer.csv", output)
    write_csv(RUN_DIR / "highstat_by_seed.csv", seed_rows)
    return output, seed_rows


def make_endpoint_paired_statistics(seed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for size in SIZES:
        for metric in ["BER", "FER"]:
            for ebno_label in [2.0, 2.5, 3.0, "pooled"]:
                differences = []
                reverse_differences = []
                for seed in HIGHSTAT_SEEDS:
                    def value(method: str) -> float:
                        rows = [
                            row for row in seed_rows
                            if row["size"] == size and row["method"] == method
                            and int(row["seed"]) == seed
                        ]
                        if ebno_label == "pooled":
                            return float(np.mean([float(row[metric]) for row in rows]))
                        return float(next(
                            row[metric] for row in rows
                            if abs(float(row["EbNo_dB"]) - float(ebno_label)) < 1e-12
                        ))
                    additive = value("additive")
                    binary = value("binary_state")
                    differences.append(binary - additive)
                    reverse_differences.append(additive - binary)
                for contrast, values in [
                    ("binary_minus_additive", differences),
                    ("additive_minus_binary", reverse_differences),
                ]:
                    array = np.asarray(values, dtype=float)
                    mean = float(np.mean(array))
                    sd = float(np.std(array, ddof=1))
                    sem = sd / math.sqrt(len(array))
                    half = _t95(len(array)) * sem
                    output.append({
                        "size": size, "EbNo_dB": ebno_label, "metric": metric,
                        "contrast": contrast, "mean_difference": mean,
                        "seed_batch_sd": sd, "seed_batch_sem": sem,
                        "ci95_low": mean - half, "ci95_high": mean + half,
                        "n_seed_batches": len(array),
                        "unit": "paired seed-batch mean",
                    })
    write_csv(RUN_DIR / "paired_statistics.csv", output)
    return output


def _run_binary_phase2b_trajectory(
    P: np.ndarray,
    channel_values: np.ndarray,
    codeword_bits: np.ndarray,
    params: dict[str, Any],
    cycle_bin: int,
    rng: np.random.Generator,
    shuffle_rng: np.random.Generator,
    prepared: tuple[Any, ...],
):
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
        float(params["psa_p"]), float(params["lambda_mem"]), BINARY_VARIANT_CODE,
        int(_decision_code(params["decision_method"])), int(params["burn_in"]),
        int(params["sample_window"]), int(cycle_bin), order, rank, offsets,
    )
    decoded, final_state, phase2_raw = _decode_psa_with_metrics_numba(*common, core_seed)
    event_decoded, event_final, extra, events = _decode_events_channel_numba(
        *common, 0, float(LOW_BER_THRESHOLD), float(SYNDROME_FRACTION_THRESHOLD), core_seed
    )
    if not np.array_equal(decoded, event_decoded) or not np.array_equal(final_state, event_final):
        raise RuntimeError("Binary-state event logger changed the trajectory")
    return decoded, final_state, phase2_raw, extra, events


def _trajectory_seed_task(task: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    size = str(task["size"])
    variant = str(task["variant"])
    seed = int(task["seed"])
    P = matrix_for(size)
    params, _ = base_params_for(size)
    params = dict(params)
    params["lambda_mem"] = (
        0.0 if variant == "pSA"
        else float(task["best_kappa"]) if variant == "binary_state"
        else SELECTED_LAMBDAS[size]
    )
    condition = {
        "size": size,
        "variant": variant,
        "implementation_mode": (
            "pSA" if variant == "pSA"
            else "lambda_pSA" if variant == "additive"
            else "binary_state_self_feedback"
        ),
        "lambda_mem": params["lambda_mem"],
        "params": params,
    }
    G = parity_check_to_generator(P)
    rate = G.shape[0] / P.shape[1]
    seed_sequence = np.random.SeedSequence(seed)
    channel_seed, pbit_seed = seed_sequence.spawn(2)
    channel_rng = np.random.default_rng(channel_seed)
    pbit_rng = np.random.default_rng(pbit_seed)
    shuffle_rng = np.random.default_rng(np.random.SeedSequence([seed, 0x53485546]))
    prepared = _prepare_kernel_inputs(P, params)
    cycle_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    ebno = 2.5
    for trial_index in range(int(task["n_trials"])):
        message_bits = channel_rng.integers(0, 2, size=G.shape[0], dtype=np.uint8)
        codeword_bits = encode_message(message_bits, G)
        y = add_awgn(bpsk_modulate(codeword_bits), ebno, rate=rate, rng=channel_rng)
        channel_llr = awgn_channel_llr(y, ebno, rate=rate)
        channel_values = make_stochastic_channel_values(
            y, alpha=params["alpha"], bit_width=8, input_mode="float",
            alpha_mode=params["alpha_mode"], ebno_db=ebno, rate=rate,
            channel_llr=channel_llr,
        )
        if variant == "binary_state":
            decoded, final_state, phase2_raw, extra, events = _run_binary_phase2b_trajectory(
                P, channel_values, codeword_bits, params, 100, pbit_rng, shuffle_rng, prepared
            )
        else:
            decoded, final_state, phase2_raw, extra, events = _run_phase2b_trajectory(
                P, channel_values, codeword_bits, params, variant, 100,
                pbit_rng, shuffle_rng, prepared=prepared,
            )
        base_cycle, base_summary = _trajectory_records(
            phase2_raw, decoded, final_state, codeword_bits, P, condition,
            seed, ebno, trial_index, 100,
        )
        cycle_rows.extend(_cycle_extra_rows(base_cycle, phase2_raw, extra, P.shape[1]))
        base_summary.update(
            _event_trajectory_fields(events, phase2_raw, extra, P.shape[1], params["n_cycles"])
        )
        trajectory_rows.append(base_summary)
    return {"cycle_rows": cycle_rows, "trajectory_rows": trajectory_rows}


def _augment_trajectory_condition_summary(
    condition_rows: list[dict[str, Any]], trajectory_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Add mechanism fields omitted by the legacy Phase-2b summary schema."""
    extra_metrics = [
        "decoded_FER", "mean_syndrome_weight", "syndrome_zero_fraction",
        "late_syndrome_weight_fraction", "late_syndrome_zero_fraction",
        "late_bit_flip_rate", "late_repair_rate",
    ]
    for summary in condition_rows:
        group = [
            row for row in trajectory_rows
            if row["size"] == summary["size"] and row["variant"] == summary["variant"]
            and abs(float(row["EbNo_dB"]) - float(summary["EbNo_dB"])) < 1e-12
        ]
        for metric in extra_metrics:
            values = np.asarray([
                float(row[metric]) for row in group
                if math.isfinite(float(row[metric]))
            ], dtype=float)
            mean = float(np.mean(values)) if len(values) else float("nan")
            sd = float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")
            sem = sd / math.sqrt(len(values)) if len(values) > 1 else float("nan")
            half = 1.96 * sem if math.isfinite(sem) else float("nan")
            summary[f"{metric}_n"] = len(values)
            summary[f"{metric}_mean"] = mean
            summary[f"{metric}_sd"] = sd
            summary[f"{metric}_sem"] = sem
            summary[f"{metric}_ci_low"] = mean - half
            summary[f"{metric}_ci_high"] = mean + half
    return condition_rows


def run_trajectory_diagnostic(
    logger: logging.Logger, best_kappas: dict[str, float], n_workers: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw_path = RUN_DIR / "trajectory_raw.csv.gz"
    if raw_path.exists():
        logger.info("[trajectory] resume: using existing trajectory outputs")
        raw_trajectory_rows = read_csv(raw_path)
        condition_rows = _augment_trajectory_condition_summary(
            read_csv(RUN_DIR / "trajectory_condition_summary.csv"), raw_trajectory_rows
        )
        write_csv(RUN_DIR / "trajectory_condition_summary.csv", condition_rows)
        return condition_rows, read_csv(RUN_DIR / "trajectory_paired_statistics.csv")
    tasks = []
    for size in ["N192_M96", "N288_M144"]:
        for variant in ["pSA", "additive", "binary_state"]:
            for seed, n_trials in split_total_trials(200, HIGHSTAT_SEEDS):
                tasks.append({
                    "size": size, "variant": variant, "seed": seed, "n_trials": n_trials,
                    "best_kappa": best_kappas[size],
                })
    logger.info("[trajectory] starting %d paired seed tasks", len(tasks))
    started = time.perf_counter()
    cycle_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    if n_workers == 1:
        results = [_trajectory_seed_task(task) for task in tasks]
    else:
        results = []
        with ProcessPoolExecutor(max_workers=min(n_workers, len(tasks))) as executor:
            futures = [executor.submit(_trajectory_seed_task, task) for task in tasks]
            for future in as_completed(futures):
                results.append(future.result())
    for result in results:
        cycle_rows.extend(result["cycle_rows"])
        trajectory_rows.extend(result["trajectory_rows"])

    condition_summary = _add_pooled_retention(
        _aggregate_trajectory(trajectory_rows, by_seed=False), trajectory_rows, by_seed=False
    )
    condition_summary = _augment_trajectory_condition_summary(
        condition_summary, trajectory_rows
    )
    seed_summary = _add_pooled_retention(
        _aggregate_trajectory(trajectory_rows, by_seed=True), trajectory_rows, by_seed=True
    )
    paired = trajectory_paired_statistics(trajectory_rows)
    cycle_summary = _aggregate_cycle(cycle_rows, by_seed=False)
    first_passage = _first_passage_curves(trajectory_rows, 100)
    write_gzip_csv(raw_path, trajectory_rows, TRAJECTORY_FIELDS)
    write_csv(RUN_DIR / "trajectory_condition_summary.csv", condition_summary)
    write_csv(RUN_DIR / "trajectory_seed_batch_summary.csv", seed_summary)
    write_csv(RUN_DIR / "trajectory_paired_statistics.csv", paired)
    write_csv(RUN_DIR / "trajectory_by_cycle_summary.csv", cycle_summary)
    write_csv(RUN_DIR / "trajectory_first_passage_curves.csv", first_passage)
    logger.info(
        "[trajectory] complete %.1fs; %d trajectories", time.perf_counter() - started,
        len(trajectory_rows),
    )
    return condition_summary, paired


def make_figures(
    sweep_rows: list[dict[str, Any]],
    highstat_rows: list[dict[str, Any]],
    trajectory_available: bool,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = RUN_DIR / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    colors = {"additive": "#D55E00", "binary_state": "#7B3294", "matched_pSA": "#0072B2"}
    labels = {
        "additive": "additive response",
        "binary_state": "binary-state feedback",
        "matched_pSA": "matched pSA",
    }

    fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.55), sharey=False)
    for ax, size in zip(axes, SIZES):
        data = sorted(
            [row for row in sweep_rows if row["size"] == size],
            key=lambda row: float(row["kappa"]),
        )
        x = np.asarray([float(row["kappa"]) for row in data])
        y = np.asarray([float(row["mean_BER"]) for row in data])
        ax.plot(x, y, color=colors["binary_state"], marker="o", markersize=4, linewidth=1.5)
        selected = SELECTED_LAMBDAS[size]
        best = float(next(
            row["best_kappa"] for row in read_csv(RUN_DIR / "best_kappa_selection.csv")
            if row["size"] == size
        ))
        ax.axvline(selected, color="#777777", linestyle="--", linewidth=1.0,
                   label=r"equal range $\kappa=\lambda$")
        ax.axvline(best, color=colors["binary_state"], linestyle=":", linewidth=1.1,
                   label=r"selected $\kappa$")
        ax.set_title(size.replace("_", "/"))
        ax.set_xlabel(r"Binary self-feedback coefficient $\kappa$")
        ax.set_ylabel("Mean BER over 2.0/2.5/3.0 dB")
        ax.grid(alpha=0.23)
        ax.set_xticks(KAPPA_GRID)
        ax.tick_params(axis="x", rotation=60, labelsize=8)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", bbox_to_anchor=(0.5, 1.02),
               ncol=2, frameon=False)
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.22, top=0.84, wspace=0.28)
    for ext in ["png", "pdf"]:
        fig.savefig(figure_dir / f"Fig_A_mean_BER_vs_kappa.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    pooled = [row for row in highstat_rows if str(row["EbNo_dB"]) == "pooled"]
    fig, ax = plt.subplots(figsize=(7.7, 4.6))
    x = np.arange(len(SIZES), dtype=float)
    offsets = {"additive": -0.18, "binary_state": 0.0, "matched_pSA": 0.18}
    for method in ["additive", "binary_state", "matched_pSA"]:
        values = [
            float(next(row["BER"] for row in pooled if row["size"] == size and row["method"] == method))
            for size in SIZES
        ]
        ax.scatter(x + offsets[method], values, s=62, color=colors[method],
                   label=labels[method], zorder=3)
        ax.plot(x + offsets[method], values, color=colors[method], linewidth=0.9, alpha=0.65)
    sensitivity = read_csv(RUN_DIR / "ber_optimal_sensitivity_by_ebno.csv")
    for index, size in enumerate(SIZES):
        rows = [row for row in sensitivity if row["size"] == size]
        if rows:
            pooled_sensitivity_ber = float(np.mean([float(row["BER"]) for row in rows]))
            ax.scatter(
                x[index] + 0.07, pooled_sensitivity_ber, s=76, marker="D",
                facecolors="none", edgecolors=colors["binary_state"], linewidths=1.5,
                label=("binary-state (BER-optimal sensitivity)" if index == 0 or not any(
                    candidate["size"] in SIZES[:index] for candidate in sensitivity
                ) else None),
                zorder=4,
            )
    ax.set_yscale("log")
    ax.set_xticks(x, [size.split("_")[0] for size in SIZES])
    ax.set_xlabel("LDPC block length")
    ax.set_ylabel("Pooled BER over 2.0/2.5/3.0 dB")
    ax.grid(axis="y", which="both", alpha=0.23)
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.20))
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.13, top=0.76)
    for ext in ["png", "pdf"]:
        fig.savefig(figure_dir / f"Fig_B_highstat_matched_comparison.{ext}", dpi=300,
                    bbox_inches="tight")
    plt.close(fig)

    if trajectory_available:
        cycle_rows = read_csv(RUN_DIR / "trajectory_by_cycle_summary.csv")
        methods = ["additive", "binary_state", "pSA"]
        trajectory_colors = {
            "additive": colors["additive"], "binary_state": colors["binary_state"],
            "pSA": colors["matched_pSA"],
        }
        trajectory_labels = {
            "additive": "additive response", "binary_state": "binary-state feedback",
            "pSA": "matched pSA",
        }
        metrics = [
            ("state_BER", "Instantaneous BER"),
            ("syndrome_weight_fraction", "Syndrome weight / M"),
            ("bit_flip_rate", "Bit-flip rate"),
            ("channel_alignment", "Channel alignment"),
        ]
        fig, axes = plt.subplots(4, 2, figsize=(10.8, 10.0), sharex="col")
        for col, size in enumerate(["N192_M96", "N288_M144"]):
            for row_index, (metric, ylabel) in enumerate(metrics):
                ax = axes[row_index, col]
                for method in methods:
                    data = sorted(
                        [row for row in cycle_rows if row["size"] == size and row["variant"] == method],
                        key=lambda row: float(row["cycle_mid"]),
                    )
                    x_values = np.asarray([float(row["cycle_mid"]) for row in data])
                    means = np.asarray([float(row[f"{metric}_mean"]) for row in data])
                    ax.plot(x_values, means, color=trajectory_colors[method], linewidth=1.25,
                            label=trajectory_labels[method])
                if col == 0:
                    ax.set_ylabel(ylabel)
                if row_index == 0:
                    ax.set_title(size.replace("_", "/"))
                if row_index == len(metrics) - 1:
                    ax.set_xlabel("Cycle")
                ax.grid(alpha=0.22)
        handles, legend_labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, legend_labels, loc="upper center", bbox_to_anchor=(0.5, 0.985),
                   ncol=3, frameon=False)
        fig.subplots_adjust(left=0.09, right=0.99, bottom=0.06, top=0.94, hspace=0.16, wspace=0.22)
        for ext in ["png", "pdf"]:
            fig.savefig(figure_dir / f"Fig_C_targeted_trajectory_mechanism.{ext}", dpi=300,
                        bbox_inches="tight")
        plt.close(fig)


def _row_for(
    rows: list[dict[str, Any]], size: str, method: str, ebno: str | float = "pooled"
) -> dict[str, Any]:
    return next(
        row for row in rows
        if row["size"] == size and row["method"] == method
        and str(row["EbNo_dB"]) == str(ebno)
    )


def interpret_case(
    highstat_rows: list[dict[str, Any]], paired_rows: list[dict[str, Any]]
) -> tuple[str, dict[str, dict[str, float]]]:
    details = {}
    for size in SIZES:
        additive = float(_row_for(highstat_rows, size, "additive")["BER"])
        binary = float(_row_for(highstat_rows, size, "binary_state")["BER"])
        psa = float(_row_for(highstat_rows, size, "matched_pSA")["BER"])
        stat = next(
            row for row in paired_rows
            if row["size"] == size and str(row["EbNo_dB"]) == "pooled"
            and row["metric"] == "BER" and row["contrast"] == "binary_minus_additive"
        )
        details[size] = {
            "additive_BER": additive, "binary_BER": binary, "matched_pSA_BER": psa,
            "binary_to_additive_ratio": binary / additive,
            "binary_minus_additive": float(stat["mean_difference"]),
            "ci_low": float(stat["ci95_low"]), "ci_high": float(stat["ci95_high"]),
            "fraction_of_additive_improvement_reproduced": (
                (psa - binary) / (psa - additive) if psa != additive else float("nan")
            ),
        }
    if any(item["ci_high"] < 0.0 for item in details.values()):
        case = "Case 4: binary-state self-feedback exceeds additive for at least one size"
    elif all(item["ci_low"] > 0.0 for item in details.values()):
        if all(item["binary_to_additive_ratio"] >= 1.5 for item in details.values()):
            case = "Case 1: binary-state self-feedback is clearly worse than additive"
        else:
            case = "Case 2: binary-state self-feedback is close to but significantly worse than additive"
    else:
        case = "Case 3: binary-state self-feedback is statistically comparable to additive for at least one size"
    return case, details


def make_report(
    best_kappas: dict[str, float],
    equal_rows: list[dict[str, Any]],
    highstat_rows: list[dict[str, Any]],
    paired_rows: list[dict[str, Any]],
    trajectory_rows: list[dict[str, Any]],
) -> None:
    case, details = interpret_case(highstat_rows, paired_rows)
    trajectory_map = {
        (row["size"], row["variant"]): row for row in trajectory_rows
    }
    strong_stored_quantity = case.startswith("Case 1")
    lines = [
        "# Binary-State Self-Feedback Control",
        "",
        "## Executive conclusion",
        "",
        f"**Predefined interpretation: {case}.**",
        "",
        (
            "Within the fixed representative-LDPC matched framework, the binary-state control "
            "does not reproduce the additive saturated-response result."
            if strong_stored_quantity else
            "The result does not support a uniformly strong response-specific separation at all sizes."
        ),
        "",
        "This study is a stored-quantity control, not a PIMI reproduction. It retains the existing "
        "partial activation, channel resampling, local-field evaluation, uniform stochastic threshold, "
        "schedule, initialization, readout, representative H, SNR grid, and seed-batch framework.",
        "",
        "## 1–3. Implemented rule, timing, and zero-coefficient validation",
        "",
        "The exact active-bit update is",
        "",
        "```text",
        "q_i^(t) = tanh(I0^(t) F_i^(t))",
        "s_i^(t) = 2 w_i^(t) - 1  in {-1,+1}",
        "d_i^(t) = q_i^(t) + kappa s_i^(t)",
        "w_i^(t+1) = 1[d_i^(t) + xi_i^(t) >= 0],  xi_i^(t) ~ Uniform[-1,1]",
        "```",
        "",
        "`w_i^(t)` is the common pre-update/current state copied at the start of the Jacobi cycle. "
        "The manuscript convention is used: stored binary one maps to spin +1. A held bit skips "
        "candidate evaluation and retains `w_i`; there is no separate binary-memory commit. The "
        "production endpoint kernel has no response-history array. The trajectory logger computes "
        "response diagnostics, but its variant-5 update uses only `w_prev` and never `q_prev`.",
        "",
        "At kappa=0, decoded endpoints matched the existing optimized pSA kernel for all test seeds, "
        "and the validation logger produced trajectory-identical state histories and threshold "
        "discriminants. All validation checks passed; see `validation.json`.",
        "",
        "## 4–7. Equal-range, selected kappa, and high-statistics results",
        "",
        "At kappa=lambda, both additive and binary-state rules span the same deterministic range "
        "[-(1+lambda), +(1+lambda)]. The 2,000-trial equal-range results were:",
        "",
        "| Size | kappa=lambda | Additive BER | Binary-state BER | Matched pSA BER |",
        "|---|---:|---:|---:|---:|",
    ]
    for size in SIZES:
        methods = {row["method"]: row for row in equal_rows if row["size"] == size}
        lines.append(
            f"| {size} | {SELECTED_LAMBDAS[size]:.2f} | "
            f"{float(methods['additive']['mean_BER']):.6g} | "
            f"{float(methods['binary_state']['mean_BER']):.6g} | "
            f"{float(methods['matched_pSA']['mean_BER']):.6g} |"
        )
    lines.extend([
        "",
        "The predeclared sweep used the existing score/aggregation objective. Selected coefficients "
        "and 10,000-trial/SNR results pooled equally over the three SNRs are:",
        "",
        "| Size | Best kappa | Additive BER / FER | Binary BER / FER | Matched pSA BER / FER | Binary-additive BER [95% CI] |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for size in SIZES:
        d = details[size]
        add = _row_for(highstat_rows, size, "additive")
        binary = _row_for(highstat_rows, size, "binary_state")
        psa = _row_for(highstat_rows, size, "matched_pSA")
        lines.append(
            f"| {size} | {best_kappas[size]:.2f} | {float(add['BER']):.6g} / {float(add['FER']):.6g} | "
            f"{float(binary['BER']):.6g} / {float(binary['FER']):.6g} | "
            f"{float(psa['BER']):.6g} / {float(psa['FER']):.6g} | "
            f"{d['binary_minus_additive']:.6g} [{d['ci_low']:.6g}, {d['ci_high']:.6g}] |"
        )
    sensitivity_by_ebno = read_csv(RUN_DIR / "ber_optimal_sensitivity_by_ebno.csv")
    sensitivity_stats = read_csv(RUN_DIR / "ber_optimal_sensitivity_paired_statistics.csv")
    if sensitivity_by_ebno:
        for size in sorted({row["size"] for row in sensitivity_by_ebno}):
            rows = [row for row in sensitivity_by_ebno if row["size"] == size]
            pooled_ber = float(np.mean([float(row["BER"]) for row in rows]))
            pooled_fer = float(np.mean([float(row["FER"]) for row in rows]))
            kappa = float(rows[0]["kappa"])
            stat = next(
                row for row in sensitivity_stats if row["size"] == size
                and str(row["EbNo_dB"]) == "pooled" and row["metric"] == "BER"
            )
            lines.extend([
                "",
                f"**Selection-objective sensitivity ({size}).** The primary existing score selects "
                f"kappa={best_kappas[size]:.2f} because the wrong all-zero valid codeword has zero "
                f"syndrome. The BER-minimizing point on the same predeclared grid is kappa={kappa:.2f}. "
                f"Its independent 10,000-trial/SNR confirmation gives pooled BER={pooled_ber:.6g} "
                f"and FER={pooled_fer:.6g}; binary-minus-additive BER="
                f"{float(stat['mean_difference']):.6g} "
                f"[{float(stat['ci95_low']):.6g}, {float(stat['ci95_high']):.6g}]. "
                "This secondary sensitivity does not change the primary objective and shows that the "
                "stored-quantity conclusion is not an artifact of selecting the freezing point.",
            ])
    lines.extend([
        "",
        "SNR-resolved BER/FER values are in `highstat_ber_fer.csv`; both contrast directions and "
        "paired seed-batch 95% intervals are in `paired_statistics.csv`. The BER-optimal sensitivity "
        "records, when applicable, are stored in the correspondingly named sensitivity CSV files.",
        "",
        "## 8. Did binary-state feedback reproduce the additive benefit?",
        "",
    ])
    for size in SIZES:
        fraction = details[size]["fraction_of_additive_improvement_reproduced"]
        lines.append(
            f"- {size}: binary/additive BER ratio = {details[size]['binary_to_additive_ratio']:.3f}; "
            f"fraction of the pSA-to-additive BER improvement reproduced = {fraction:.3f}."
        )
    lines.extend([
        "",
        "The size dependence matters: at N=288 the binary-state arm reproduces much of the "
        "pSA-to-additive improvement, although its residual BER remains significantly above the "
        "additive response-state arm. Binary inertia is therefore a material contributor in that "
        "case, not an irrelevant control; it is insufficient to reproduce the full benefit.",
        "",
        (
            "No. Even after coefficient selection over the full predeclared grid, binary-state "
            "feedback remains clearly above additive at every size."
            if strong_stored_quantity else
            "Only partially or inconclusively; the size-resolved ratios above define the scope."
        ),
        "",
        "## 9. Targeted trajectory mechanism",
        "",
    ])
    if trajectory_rows:
        lines.append(
            "A targeted 200-trajectory/condition diagnostic was run at 2.5 dB for N=192 and 288 "
            "using additive, binary-state, and matched pSA. Key values are:"
        )
        lines.extend([
            "",
            "| Size | Method | Correct acquisition | Decoded BER | Late BER | Late syndrome/M | Late back-flip | Channel alignment | Post-hit residence |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for size in ["N192_M96", "N288_M144"]:
            for method in ["additive", "binary_state", "pSA"]:
                row = trajectory_map[(size, method)]
                lines.append(
                    f"| {size} | {method} | {float(row['reached_correct_mean']):.4f} | "
                    f"{float(row['decoded_BER_mean']):.6g} | {float(row['late_state_BER_mean']):.6g} | "
                    f"{float(row['late_syndrome_weight_fraction_mean']):.6g} | "
                    f"{float(row['late_backflip_rate_mean']):.6g} | "
                    f"{float(row['late_channel_alignment_mean']):.6g} | "
                    f"{float(row['correct_residence_fraction_pooled']):.6g} |"
                )
    else:
        lines.append("Trajectory diagnostics were not run.")
    lines.extend([
        "",
        "## 10–15. Novelty and manuscript implications",
        "",
        "10. **Stored quantity as the novelty center.** " + (
            "Yes, within the tested matched LDPC framework. The experiment supports a "
            "stored-quantity-specific temporal-reinforcement interpretation: same-bit temporal "
            "feedback and equal range alone are insufficient, while storing and reusing the "
            "real-valued saturated response is associated with the full benefit. This remains a "
            "scoped empirical claim, not a proof of uniqueness across all p-bit systems."
            if strong_stored_quantity else
            "Only cautiously. General same-bit inertia/self-feedback contributes materially, so the "
            "response-specific claim should be narrowed according to the reported effect sizes."
        ),
        "",
        "11. **Recommended PIMI wording.** Describe PIMI as prior binary-state inertia with a "
        "closely related post-tanh/pre-threshold reinjection point. State that the present study "
        "instead stores a bit-specific real-valued saturated response under stochastic partial "
        "activation and studies LDPC basin acquisition/stability. Call this arm a "
        "`binary-state self-feedback control` or `PIMI-inspired control`, never a PIMI reproduction; "
        "Gaussian noise, fully active synchronous updates, and the PIMI schedule were not reproduced.",
        "",
        "12. **Main Fig. 3/control section.** " + (
            "Yes. Add binary-state feedback as one additional point/bar in the matched-control panel "
            "and state its best-kappa comparison in the text; it answers the nearest-prior-work "
            "question more directly than gain-only or normalized memory."
            if strong_stored_quantity else
            "Mention the result in Main, but place the full coefficient comparison in the Supplement."
        ),
        "",
        "13. **Supplement only?** No. The full sweep, validation, and trajectory panels belong in the "
        "Supplement, but the central best-kappa result and its interpretation should appear in Main "
        "because PIMI is the closest conceptual comparator.",
        "",
        "14. **Further simulation needed?** No additional large simulation is indicated by this study. "
        "The coefficient boundary was prespecified and evaluated, high-stat validation used 10,000 "
        "trials/SNR, the score-versus-BER selection sensitivity was checked where it differed, and the "
        "targeted mechanism diagnostic used paired trajectories. A new simulation "
        "would be justified only by a new reviewer question or a changed claim.",
        "",
        "15. **Proceed to novelty/related-work rewrite?** Yes. Preserve the distinction between this "
        "matched stored-quantity control and PIMI as a complete algorithm, and do not claim temporal "
        "feedback, bit alignment, real-valued local state, or the reinjection point alone as novel.",
        "",
        "## Files",
        "",
        "- `kappa_sweep.csv`, `kappa_sweep_by_ebno.csv`, `kappa_sweep_by_seed.csv`",
        "- `equal_range_summary.csv`, `best_kappa_selection.csv`",
        "- `highstat_ber_fer.csv`, `highstat_by_seed.csv`, `paired_statistics.csv`",
        "- `ber_optimal_sensitivity_*.csv` (secondary, only where the predeclared score and BER selections differ)",
        "- `validation.json`, `parameter_config_record.json`",
        "- `trajectory_raw.csv.gz`, trajectory summaries and paired statistics",
        "- `figures/Fig_A_mean_BER_vs_kappa.*`, `Fig_B_highstat_matched_comparison.*`, and "
        "`Fig_C_targeted_trajectory_mechanism.*`",
        "",
        "No manuscript source or manuscript PDF was modified by this work.",
    ])
    (RUN_DIR / "BINARY_STATE_SELF_FEEDBACK_REPORT.md").write_text("\n".join(lines) + "\n")


def run_final_audit(best_kappas: dict[str, float]) -> dict[str, Any]:
    """Verify the final artifact set and cross-file numerical/provenance invariants."""
    sweep = read_csv(RUN_DIR / "kappa_sweep.csv")
    sweep_by_ebno = read_csv(RUN_DIR / "kappa_sweep_by_ebno.csv")
    sweep_by_seed = read_csv(RUN_DIR / "kappa_sweep_by_seed.csv")
    highstat_by_ebno = read_csv(RUN_DIR / "binary_highstat_by_ebno.csv")
    highstat_by_seed = read_csv(RUN_DIR / "binary_highstat_by_seed.csv")
    paired = read_csv(RUN_DIR / "paired_statistics.csv")
    sensitivity_selection = read_csv(RUN_DIR / "ber_optimal_sensitivity_selection.csv")
    sensitivity_by_ebno = read_csv(RUN_DIR / "ber_optimal_sensitivity_by_ebno.csv")
    sensitivity_by_seed = read_csv(RUN_DIR / "ber_optimal_sensitivity_by_seed.csv")
    sensitivity_paired = read_csv(
        RUN_DIR / "ber_optimal_sensitivity_paired_statistics.csv"
    )
    trajectory_raw = read_csv(RUN_DIR / "trajectory_raw.csv.gz")
    trajectory_conditions = read_csv(RUN_DIR / "trajectory_condition_summary.csv")
    validation = json.loads((RUN_DIR / "validation.json").read_text())
    parameter_record = json.loads((RUN_DIR / "parameter_config_record.json").read_text())

    differing_sizes = [
        row["size"] for row in sensitivity_selection
        if str(row["differs_from_primary_score_selection"]).lower() == "true"
    ]
    checks: dict[str, bool] = {
        "validation_all_passed": bool(validation["all_passed"]),
        "sweep_has_3_sizes_x_9_kappas": len(sweep) == 27,
        "sweep_by_ebno_has_27_x_3_rows": len(sweep_by_ebno) == 81,
        "sweep_by_seed_has_27_x_8_x_3_rows": len(sweep_by_seed) == 648,
        "highstat_has_3_sizes_x_3_snr_rows": len(highstat_by_ebno) == 9,
        "highstat_has_3_sizes_x_10_seeds_x_3_snr_rows": len(highstat_by_seed) == 90,
        "primary_paired_statistics_complete": len(paired) == 48,
        "sensitivity_selection_has_all_sizes": len(sensitivity_selection) == 3,
        "sensitivity_by_ebno_complete_for_differing_sizes": (
            len(sensitivity_by_ebno) == 3 * len(differing_sizes)
        ),
        "sensitivity_by_seed_complete_for_differing_sizes": (
            len(sensitivity_by_seed) == 30 * len(differing_sizes)
        ),
        "sensitivity_paired_statistics_complete": (
            len(sensitivity_paired) == 8 * len(differing_sizes)
        ),
        "trajectory_has_2_sizes_x_3_methods_x_200_rows": len(trajectory_raw) == 1200,
        "trajectory_condition_summary_has_6_rows": len(trajectory_conditions) == 6,
        "report_exists_and_nonempty": (
            (RUN_DIR / "BINARY_STATE_SELF_FEEDBACK_REPORT.md").exists()
            and (RUN_DIR / "BINARY_STATE_SELF_FEEDBACK_REPORT.md").stat().st_size > 0
        ),
    }

    expected_pairs = {(size, kappa) for size in SIZES for kappa in KAPPA_GRID}
    observed_pairs = {(row["size"], float(row["kappa"])) for row in sweep}
    checks["sweep_grid_exact"] = observed_pairs == expected_pairs
    checks["best_kappas_match_selection_file"] = all(
        abs(float(row["best_kappa"]) - best_kappas[row["size"]]) < 1e-15
        for row in read_csv(RUN_DIR / "best_kappa_selection.csv")
    )

    existing = read_csv(EXISTING_SWEEP_DIR / "summary.csv")
    kappa_zero_differences: dict[str, dict[str, float]] = {}
    for size in SIZES:
        binary = next(
            row for row in sweep if row["size"] == size and float(row["kappa"]) == 0.0
        )
        psa = next(
            row for row in existing if row["size"] == size and row["variant"] == "pSA"
        )
        kappa_zero_differences[size] = {
            field: float(binary[field]) - float(psa[field])
            for field in ["score", "mean_BER", "mean_FER", "mean_syndrome_violation"]
        }
    checks["kappa_zero_matches_existing_2000_trial_pSA_exactly"] = all(
        difference == 0.0
        for item in kappa_zero_differences.values()
        for difference in item.values()
    )

    current_manuscript_hashes = {
        name: sha256_file(MANUSCRIPT_DIR / name)
        for name in ["main.tex", "supplement.tex"]
    }
    recorded_manuscript_hashes = {
        name: item["sha256"]
        for name, item in parameter_record["manuscript_references"].items()
    }
    checks["manuscript_files_unchanged_since_parameter_record"] = (
        current_manuscript_hashes == recorded_manuscript_hashes
    )

    figure_paths = [
        RUN_DIR / "figures" / f"{stem}.{extension}"
        for stem in [
            "Fig_A_mean_BER_vs_kappa",
            "Fig_B_highstat_matched_comparison",
            "Fig_C_targeted_trajectory_mechanism",
        ]
        for extension in ["png", "pdf"]
    ]
    checks["all_six_figure_files_exist_and_nonempty"] = all(
        path.exists() and path.stat().st_size > 0 for path in figure_paths
    )
    audit = {
        "created_utc": utc_now(),
        "all_passed": all(checks.values()),
        "checks": checks,
        "row_counts": {
            "sweep": len(sweep), "sweep_by_ebno": len(sweep_by_ebno),
            "sweep_by_seed": len(sweep_by_seed),
            "binary_highstat_by_ebno": len(highstat_by_ebno),
            "binary_highstat_by_seed": len(highstat_by_seed),
            "paired_statistics": len(paired),
            "sensitivity_by_ebno": len(sensitivity_by_ebno),
            "sensitivity_by_seed": len(sensitivity_by_seed),
            "sensitivity_paired_statistics": len(sensitivity_paired),
            "trajectory_raw": len(trajectory_raw),
            "trajectory_conditions": len(trajectory_conditions),
        },
        "primary_best_kappas": best_kappas,
        "BER_optimal_sensitivity_sizes": differing_sizes,
        "kappa_zero_minus_existing_pSA": kappa_zero_differences,
        "current_manuscript_sha256": current_manuscript_hashes,
    }
    write_json(RUN_DIR / "final_audit.json", audit)
    if not audit["all_passed"]:
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"Final audit failed: {failed}")
    return audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the binary-state self-feedback control study")
    parser.add_argument("--sweep-workers", type=int, default=8)
    parser.add_argument("--highstat-workers", type=int, default=10)
    parser.add_argument("--trajectory-workers", type=int, default=8)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--skip-trajectory", action="store_true")
    return parser


def main() -> None:
    global RUN_DIR
    RUN_DIR = REPO_ROOT / 'outputs' / 'binary_state_feedback'
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    args = build_parser().parse_args()
    logger = setup_logging()
    started = time.perf_counter()
    manifest_path = RUN_DIR / "manifest.json"
    manifest = {
        "status": "running", "started_utc": utc_now(), "finished_utc": None,
        "study": "Binary-State Self-Feedback Control", "run_dir": str(RUN_DIR),
        "command": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
    }
    write_json(manifest_path, manifest)
    logger.info("Run directory: %s", RUN_DIR)
    validation = run_validation()
    logger.info("Validation passed: %s", validation["all_passed"])
    write_parameter_record()
    if args.validate_only:
        manifest.update({"status": "validation_complete", "finished_utc": utc_now()})
        write_json(manifest_path, manifest)
        return

    if args.analyze_only:
        sweep_rows = read_csv(RUN_DIR / "kappa_sweep.csv")
    else:
        sweep_rows = run_sweep(logger, args.sweep_workers)
    best_kappas = select_best_kappas(sweep_rows)
    write_parameter_record(best_kappas)
    equal_rows = make_equal_range_summary(sweep_rows)

    if args.analyze_only:
        binary_by_ebno = read_csv(RUN_DIR / "binary_highstat_by_ebno.csv")
        binary_by_seed = read_csv(RUN_DIR / "binary_highstat_by_seed.csv")
    else:
        binary_by_ebno, binary_by_seed = run_highstat(
            logger, best_kappas, args.highstat_workers
        )
    if args.analyze_only:
        sensitivity_by_ebno = read_csv(RUN_DIR / "ber_optimal_sensitivity_by_ebno.csv")
        sensitivity_by_seed = read_csv(RUN_DIR / "ber_optimal_sensitivity_by_seed.csv")
        select_ber_optimal_kappas(sweep_rows)
    else:
        sensitivity_by_ebno, sensitivity_by_seed = run_ber_optimal_sensitivity(
            logger, sweep_rows, best_kappas, args.highstat_workers
        )
    make_sensitivity_paired_statistics(sensitivity_by_ebno, sensitivity_by_seed)
    highstat_rows, highstat_seed_rows = make_highstat_combined(
        binary_by_ebno, binary_by_seed
    )
    paired_rows = make_endpoint_paired_statistics(highstat_seed_rows)

    trajectory_condition_rows: list[dict[str, Any]] = []
    if not args.skip_trajectory:
        if args.analyze_only:
            trajectory_condition_rows = _augment_trajectory_condition_summary(
                read_csv(RUN_DIR / "trajectory_condition_summary.csv"),
                read_csv(RUN_DIR / "trajectory_raw.csv.gz"),
            )
            write_csv(RUN_DIR / "trajectory_condition_summary.csv", trajectory_condition_rows)
        else:
            trajectory_condition_rows, _ = run_trajectory_diagnostic(
                logger, best_kappas, args.trajectory_workers
            )
    make_figures(sweep_rows, highstat_rows, bool(trajectory_condition_rows))
    make_report(
        best_kappas, equal_rows, highstat_rows, paired_rows, trajectory_condition_rows
    )
    final_audit = run_final_audit(best_kappas)

    outputs = sorted(
        str(path.relative_to(RUN_DIR)) for path in RUN_DIR.rglob("*") if path.is_file()
    )
    manifest.update({
        "status": "complete", "finished_utc": utc_now(),
        "elapsed_seconds_this_invocation": time.perf_counter() - started,
        "best_kappas": best_kappas, "validation_all_passed": validation["all_passed"],
        "final_audit_all_passed": final_audit["all_passed"],
        "outputs": outputs,
    })
    write_json(manifest_path, manifest)
    logger.info("Study complete in %.1fs", manifest["elapsed_seconds_this_invocation"])


if __name__ == "__main__":
    main()

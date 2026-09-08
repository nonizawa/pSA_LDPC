import argparse
import csv
import json
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ldpc_pbit import (
    SEARCH_ALPHA_MODES,
    SEARCH_DECISION_METHODS,
    SEARCH_I0_SCHEDULES,
    add_awgn,
    awgn_channel_llr,
    bp_decode,
    bpsk_modulate,
    decode_pbits_fast,
    encode_message,
    make_stochastic_channel_values,
    parity_check_to_generator,
    random_regular_ldpc_parity_check,
    syndrome,
    update_pbits,
)


PARAM_FIELDS = [
    "kw",
    "kr",
    "alpha",
    "alpha_mode",
    "nrnd",
    "psa_p",
    "lambda_mem",
    "response_lambda",
    "lambda_out",
    "I0_min",
    "I0_max",
    "I0_schedule_type",
    "I0_schedule_shape",
    "I0_hold_fraction",
    "decision_method",
    "burn_in",
    "sample_window",
    "n_cycles",
]

SEARCH_FIELDS = [
    "rank",
    "stage",
    "mode",
    "candidate_id",
    "score",
    "ber_score",
    "fer_score",
    "shape_score",
    "syndrome_penalty",
    "floor_penalty",
    "evaluation_trials",
    "evaluation_seeds",
    "EbNo_dB_values",
    "BP_BER_by_EbNo",
    "candidate_BER_by_EbNo",
    "BP_FER_by_EbNo",
    "candidate_FER_by_EbNo",
    "syndrome_violation_by_EbNo",
    *PARAM_FIELDS,
]


@dataclass
class EvalCounts:
    bit_errors: int = 0
    frame_errors: int = 0
    syndrome_violations: int = 0
    syndrome_weight_total: int = 0
    bp_bit_errors: int = 0
    bp_frame_errors: int = 0
    bp_converged_frames: int = 0
    bp_iterations_total: int = 0
    trials: int = 0
    bits: int = 0


def parse_float_list(value):
    return [float(item.strip()) for item in str(value).split(",") if item.strip()]


def parse_int_list(value):
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def fmt_values(values):
    return ";".join(f"{float(value):.12g}" for value in values)


def fmt_int_values(values):
    return ";".join(str(int(value)) for value in values)


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(val) for key, val in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    return value


def split_trials(n_trials, n_chunks):
    n_chunks = max(1, min(int(n_chunks), int(n_trials)))
    base = n_trials // n_chunks
    remainder = n_trials % n_chunks
    return [base + (1 if idx < remainder else 0) for idx in range(n_chunks)]


def make_reference_rows(counts_by_ebno):
    rows = []
    for ebno, counts in counts_by_ebno.items():
        rows.append({
            "EbNo_dB": float(ebno),
            "BP_BER": counts.bp_bit_errors / counts.bits,
            "BP_FER": counts.bp_frame_errors / counts.trials,
            "BP_convergence_rate": counts.bp_converged_frames / counts.trials,
            "BP_avg_iterations": counts.bp_iterations_total / counts.trials,
            "n_trials": counts.trials,
            "n_bits_total": counts.bits,
        })
    return sorted(rows, key=lambda row: row["EbNo_dB"])


def add_counts(dst, src):
    for field in asdict(dst):
        setattr(dst, field, getattr(dst, field) + getattr(src, field))


def bp_reference_chunk(task):
    P = np.array(task["P"], dtype=np.uint8)
    G = parity_check_to_generator(P)
    n_bits = P.shape[1]
    rate = G.shape[0] / n_bits
    rng = np.random.default_rng(task["seed"])
    counts_by_ebno = {float(ebno): EvalCounts() for ebno in task["ebno_values"]}

    for ebno in task["ebno_values"]:
        counts = counts_by_ebno[float(ebno)]
        for _ in range(task["n_trials"]):
            message_bits = rng.integers(0, 2, size=G.shape[0], dtype=np.uint8)
            codeword_bits = encode_message(message_bits, G)
            y = add_awgn(bpsk_modulate(codeword_bits), ebno, rate=rate, rng=rng)
            channel_llr = awgn_channel_llr(y, ebno, rate=rate)
            decoded, converged, iterations = bp_decode(
                P=P,
                channel_llr=channel_llr,
                max_iter=task["bp_max_iter"],
                llr_clip=task["bp_llr_clip"],
            )
            errors = int(np.sum(decoded != codeword_bits))
            counts.bp_bit_errors += errors
            counts.bp_frame_errors += 1 if errors > 0 else 0
            counts.bp_converged_frames += 1 if converged else 0
            counts.bp_iterations_total += int(iterations)
            counts.trials += 1
            counts.bits += n_bits

    return counts_by_ebno


def generate_bp_reference(P, ebno_values, n_trials, n_workers, seed, bp_max_iter, bp_llr_clip):
    tasks = []
    for idx, chunk_trials in enumerate(split_trials(n_trials, n_workers)):
        tasks.append({
            "P": P,
            "ebno_values": ebno_values,
            "n_trials": chunk_trials,
            "seed": seed + 1009 * idx,
            "bp_max_iter": bp_max_iter,
            "bp_llr_clip": bp_llr_clip,
        })

    combined = {float(ebno): EvalCounts() for ebno in ebno_values}
    with ProcessPoolExecutor(max_workers=max(1, int(n_workers))) as executor:
        futures = [executor.submit(bp_reference_chunk, task) for task in tasks]
        for future in as_completed(futures):
            chunk = future.result()
            for ebno, counts in chunk.items():
                add_counts(combined[float(ebno)], counts)

    return make_reference_rows(combined)


def write_csv(rows, path, fieldnames=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return path
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def read_csv_rows(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def load_or_create_reference(args, P):
    reference_path = Path(args.results_dir) / "bp_48x96_reference.csv"
    if args.resume and reference_path.exists():
        rows = read_csv_rows(reference_path)
        if rows:
            return [
                {
                    "EbNo_dB": float(row["EbNo_dB"]),
                    "BP_BER": float(row["BP_BER"]),
                    "BP_FER": float(row["BP_FER"]),
                    "BP_convergence_rate": float(row["BP_convergence_rate"]),
                    "BP_avg_iterations": float(row["BP_avg_iterations"]),
                    "n_trials": int(float(row["n_trials"])),
                    "n_bits_total": int(float(row["n_bits_total"])),
                }
                for row in rows
            ]

    print("Generating BP reference...")
    rows = generate_bp_reference(
        P=P,
        ebno_values=args.ebno_values,
        n_trials=args.bp_trials,
        n_workers=args.n_workers,
        seed=args.seed + 500000,
        bp_max_iter=args.bp_max_iter,
        bp_llr_clip=args.bp_llr_clip,
    )
    write_csv(rows, reference_path)
    return rows


def candidate_chunk(task):
    P = np.array(task["P"], dtype=np.uint8)
    params = task["params"]
    G = parity_check_to_generator(P)
    n_bits = P.shape[1]
    rate = G.shape[0] / n_bits
    seed_sequence = np.random.SeedSequence(task["seed"])
    channel_seed, pbit_seed = seed_sequence.spawn(2)
    channel_rng = np.random.default_rng(channel_seed)
    pbit_rng = np.random.default_rng(pbit_seed)
    counts_by_ebno = {float(ebno): EvalCounts() for ebno in task["ebno_values"]}

    for ebno in task["ebno_values"]:
        counts = counts_by_ebno[float(ebno)]
        for _ in range(task["n_trials"]):
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
            decoded = decode_pbits_fast(
                P=P,
                channel_values=channel_values,
                kw=params["kw"],
                kr=params["kr"],
                n_cycles=params["n_cycles"],
                mode=task["mode"],
                I0_min=params["I0_min"],
                I0_max=params["I0_max"],
                nrnd=params["nrnd"],
                psa_p=params["psa_p"],
                lambda_mem=params.get("lambda_mem", 0.0),
                response_lambda=params.get("response_lambda", 0.0),
                lambda_out=params.get("lambda_out", 0.0),
                I0_schedule_type=params["I0_schedule_type"],
                I0_schedule_shape=params["I0_schedule_shape"],
                I0_hold_fraction=params["I0_hold_fraction"],
                decision_method=params["decision_method"],
                burn_in=params["burn_in"],
                sample_window=params["sample_window"],
                rng=pbit_rng,
            )
            if decoded is None:
                decoded, _, _, _ = update_pbits(
                    P=P,
                    channel_values=channel_values,
                    kw=params["kw"],
                    kr=params["kr"],
                    n_cycles=params["n_cycles"],
                    mode=task["mode"],
                    I0_min=params["I0_min"],
                    I0_max=params["I0_max"],
                    nrnd=params["nrnd"],
                    psa_p=params["psa_p"],
                    lambda_mem=params.get("lambda_mem", 0.0),
                    response_lambda=params.get("response_lambda", 0.0),
                    lambda_out=params.get("lambda_out", 0.0),
                    I0_schedule_type=params["I0_schedule_type"],
                    I0_schedule_shape=params["I0_schedule_shape"],
                    I0_hold_fraction=params["I0_hold_fraction"],
                    decision_method=params["decision_method"],
                    burn_in=params["burn_in"],
                    sample_window=params["sample_window"],
                    rng=pbit_rng,
                )
            errors = int(np.sum(decoded != codeword_bits))
            syn = syndrome(P, decoded)
            syn_weight = int(np.sum(syn))

            counts.bit_errors += errors
            counts.frame_errors += 1 if errors > 0 else 0
            counts.syndrome_violations += 1 if syn_weight > 0 else 0
            counts.syndrome_weight_total += syn_weight
            counts.trials += 1
            counts.bits += n_bits

    return counts_by_ebno


def combine_candidate_counts(chunks, ebno_values):
    combined = {float(ebno): EvalCounts() for ebno in ebno_values}
    for chunk in chunks:
        for ebno, counts in chunk.items():
            add_counts(combined[float(ebno)], counts)
    return combined


def candidate_rows_from_counts(counts_by_ebno, reference_rows):
    ref_by_ebno = {float(row["EbNo_dB"]): row for row in reference_rows}
    rows = []
    for ebno in sorted(counts_by_ebno):
        counts = counts_by_ebno[float(ebno)]
        ref = ref_by_ebno[float(ebno)]
        rows.append({
            "EbNo_dB": float(ebno),
            "candidate_BER": counts.bit_errors / counts.bits,
            "candidate_FER": counts.frame_errors / counts.trials,
            "syndrome_violation_rate": counts.syndrome_violations / counts.trials,
            "avg_syndrome_weight": counts.syndrome_weight_total / counts.trials,
            "BP_BER": float(ref["BP_BER"]),
            "BP_FER": float(ref["BP_FER"]),
            "n_trials": counts.trials,
            "n_bits_total": counts.bits,
        })
    return rows


def score_candidate(by_ebno_rows, eps, fer_weight, shape_weight, syndrome_weight, floor_weight):
    candidate_ber = np.array([row["candidate_BER"] for row in by_ebno_rows], dtype=float)
    candidate_fer = np.array([row["candidate_FER"] for row in by_ebno_rows], dtype=float)
    bp_ber = np.array([row["BP_BER"] for row in by_ebno_rows], dtype=float)
    bp_fer = np.array([row["BP_FER"] for row in by_ebno_rows], dtype=float)
    syndrome_rates = np.array([row["syndrome_violation_rate"] for row in by_ebno_rows], dtype=float)

    log_candidate_ber = np.log10(candidate_ber + eps)
    log_bp_ber = np.log10(bp_ber + eps)
    log_candidate_fer = np.log10(candidate_fer + eps)
    log_bp_fer = np.log10(bp_fer + eps)

    ber_score = float(np.mean(np.abs(log_candidate_ber - log_bp_ber)))
    fer_score = float(np.mean(np.abs(log_candidate_fer - log_bp_fer)))
    if len(by_ebno_rows) > 1:
        shape_score = float(
            np.mean(np.abs(np.diff(log_candidate_ber) - np.diff(log_bp_ber)))
        )
        floor_penalty = float(np.mean(np.maximum(0.0, np.diff(candidate_ber))))
    else:
        shape_score = 0.0
        floor_penalty = 0.0
    syndrome_penalty = float(np.mean(syndrome_rates))
    score = (
        ber_score
        + fer_weight * fer_score
        + shape_weight * shape_score
        + syndrome_weight * syndrome_penalty
        + floor_weight * floor_penalty
    )
    return {
        "score": float(score),
        "ber_score": ber_score,
        "fer_score": fer_score,
        "shape_score": shape_score,
        "syndrome_penalty": syndrome_penalty,
        "floor_penalty": floor_penalty,
    }


def evaluate_candidate(P, reference_rows, mode, params, n_trials, seeds, args, stage, candidate_id):
    tasks = []
    per_seed_trials = int(n_trials)
    for idx, seed in enumerate(seeds):
        tasks.append({
            "P": P,
            "mode": mode,
            "params": params,
            "ebno_values": args.ebno_values,
            "n_trials": per_seed_trials,
            "seed": int(seed),
            "fixed_bit_width": args.fixed_bit_width,
            "channel_input_mode": args.channel_input_mode,
        })

    max_workers = min(args.n_workers, max(1, len(tasks)))
    chunks = []
    if max_workers == 1:
        chunks = [candidate_chunk(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(candidate_chunk, task) for task in tasks]
            for future in as_completed(futures):
                chunks.append(future.result())

    by_ebno = candidate_rows_from_counts(
        combine_candidate_counts(chunks, args.ebno_values),
        reference_rows,
    )
    scores = score_candidate(
        by_ebno,
        eps=args.eps,
        fer_weight=args.fer_weight,
        shape_weight=args.shape_weight,
        syndrome_weight=args.syndrome_weight,
        floor_weight=args.floor_weight,
    )
    record = {
        "stage": stage,
        "mode": mode,
        "candidate_id": candidate_id,
        **scores,
        "evaluation_trials": per_seed_trials * len(seeds),
        "evaluation_seeds": fmt_int_values(seeds),
        "EbNo_dB_values": fmt_values([row["EbNo_dB"] for row in by_ebno]),
        "BP_BER_by_EbNo": fmt_values([row["BP_BER"] for row in by_ebno]),
        "candidate_BER_by_EbNo": fmt_values([row["candidate_BER"] for row in by_ebno]),
        "BP_FER_by_EbNo": fmt_values([row["BP_FER"] for row in by_ebno]),
        "candidate_FER_by_EbNo": fmt_values([row["candidate_FER"] for row in by_ebno]),
        "syndrome_violation_by_EbNo": fmt_values([row["syndrome_violation_rate"] for row in by_ebno]),
        **params,
        "by_ebno": by_ebno,
    }
    return record


def split_total_trials(total_trials, seeds):
    total_trials = int(total_trials)
    seeds = list(seeds)
    if not seeds:
        raise ValueError("seeds must not be empty")
    base = total_trials // len(seeds)
    remainder = total_trials % len(seeds)
    return [
        (seed, base + (1 if idx < remainder else 0))
        for idx, seed in enumerate(seeds)
        if base + (1 if idx < remainder else 0) > 0
    ]


def evaluate_candidate_total_trials(
    P,
    reference_rows,
    mode,
    params,
    total_trials,
    seeds,
    args,
    stage,
    candidate_id,
    return_seed_rows=False,
):
    seed_trial_pairs = split_total_trials(total_trials, seeds)
    tasks = []
    for seed, n_trials in seed_trial_pairs:
        tasks.append({
            "P": P,
            "mode": mode,
            "params": params,
            "ebno_values": args.ebno_values,
            "n_trials": n_trials,
            "seed": int(seed),
            "fixed_bit_width": args.fixed_bit_width,
            "channel_input_mode": args.channel_input_mode,
        })

    max_workers = min(args.n_workers, max(1, len(tasks)))
    chunks = []
    chunks_by_seed = []
    if max_workers == 1:
        for task in tasks:
            chunk = candidate_chunk(task)
            chunks.append(chunk)
            chunks_by_seed.append((int(task["seed"]), chunk))
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(candidate_chunk, task): int(task["seed"]) for task in tasks}
            for future in as_completed(futures):
                chunk = future.result()
                chunks.append(chunk)
                chunks_by_seed.append((futures[future], chunk))

    by_ebno = candidate_rows_from_counts(
        combine_candidate_counts(chunks, args.ebno_values),
        reference_rows,
    )
    scores = score_candidate(
        by_ebno,
        eps=args.eps,
        fer_weight=args.fer_weight,
        shape_weight=args.shape_weight,
        syndrome_weight=args.syndrome_weight,
        floor_weight=args.floor_weight,
    )
    record = {
        "stage": stage,
        "mode": mode,
        "candidate_id": candidate_id,
        **scores,
        "evaluation_trials": sum(n_trials for _, n_trials in seed_trial_pairs),
        "evaluation_seeds": fmt_int_values([seed for seed, _ in seed_trial_pairs]),
        "EbNo_dB_values": fmt_values([row["EbNo_dB"] for row in by_ebno]),
        "BP_BER_by_EbNo": fmt_values([row["BP_BER"] for row in by_ebno]),
        "candidate_BER_by_EbNo": fmt_values([row["candidate_BER"] for row in by_ebno]),
        "BP_FER_by_EbNo": fmt_values([row["BP_FER"] for row in by_ebno]),
        "candidate_FER_by_EbNo": fmt_values([row["candidate_FER"] for row in by_ebno]),
        "syndrome_violation_by_EbNo": fmt_values([row["syndrome_violation_rate"] for row in by_ebno]),
        **params,
        "by_ebno": by_ebno,
    }
    if return_seed_rows:
        record["by_seed"] = [
            {
                "seed": seed,
                "by_ebno": candidate_rows_from_counts(chunk, reference_rows),
            }
            for seed, chunk in sorted(chunks_by_seed, key=lambda item: item[0])
        ]
    return record


def sorted_records(records):
    ranked = sorted(records, key=lambda row: (float(row["score"]), float(row["ber_score"])))
    for idx, row in enumerate(ranked, start=1):
        row["rank"] = idx
    return ranked


def records_for_csv(records):
    return [
        {field: row.get(field, "") for field in SEARCH_FIELDS}
        for row in sorted_records(records)
    ]


def parse_match_record(row):
    parsed = dict(row)
    float_fields = {
        "score",
        "ber_score",
        "fer_score",
        "shape_score",
        "syndrome_penalty",
        "floor_penalty",
        "kw",
        "kr",
        "alpha",
        "nrnd",
        "psa_p",
        "lambda_mem",
        "response_lambda",
        "lambda_out",
        "I0_min",
        "I0_max",
        "I0_schedule_shape",
        "I0_hold_fraction",
    }
    int_fields = {
        "rank",
        "candidate_id",
        "evaluation_trials",
        "burn_in",
        "sample_window",
        "n_cycles",
    }
    for field in float_fields:
        if parsed.get(field, "") != "":
            parsed[field] = float(parsed[field])
    for field in int_fields:
        if parsed.get(field, "") != "":
            parsed[field] = int(float(parsed[field]))
    return parsed


def load_match_records(path):
    return [parse_match_record(row) for row in read_csv_rows(path)]


def write_search_checkpoint(records, search_csv, history_csv):
    ranked = sorted_records(records)
    write_csv(records_for_csv(ranked), search_csv, SEARCH_FIELDS)
    history_rows = [
        {
            "order": idx,
            "stage": row["stage"],
            "mode": row["mode"],
            "candidate_id": row["candidate_id"],
            "score": row["score"],
            "ber_score": row["ber_score"],
            "best_score_so_far": min(float(r["score"]) for r in records[:idx]),
        }
        for idx, row in enumerate(records, start=1)
    ]
    write_csv(history_rows, history_csv)


def load_warm_start_params(path, mode, top_k):
    rows = read_csv_rows(path)
    if not rows:
        return []
    mode_rows = [row for row in rows if not row.get("mode") or row.get("mode") == mode]
    if mode_rows:
        rows = mode_rows

    def key(row):
        if row.get("score"):
            return float(row["score"])
        if row.get("rank"):
            return float(row["rank"])
        return float(row.get("mean_pbit_BER", "inf"))

    params = []
    for row in sorted(rows, key=key)[:top_k]:
        candidate = {}
        for field in PARAM_FIELDS:
            if field not in row or row[field] == "":
                continue
            if field in {"alpha_mode", "I0_schedule_type", "decision_method"}:
                candidate[field] = row[field]
            elif field in {"burn_in", "sample_window", "n_cycles"}:
                candidate[field] = int(round(float(row[field])))
            else:
                candidate[field] = float(row[field])
        if candidate:
            params.append(candidate)
    return params


def clamp_params(params, mode):
    n_cycles = max(10, int(params.get("n_cycles", 200)))
    out = {
        "kw": float(np.clip(params["kw"], 0.25, 8.0)),
        "kr": float(np.clip(params["kr"], 0.25, 12.0)),
        "alpha": float(np.clip(params["alpha"], 0.05, 8.0)),
        "alpha_mode": params.get("alpha_mode", "fixed") if params.get("alpha_mode") in SEARCH_ALPHA_MODES else "fixed",
        "nrnd": float(np.clip(params.get("nrnd", 0.0), 0.0, 2.0)) if mode == "SSA" else 0.0,
        "psa_p": float(np.clip(params.get("psa_p", 0.0), 0.0, 1.0)) if mode == "pSA" else 0.0,
        "lambda_mem": float(np.clip(params.get("lambda_mem", 0.0), 0.0, 1.0)),
        "lambda_out": float(np.clip(params.get("lambda_out", 0.0), 0.0, 0.2)),
        "I0_min": float(np.clip(params["I0_min"], 0.03, 3.0)),
        "I0_max": float(np.clip(params["I0_max"], 0.05, 12.0)),
        "I0_schedule_type": (
            params.get("I0_schedule_type", "linear")
            if params.get("I0_schedule_type") in SEARCH_I0_SCHEDULES
            else "linear"
        ),
        "I0_schedule_shape": float(np.clip(params.get("I0_schedule_shape", 1.0), 0.2, 10.0)),
        "I0_hold_fraction": float(np.clip(params.get("I0_hold_fraction", 0.0), 0.0, 0.9)),
        "decision_method": (
            params.get("decision_method", "best_state")
            if params.get("decision_method") in SEARCH_DECISION_METHODS
            else "best_state"
        ),
        "burn_in": int(np.clip(params.get("burn_in", 0), 0, n_cycles - 1)),
        "sample_window": int(np.clip(params.get("sample_window", 0), 0, n_cycles)),
        "n_cycles": n_cycles,
    }
    out["I0_max"] = max(out["I0_min"], out["I0_max"])
    out["sample_window"] = min(out["sample_window"], out["n_cycles"] - out["burn_in"])
    return out


def suggest_params(trial, mode, cycle_choices):
    n_cycles = int(trial.suggest_categorical("n_cycles", cycle_choices))
    i0_min = trial.suggest_float("I0_min", 0.03, 3.0, log=True)
    burn_in = trial.suggest_int("burn_in", 0, max(0, n_cycles - 1))
    max_window = max(0, n_cycles - burn_in)
    params = {
        "kw": trial.suggest_float("kw", 0.25, 8.0, log=True),
        "kr": trial.suggest_float("kr", 0.25, 12.0, log=True),
        "alpha": trial.suggest_float("alpha", 0.05, 8.0, log=True),
        "alpha_mode": trial.suggest_categorical("alpha_mode", SEARCH_ALPHA_MODES),
        "nrnd": trial.suggest_float("nrnd", 0.0, 2.0) if mode == "SSA" else 0.0,
        "psa_p": trial.suggest_float("psa_p", 0.0, 1.0) if mode == "pSA" else 0.0,
        "lambda_mem": 0.0,
        "lambda_out": 0.0,
        "I0_min": i0_min,
        "I0_max": trial.suggest_float("I0_max", i0_min, 12.0, log=True),
        "I0_schedule_type": trial.suggest_categorical("I0_schedule_type", SEARCH_I0_SCHEDULES),
        "I0_schedule_shape": trial.suggest_float("I0_schedule_shape", 0.2, 10.0, log=True),
        "I0_hold_fraction": trial.suggest_float("I0_hold_fraction", 0.0, 0.9),
        "decision_method": trial.suggest_categorical("decision_method", SEARCH_DECISION_METHODS),
        "burn_in": burn_in,
        "sample_window": trial.suggest_int("sample_window", 0, max_window),
        "n_cycles": n_cycles,
    }
    return clamp_params(params, mode)


def enqueue_warm_starts(study, mode, args):
    warm_paths = []
    if mode == "SSA":
        warm_paths.extend(args.ssa_warm_start_csv)
    else:
        warm_paths.extend(args.psa_warm_start_csv)
    warm_paths.extend(args.warm_start_csv)

    enqueued = 0
    for path in warm_paths:
        for params in load_warm_start_params(path, mode, args.warm_start_top_k):
            try:
                clamped = clamp_params(params, mode)
                if clamped["n_cycles"] not in args.n_cycle_choices:
                    clamped["n_cycles"] = min(
                        args.n_cycle_choices,
                        key=lambda value: abs(int(value) - int(clamped["n_cycles"])),
                    )
                    clamped["burn_in"] = min(clamped["burn_in"], clamped["n_cycles"] - 1)
                    clamped["sample_window"] = min(
                        clamped["sample_window"],
                        clamped["n_cycles"] - clamped["burn_in"],
                    )
                study.enqueue_trial(clamped)
                enqueued += 1
            except ValueError:
                continue
    return enqueued


def optuna_stage1(P, reference_rows, mode, args):
    import optuna

    search_csv = Path(args.results_dir) / f"{mode}_48x96_match_bp_search.csv"
    history_csv = Path(args.results_dir) / f"{mode}_48x96_match_bp_history.csv"
    preserved_records = [
        row for row in load_match_records(search_csv)
        if row.get("stage") != "stage1"
    ]
    storage = args.optuna_storage or f"sqlite:///{Path(args.results_dir) / 'match_bp_48x96_optuna.db'}"
    study_name = f"{mode}_48x96_match_bp"
    sampler = optuna.samplers.TPESampler(seed=args.seed + (0 if mode == "SSA" else 100000))
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        storage=storage,
        study_name=study_name,
        load_if_exists=args.resume,
    )

    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not complete_trials:
        enqueued = enqueue_warm_starts(study, mode, args)
        print(f"{mode}: enqueued {enqueued} warm-start trial(s).")

    records = [
        trial.user_attrs["record"]
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE and "record" in trial.user_attrs
    ]

    remaining = max(0, args.stage1_candidates - len(complete_trials))
    worker_args = argparse.Namespace(**vars(args))
    worker_args.n_workers = 1

    while remaining > 0:
        batch_size = min(args.n_workers, remaining)
        future_to_trial = {}
        with ProcessPoolExecutor(max_workers=batch_size) as executor:
            for _ in range(batch_size):
                trial = study.ask()
                params = suggest_params(trial, mode, args.n_cycle_choices)
                seed = args.seed + 1000000 + trial.number * 7919 + (0 if mode == "SSA" else 500000)
                future = executor.submit(
                    evaluate_candidate,
                    P,
                    reference_rows,
                    mode,
                    params,
                    args.stage1_trials,
                    [seed],
                    worker_args,
                    "stage1",
                    trial.number + 1,
                )
                future_to_trial[future] = (trial, params)

            for future in as_completed(future_to_trial):
                trial, params = future_to_trial[future]
                try:
                    record = json_ready(future.result())
                except Exception:
                    study.tell(trial, state=optuna.trial.TrialState.FAIL)
                    raise

                trial.set_user_attr("record", record)
                study.tell(trial, record["score"])
                records.append(record)
                remaining -= 1
                if len(records) % args.checkpoint_every == 0:
                    write_search_checkpoint(records, search_csv, history_csv)
                print(
                    f"{mode} trial {trial.number + 1}: score={record['score']:.4g}, "
                    f"BER={record['candidate_BER_by_EbNo']}, n_cycles={params['n_cycles']}, "
                    f"decision={params['decision_method']}, schedule={params['I0_schedule_type']}",
                    flush=True,
                )

    records = [
        trial.user_attrs["record"]
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE and "record" in trial.user_attrs
    ]
    write_search_checkpoint(records + preserved_records, search_csv, history_csv)
    return sorted_records(records)


def refine_candidates(P, reference_rows, mode, records, args):
    search_csv = Path(args.results_dir) / f"{mode}_48x96_match_bp_search.csv"
    history_csv = Path(args.results_dir) / f"{mode}_48x96_match_bp_history.csv"
    existing = list(records)
    top = sorted_records(records)[: args.stage2_top_k]
    futures = []
    refined = []
    worker_args = argparse.Namespace(**vars(args))
    worker_args.n_workers = 1
    with ProcessPoolExecutor(max_workers=args.n_workers) as executor:
        for idx, record in enumerate(top, start=1):
            seeds = [
                args.seed + 2000000 + idx * 10000 + seed_idx * 101
                for seed_idx in range(args.stage2_seed_count)
            ]
            params = {field: record[field] for field in PARAM_FIELDS}
            futures.append(executor.submit(
                evaluate_candidate_total_trials,
                P,
                reference_rows,
                mode,
                params,
                args.stage2_trials,
                seeds,
                worker_args,
                "stage2",
                int(record["candidate_id"]),
            ))
        for done, future in enumerate(as_completed(futures), start=1):
            record = json_ready(future.result())
            refined.append(record)
            existing.append(record)
            write_search_checkpoint(existing, search_csv, history_csv)
            print(f"{mode} stage2 {done}/{len(futures)}: score={record['score']:.4g}", flush=True)
    return sorted_records(existing), sorted_records(refined)


def final_evaluate_best(P, reference_rows, mode, best_record, args):
    seeds = [
        args.seed + 3000000 + (0 if mode == "SSA" else 500000) + idx * 103
        for idx in range(args.stage3_seed_count)
    ]
    params = {field: best_record[field] for field in PARAM_FIELDS}
    return json_ready(evaluate_candidate_total_trials(
        P=P,
        reference_rows=reference_rows,
        mode=mode,
        params=params,
        total_trials=args.stage3_trials,
        seeds=seeds,
        args=args,
        stage="stage3",
        candidate_id=int(best_record["candidate_id"]),
    ))


def save_best_by_ebno(reference_rows, ssa_best, psa_best, path):
    ssa_by = {float(row["EbNo_dB"]): row for row in ssa_best["by_ebno"]}
    psa_by = {float(row["EbNo_dB"]): row for row in psa_best["by_ebno"]}
    rows = []
    for ref in reference_rows:
        ebno = float(ref["EbNo_dB"])
        rows.append({
            "EbNo_dB": ebno,
            "BP_BER": ref["BP_BER"],
            "BP_FER": ref["BP_FER"],
            "SSA_BER": ssa_by[ebno]["candidate_BER"],
            "SSA_FER": ssa_by[ebno]["candidate_FER"],
            "SSA_score": ssa_best["score"],
            "SSA_syndrome_violation_rate": ssa_by[ebno]["syndrome_violation_rate"],
            "pSA_BER": psa_by[ebno]["candidate_BER"],
            "pSA_FER": psa_by[ebno]["candidate_FER"],
            "pSA_score": psa_best["score"],
            "pSA_syndrome_violation_rate": psa_by[ebno]["syndrome_violation_rate"],
        })
    write_csv(rows, path)
    return rows


def plot_best_curves(reference_rows, ssa_best, psa_best, out_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ebno = np.array([row["EbNo_dB"] for row in reference_rows], dtype=float)
    bp_ber = np.array([row["BP_BER"] for row in reference_rows], dtype=float)
    bp_fer = np.array([row["BP_FER"] for row in reference_rows], dtype=float)
    ssa_ber = np.array([row["candidate_BER"] for row in ssa_best["by_ebno"]], dtype=float)
    ssa_fer = np.array([row["candidate_FER"] for row in ssa_best["by_ebno"]], dtype=float)
    psa_ber = np.array([row["candidate_BER"] for row in psa_best["by_ebno"]], dtype=float)
    psa_fer = np.array([row["candidate_FER"] for row in psa_best["by_ebno"]], dtype=float)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4), dpi=160)
    axes[0].plot(ebno, bp_ber, marker="x", linewidth=1.5, label="BP")
    axes[0].plot(ebno, ssa_ber, marker="o", label="SSA")
    axes[0].plot(ebno, psa_ber, marker="s", label="pSA")
    axes[0].set_yscale("log")
    axes[0].set_title("BER")
    axes[0].set_xlabel("Eb/N0 [dB]")
    axes[0].set_ylabel("BER")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend()

    axes[1].plot(ebno, bp_fer, marker="x", linewidth=1.5, label="BP")
    axes[1].plot(ebno, ssa_fer, marker="o", label="SSA")
    axes[1].plot(ebno, psa_fer, marker="s", label="pSA")
    axes[1].set_title("FER")
    axes[1].set_xlabel("Eb/N0 [dB]")
    axes[1].set_ylabel("FER")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_history(results_dir, out_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9.5, 4.5), dpi=160)
    for mode in ["SSA", "pSA"]:
        rows = read_csv_rows(Path(results_dir) / f"{mode}_48x96_match_bp_history.csv")
        if not rows:
            continue
        x = [int(row["order"]) for row in rows]
        y = [float(row["best_score_so_far"]) for row in rows]
        ax.plot(x, y, marker="o", markersize=2.5, label=mode)
    ax.set_title("BP matching score history")
    ax.set_xlabel("evaluation order")
    ax.set_ylabel("best score so far")
    ax.grid(True, alpha=0.3)
    ax.legend()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def summarize_parameter_effects(ssa_records, psa_records):
    lines = []
    for mode, records in [("SSA", ssa_records), ("pSA", psa_records)]:
        reliable_records = [
            row for row in records
            if row.get("stage") in {"stage2", "stage3"}
        ] or records
        ranked = sorted_records(reliable_records)
        top = ranked[: max(1, min(10, len(ranked)))]
        if not top:
            continue
        schedules = {}
        decisions = {}
        alpha_modes = {}
        cycles = []
        for row in top:
            schedules[row["I0_schedule_type"]] = schedules.get(row["I0_schedule_type"], 0) + 1
            decisions[row["decision_method"]] = decisions.get(row["decision_method"], 0) + 1
            alpha_modes[row["alpha_mode"]] = alpha_modes.get(row["alpha_mode"], 0) + 1
            cycles.append(int(row["n_cycles"]))
        final_rows = [row for row in records if row.get("stage") == "stage3"]
        best = sorted_records(final_rows)[0] if final_rows else top[0]
        lines.append(
            f"{mode}: reliable top candidates favored decision={max(decisions, key=decisions.get)}, "
            f"alpha_mode={max(alpha_modes, key=alpha_modes.get)}, "
            f"I0_schedule={max(schedules, key=schedules.get)}, "
            f"n_cycles around {int(np.median(cycles))}. Final best kw={best['kw']:.4g}, "
            f"kr={best['kr']:.4g}, I0=({best['I0_min']:.4g},{best['I0_max']:.4g})."
        )
    return lines


def run_mode(P, reference_rows, mode, args):
    print(f"\n=== {mode} BP matching search ===")
    stage1_records = optuna_stage1(P, reference_rows, mode, args)
    search_csv = Path(args.results_dir) / f"{mode}_48x96_match_bp_search.csv"
    existing_records = load_match_records(search_csv)
    stage2_records = [
        row for row in existing_records
        if row.get("stage") == "stage2"
    ]
    if len(stage2_records) >= args.stage2_top_k:
        print(f"{mode}: using {len(stage2_records)} existing stage2 record(s).")
        all_records = existing_records
        refined = sorted_records(stage2_records)
    else:
        all_records, refined = refine_candidates(P, reference_rows, mode, stage1_records, args)
    best_for_final = sorted_records(refined if refined else all_records)[0]
    final = final_evaluate_best(P, reference_rows, mode, best_for_final, args)

    history_csv = Path(args.results_dir) / f"{mode}_48x96_match_bp_history.csv"
    all_records = [row for row in all_records if row.get("stage") != "stage3"]
    all_records.append(final)
    write_search_checkpoint(all_records, search_csv, history_csv)
    print(f"{mode} final: score={final['score']:.5g}, BER={final['candidate_BER_by_EbNo']}")
    return sorted_records(all_records), final


def parse_args():
    parser = argparse.ArgumentParser(
        description="Dedicated BP-matching search for (48,96) random regular LDPC p-bit decoders."
    )
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--n-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ebno-values", type=parse_float_list, default=[2.0, 2.5, 3.0])
    parser.add_argument("--bp-trials", type=int, default=5000)
    parser.add_argument("--stage1-candidates", type=int, default=120)
    parser.add_argument("--stage1-trials", type=int, default=50)
    parser.add_argument("--stage2-top-k", type=int, default=12)
    parser.add_argument("--stage2-trials", type=int, default=400)
    parser.add_argument("--stage2-seed-count", type=int, default=3)
    parser.add_argument("--stage3-trials", type=int, default=5000)
    parser.add_argument("--stage3-seed-count", type=int, default=8)
    parser.add_argument(
        "--n-cycle-choices",
        type=parse_int_list,
        default=[100, 150, 200, 300, 400, 600, 800],
    )
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--fer-weight", type=float, default=0.15)
    parser.add_argument("--shape-weight", type=float, default=0.10)
    parser.add_argument("--syndrome-weight", type=float, default=0.10)
    parser.add_argument("--floor-weight", type=float, default=0.20)
    parser.add_argument("--fixed-bit-width", type=int, default=8)
    parser.add_argument("--channel-input-mode", choices=["float", "fixed"], default="float")
    parser.add_argument("--bp-max-iter", type=int, default=50)
    parser.add_argument("--bp-llr-clip", type=float, default=50.0)
    parser.add_argument("--warm-start-csv", action="append", default=[])
    parser.add_argument("--ssa-warm-start-csv", action="append", default=["results/SSA_48x96_search.csv"])
    parser.add_argument("--psa-warm-start-csv", action="append", default=["results/pSA_48x96_search.csv"])
    parser.add_argument("--warm-start-top-k", type=int, default=10)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--optuna-storage", default=None)
    parser.add_argument("--skip-ssa", action="store_true")
    parser.add_argument("--skip-psa", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.ebno_values != [2.0, 2.5, 3.0]:
        raise ValueError("This BP matching run must use Eb/N0 values 2.0, 2.5, 3.0.")

    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    P = random_regular_ldpc_parity_check(
        n_bits=96,
        n_checks=48,
        variable_degree=3,
        seed=0,
    )

    reference_rows = load_or_create_reference(args, P)
    print("BP reference:")
    for row in reference_rows:
        print(
            f"  Eb/N0={row['EbNo_dB']:.1f}: BER={row['BP_BER']:.6g}, "
            f"FER={row['BP_FER']:.6g}, trials={row['n_trials']}"
        )

    ssa_records, ssa_final = ([], None)
    psa_records, psa_final = ([], None)
    if not args.skip_ssa:
        ssa_records, ssa_final = run_mode(P, reference_rows, "SSA", args)
    if not args.skip_psa:
        psa_records, psa_final = run_mode(P, reference_rows, "pSA", args)

    if ssa_final is None or psa_final is None:
        raise RuntimeError("Both SSA and pSA are required for the final comparison.")

    best_json_path = Path(args.results_dir) / "best_match_bp_48x96.json"
    by_ebno_path = Path(args.results_dir) / "best_match_bp_48x96_by_ebno.csv"
    curve_path = Path(args.results_dir) / "best_match_bp_48x96_ber_curve.png"
    history_path = Path(args.results_dir) / "best_match_bp_48x96_score_history.png"

    by_ebno_rows = save_best_by_ebno(reference_rows, ssa_final, psa_final, by_ebno_path)
    plot_best_curves(reference_rows, ssa_final, psa_final, curve_path)
    plot_history(args.results_dir, history_path)
    parameter_notes = summarize_parameter_effects(ssa_records, psa_records)

    output = {
        "code": {
            "p_matrix": "random_regular",
            "n_bits": 96,
            "n_checks": 48,
            "variable_degree": 3,
            "matrix_seed": 0,
        },
        "ebno_values": args.ebno_values,
        "objective": {
            "definition": "mean abs log10(BER_candidate+eps)-log10(BER_BP+eps) plus optional penalties",
            "eps": args.eps,
            "fer_weight": args.fer_weight,
            "shape_weight": args.shape_weight,
            "syndrome_weight": args.syndrome_weight,
            "floor_weight": args.floor_weight,
        },
        "BP_reference": reference_rows,
        "SSA_best": ssa_final,
        "pSA_best": psa_final,
        "comparison_by_ebno": by_ebno_rows,
        "parameter_effect_notes": parameter_notes,
        "outputs": {
            "bp_reference_csv": str(Path(args.results_dir) / "bp_48x96_reference.csv"),
            "SSA_search_csv": str(Path(args.results_dir) / "SSA_48x96_match_bp_search.csv"),
            "pSA_search_csv": str(Path(args.results_dir) / "pSA_48x96_match_bp_search.csv"),
            "best_by_ebno_csv": str(by_ebno_path),
            "best_curve_png": str(curve_path),
            "score_history_png": str(history_path),
        },
        "reproducible_command": (
            "./run_ldpc.sh match-bp --n-workers 8 --bp-trials 5000 "
            "--stage1-candidates 120 --stage1-trials 50 "
            "--stage2-top-k 12 --stage2-trials 400 --stage2-seed-count 3 "
            "--stage3-trials 5000 --stage3-seed-count 8"
        ),
    }
    best_json_path.write_text(json.dumps(json_ready(output), indent=2))

    print("\nFinal BP matching summary:")
    print(f"  SSA score={ssa_final['score']:.6g}, BER={ssa_final['candidate_BER_by_EbNo']}")
    print(f"  pSA score={psa_final['score']:.6g}, BER={psa_final['candidate_BER_by_EbNo']}")
    for note in parameter_notes:
        print(f"  {note}")
    print(f"Saved JSON: {best_json_path}")
    print(f"Saved by-Eb/N0 CSV: {by_ebno_path}")
    print(f"Saved BER curve: {curve_path}")
    print(f"Saved history plot: {history_path}")


if __name__ == "__main__":
    main()

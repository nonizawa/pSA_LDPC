import argparse
import csv
import json
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from auto_match_bp import (
    evaluate_candidate_total_trials,
    generate_bp_reference,
    json_ready,
    read_csv_rows,
    write_csv,
)
from ldpc_pbit import (
    SEARCH_ALPHA_MODES,
    SEARCH_DECISION_METHODS,
    SEARCH_I0_SCHEDULES,
    random_regular_ldpc_parity_check,
    toy_parity_check_3x6,
)


MODES = ["pSA", "lambda_pSA", "tau_pSA", "SSA", "lambda_SSA", "dual_memory"]
DEFAULT_MODES = ["pSA", "lambda_pSA", "tau_pSA", "SSA", "lambda_SSA"]
DUAL_MEMORY_MODES = ["lambda_pSA", "lambda_SSA", "dual_memory"]
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
FIXED_PARAM_FIELDS = [
    "mode",
    "source",
    "source_stage",
    "source_score",
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
MODE_OFFSETS = {
    "pSA": 100_000,
    "lambda_pSA": 200_000,
    "tau_pSA": 250_000,
    "SSA": 300_000,
    "lambda_SSA": 400_000,
    "dual_memory": 500_000,
}


def parse_float_list(value):
    return [float(item.strip()) for item in str(value).split(",") if item.strip()]


def parse_int_list(value):
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def fmt_values(values):
    return ";".join(f"{float(value):.12g}" for value in values)


def code_label(n_checks, n_bits):
    return f"{int(n_checks)}x{int(n_bits)}"


def default_config(preset):
    if preset == "24x48":
        return {
            "prefix": "lambda_memory_24x48",
            "p_matrix": "random_regular",
            "n_bits": 48,
            "n_checks": 24,
            "variable_degree": 3,
            "check_degree": 6,
            "matrix_seed": 0,
            "ebno_values": [1.5, 2.0, 2.5],
            "n_cycle_choices": [400, 800, 1600, 3200],
            "bp_trials": 2000,
            "search_candidates": 64,
            "search_trials": 80,
            "refine_top_k": 6,
            "refine_trials": 400,
            "refine_seed_count": 3,
            "final_trials": 2000,
            "final_seed_count": 8,
        }
    if preset == "lambda_mem_sweep48x96":
        return {
            "prefix": "lambda_mem_sweep_48x96",
            "p_matrix": "random_regular",
            "n_bits": 96,
            "n_checks": 48,
            "variable_degree": 3,
            "check_degree": 6,
            "matrix_seed": 0,
            "ebno_values": [2.0, 2.5, 3.0],
            "n_cycle_choices": [1600, 3200, 6400, 9600, 12800],
            "bp_trials": 5000,
            "search_candidates": 96,
            "search_trials": 80,
            "refine_top_k": 8,
            "refine_trials": 600,
            "refine_seed_count": 4,
            "final_trials": 5000,
            "final_seed_count": 8,
            "lambda_values": [round(0.05 * i, 2) for i in range(21)],
        }
    if preset == "lambda_mem_sweep48x96_fine":
        return {
            "prefix": "lambda_mem_sweep_48x96_fine",
            "p_matrix": "random_regular",
            "n_bits": 96,
            "n_checks": 48,
            "variable_degree": 3,
            "check_degree": 6,
            "matrix_seed": 0,
            "ebno_values": [2.0, 2.5, 3.0],
            "n_cycle_choices": [1600, 3200, 6400, 9600, 12800],
            "bp_trials": 5000,
            "search_candidates": 96,
            "search_trials": 80,
            "refine_top_k": 8,
            "refine_trials": 600,
            "refine_seed_count": 4,
            "final_trials": 5000,
            "final_seed_count": 8,
            "lambda_values": [round(0.02 * i, 2) for i in range(16)],
        }
    if preset == "48x96":
        return {
            "prefix": "lambda_memory_48x96",
            "p_matrix": "random_regular",
            "n_bits": 96,
            "n_checks": 48,
            "variable_degree": 3,
            "check_degree": 6,
            "matrix_seed": 0,
            "ebno_values": [2.0, 2.5, 3.0],
            "n_cycle_choices": [1600, 3200, 6400, 9600, 12800],
            "bp_trials": 5000,
            "search_candidates": 96,
            "search_trials": 80,
            "refine_top_k": 8,
            "refine_trials": 600,
            "refine_seed_count": 4,
            "final_trials": 5000,
            "final_seed_count": 8,
        }
    if preset == "smoke":
        return {
            "prefix": "lambda_memory_smoke",
            "p_matrix": "toy",
            "n_bits": 6,
            "n_checks": 3,
            "variable_degree": 3,
            "check_degree": None,
            "matrix_seed": 0,
            "ebno_values": [1.0],
            "n_cycle_choices": [5],
            "bp_trials": 2,
            "search_candidates": 1,
            "search_trials": 2,
            "refine_top_k": 1,
            "refine_trials": 2,
            "refine_seed_count": 1,
            "final_trials": 2,
            "final_seed_count": 1,
        }
    if preset == "dual24x48":
        return {
            "prefix": "dual_memory_24x48",
            "p_matrix": "random_regular",
            "n_bits": 48,
            "n_checks": 24,
            "variable_degree": 3,
            "check_degree": 6,
            "matrix_seed": 0,
            "ebno_values": [2.5],
            "n_cycle_choices": [400, 800, 1600, 3200],
            "bp_trials": 1000,
            "search_candidates": 48,
            "search_trials": 60,
            "refine_top_k": 6,
            "refine_trials": 300,
            "refine_seed_count": 3,
            "final_trials": 1000,
            "final_seed_count": 4,
        }
    raise ValueError("preset must be 24x48, 48x96, lambda_mem_sweep48x96, lambda_mem_sweep48x96_fine, dual24x48, or smoke")


def make_matrix(args):
    if args.p_matrix == "toy":
        return toy_parity_check_3x6()
    if args.p_matrix == "random_regular":
        return random_regular_ldpc_parity_check(
            n_bits=args.n_bits,
            n_checks=args.n_checks,
            variable_degree=args.variable_degree,
            check_degree=args.check_degree,
            seed=args.matrix_seed,
        )
    raise ValueError("p_matrix must be toy or random_regular")


def load_or_create_reference(args, P, reference_path):
    if args.resume and reference_path.exists():
        rows = read_csv_rows(reference_path)
        if rows:
            trial_counts = {
                int(float(row["n_trials"]))
                for row in rows
                if row.get("n_trials", "") != ""
            }
            if trial_counts and trial_counts != {int(args.bp_trials)}:
                print(
                    f"Existing BP reference {reference_path} has n_trials={sorted(trial_counts)}; "
                    f"regenerating for --bp-trials {args.bp_trials}.",
                    flush=True,
                )
            else:
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

    rows = generate_bp_reference(
        P=P,
        ebno_values=args.ebno_values,
        n_trials=args.bp_trials,
        n_workers=args.n_workers,
        seed=args.seed + 900_000,
        bp_max_iter=args.bp_max_iter,
        bp_llr_clip=args.bp_llr_clip,
    )
    write_csv(rows, reference_path)
    return rows


def eval_args(args, n_workers=None):
    return argparse.Namespace(
        ebno_values=args.ebno_values,
        n_workers=args.n_workers if n_workers is None else n_workers,
        fixed_bit_width=args.fixed_bit_width,
        channel_input_mode=args.channel_input_mode,
        eps=args.eps,
        fer_weight=args.fer_weight,
        shape_weight=0.0,
        syndrome_weight=args.syndrome_weight,
        floor_weight=0.0,
    )


def clamp(value, low, high):
    return min(max(float(value), float(low)), float(high))


def is_psa_mode(mode):
    return mode in {"pSA", "lambda_pSA", "tau_pSA"}


def is_ssa_mode(mode):
    return mode in {"SSA", "lambda_SSA", "dual_memory"}


def clamp_params(params, mode):
    n_cycles = max(1, int(params["n_cycles"]))
    out = {
        "kw": clamp(params["kw"], 0.05, 8.0),
        "kr": clamp(params["kr"], 0.05, 12.0),
        "alpha": clamp(params["alpha"], 0.05, 8.0),
        "alpha_mode": params.get("alpha_mode", "fixed")
        if params.get("alpha_mode") in SEARCH_ALPHA_MODES
        else "fixed",
        "nrnd": clamp(params.get("nrnd", 0.0), 0.0, 2.0) if is_ssa_mode(mode) else 0.0,
        "psa_p": clamp(params.get("psa_p", 0.0), 0.0, 1.0) if is_psa_mode(mode) else 0.0,
        "lambda_mem": 0.0,
        "response_lambda": 0.0,
        "lambda_out": 0.0,
        "I0_min": clamp(params["I0_min"], 0.03, 3.0),
        "I0_max": clamp(params["I0_max"], 0.05, 12.0),
        "I0_schedule_type": params.get("I0_schedule_type", "linear")
        if params.get("I0_schedule_type") in SEARCH_I0_SCHEDULES
        else "linear",
        "I0_schedule_shape": clamp(params.get("I0_schedule_shape", 1.0), 0.05, 10.0),
        "I0_hold_fraction": clamp(params.get("I0_hold_fraction", 0.0), 0.0, 0.9),
        "decision_method": params.get("decision_method", "best_state")
        if params.get("decision_method") in SEARCH_DECISION_METHODS
        else "best_state",
        "burn_in": int(np.clip(params.get("burn_in", 0), 0, n_cycles - 1)),
        "sample_window": int(np.clip(params.get("sample_window", 0), 0, n_cycles)),
        "n_cycles": n_cycles,
    }
    if mode == "lambda_pSA":
        out["lambda_mem"] = clamp(params.get("lambda_mem", 0.0), 0.0, 0.95)
    elif mode == "tau_pSA":
        out["response_lambda"] = clamp(params.get("response_lambda", 0.0), 0.0, 0.99)
    elif mode in {"lambda_SSA", "dual_memory"}:
        out["lambda_mem"] = clamp(params.get("lambda_mem", 1.0), 0.0, 1.0)
    if mode == "dual_memory":
        out["lambda_out"] = clamp(params.get("lambda_out", 0.02), 0.0, 0.2)
    out["I0_max"] = max(out["I0_min"], out["I0_max"])
    out["sample_window"] = min(out["sample_window"], out["n_cycles"] - out["burn_in"])
    return out


def random_params(rng, mode, cycle_choices):
    n_cycles = int(rng.choice(cycle_choices))
    i0_min = float(np.exp(rng.uniform(np.log(0.03), np.log(3.0))))
    burn_in = int(rng.integers(0, max(1, n_cycles)))
    max_window = max(0, n_cycles - burn_in)
    params = {
        "kw": float(np.exp(rng.uniform(np.log(0.05), np.log(8.0)))),
        "kr": float(np.exp(rng.uniform(np.log(0.05), np.log(12.0)))),
        "alpha": float(np.exp(rng.uniform(np.log(0.05), np.log(8.0)))),
        "alpha_mode": str(rng.choice(SEARCH_ALPHA_MODES)),
        "nrnd": float(rng.uniform(0.0, 2.0)) if is_ssa_mode(mode) else 0.0,
        "psa_p": float(rng.uniform(0.0, 0.8 if mode == "tau_pSA" else 1.0)) if is_psa_mode(mode) else 0.0,
        "lambda_mem": 0.0,
        "response_lambda": 0.0,
        "lambda_out": 0.0,
        "I0_min": i0_min,
        "I0_max": float(np.exp(rng.uniform(np.log(i0_min), np.log(12.0)))),
        "I0_schedule_type": str(rng.choice(SEARCH_I0_SCHEDULES)),
        "I0_schedule_shape": float(np.exp(rng.uniform(np.log(0.05), np.log(10.0)))),
        "I0_hold_fraction": float(rng.uniform(0.0, 0.9)),
        "decision_method": str(rng.choice(SEARCH_DECISION_METHODS)),
        "burn_in": burn_in,
        "sample_window": int(rng.integers(0, max_window + 1)),
        "n_cycles": n_cycles,
    }
    if mode == "lambda_pSA":
        params["lambda_mem"] = float(rng.uniform(0.0, 0.95))
    elif mode == "tau_pSA":
        params["response_lambda"] = float(rng.uniform(0.0, 0.99))
    elif mode in {"lambda_SSA", "dual_memory"}:
        params["lambda_mem"] = float(rng.uniform(0.0, 1.0))
    if mode == "dual_memory":
        params["lambda_out"] = float(rng.uniform(0.0, 0.2))
    return clamp_params(params, mode)


def suggest_params(trial, mode, cycle_choices):
    n_cycles = int(trial.suggest_categorical("n_cycles", cycle_choices))
    i0_min = trial.suggest_float("I0_min", 0.03, 3.0, log=True)
    burn_in = trial.suggest_int("burn_in", 0, max(0, n_cycles - 1))
    max_window = max(0, n_cycles - burn_in)
    params = {
        "kw": trial.suggest_float("kw", 0.05, 8.0, log=True),
        "kr": trial.suggest_float("kr", 0.05, 12.0, log=True),
        "alpha": trial.suggest_float("alpha", 0.05, 8.0, log=True),
        "alpha_mode": trial.suggest_categorical("alpha_mode", SEARCH_ALPHA_MODES),
        "nrnd": trial.suggest_float("nrnd", 0.0, 2.0) if is_ssa_mode(mode) else 0.0,
        "psa_p": trial.suggest_float("psa_p", 0.0, 0.8 if mode == "tau_pSA" else 1.0) if is_psa_mode(mode) else 0.0,
        "lambda_mem": 0.0,
        "response_lambda": 0.0,
        "lambda_out": 0.0,
        "I0_min": i0_min,
        "I0_max": trial.suggest_float("I0_max", i0_min, 12.0, log=True),
        "I0_schedule_type": trial.suggest_categorical("I0_schedule_type", SEARCH_I0_SCHEDULES),
        "I0_schedule_shape": trial.suggest_float("I0_schedule_shape", 0.05, 10.0, log=True),
        "I0_hold_fraction": trial.suggest_float("I0_hold_fraction", 0.0, 0.9),
        "decision_method": trial.suggest_categorical("decision_method", SEARCH_DECISION_METHODS),
        "burn_in": burn_in,
        "sample_window": trial.suggest_int("sample_window", 0, max_window),
        "n_cycles": n_cycles,
    }
    if mode == "lambda_pSA":
        params["lambda_mem"] = trial.suggest_float("lambda_mem", 0.0, 0.95)
    elif mode == "tau_pSA":
        params["response_lambda"] = trial.suggest_float("response_lambda", 0.0, 0.99)
    elif mode in {"lambda_SSA", "dual_memory"}:
        params["lambda_mem"] = trial.suggest_float("lambda_mem", 0.0, 1.0)
    if mode == "dual_memory":
        params["lambda_out"] = trial.suggest_float("lambda_out", 0.0, 0.2)
    return clamp_params(params, mode)


def mode_seed(args, mode, candidate_id, stage_offset):
    return args.seed + MODE_OFFSETS[mode] + stage_offset + int(candidate_id) * 7919


def evaluate_record(P, reference_rows, mode, params, total_trials, seeds, args, stage, candidate_id, n_workers=None):
    record = evaluate_candidate_total_trials(
        P=P,
        reference_rows=reference_rows,
        mode=mode,
        params=params,
        total_trials=total_trials,
        seeds=seeds,
        args=eval_args(args, n_workers=args.n_workers if n_workers is None else n_workers),
        stage=stage,
        candidate_id=candidate_id,
    )
    return json_ready(record)


def run_random_search(P, reference_rows, mode, args):
    rng = np.random.default_rng(args.seed + MODE_OFFSETS[mode])
    params_list = [
        random_params(rng, mode, args.n_cycle_choices)
        for _ in range(args.search_candidates)
    ]
    return evaluate_params_list(P, reference_rows, mode, params_list, args, "search", args.search_trials, 10_000)


def run_optuna_search(P, reference_rows, mode, args):
    import optuna

    sampler = optuna.samplers.TPESampler(seed=args.seed + MODE_OFFSETS[mode])
    study = optuna.create_study(direction="minimize", sampler=sampler)
    records = []
    remaining = int(args.search_candidates)

    while remaining > 0:
        batch_size = min(args.n_workers, remaining)
        futures = {}
        with ProcessPoolExecutor(max_workers=batch_size) as executor:
            for _ in range(batch_size):
                trial = study.ask()
                params = suggest_params(trial, mode, args.n_cycle_choices)
                seed = mode_seed(args, mode, trial.number + 1, 10_000)
                future = executor.submit(
                    evaluate_record,
                    P,
                    reference_rows,
                    mode,
                    params,
                    args.search_trials,
                    [seed],
                    args,
                    "search",
                    trial.number + 1,
                    1,
                )
                futures[future] = trial

            for future in as_completed(futures):
                trial = futures[future]
                record = future.result()
                study.tell(trial, record["score"])
                records.append(record)
                remaining -= 1
                print(
                    f"{mode} trial {trial.number + 1}: score={record['score']:.4g}, "
                    f"BER={record['candidate_BER_by_EbNo']}, n_cycles={record['n_cycles']}",
                    flush=True,
                )
    return rank_records(records)


def evaluate_params_list(P, reference_rows, mode, params_list, args, stage, total_trials, stage_offset):
    records = []
    if args.n_workers <= 1:
        for idx, params in enumerate(params_list, start=1):
            record = evaluate_record(
                P,
                reference_rows,
                mode,
                params,
                total_trials,
                [mode_seed(args, mode, idx, stage_offset)],
                args,
                stage,
                idx,
                1,
            )
            records.append(record)
            print(f"{mode} {stage} {idx}/{len(params_list)}: score={record['score']:.4g}", flush=True)
        return rank_records(records)

    futures = {}
    with ProcessPoolExecutor(max_workers=args.n_workers) as executor:
        for idx, params in enumerate(params_list, start=1):
            future = executor.submit(
                evaluate_record,
                P,
                reference_rows,
                mode,
                params,
                total_trials,
                [mode_seed(args, mode, idx, stage_offset)],
                args,
                stage,
                idx,
                1,
            )
            futures[future] = idx
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            print(f"{mode} {stage} {len(records)}/{len(params_list)}: score={record['score']:.4g}", flush=True)
    return rank_records(records)


def refine_top(P, reference_rows, mode, search_records, args):
    top = rank_records(search_records)[: max(0, int(args.refine_top_k))]
    if not top:
        return []
    futures = {}
    refined = []
    worker_args = argparse.Namespace(**vars(args))
    worker_args.n_workers = 1
    with ProcessPoolExecutor(max_workers=args.n_workers) as executor:
        for idx, row in enumerate(top, start=1):
            params = {field: row[field] for field in PARAM_FIELDS}
            seeds = [
                mode_seed(args, mode, idx, 100_000 + seed_idx * 101)
                for seed_idx in range(args.refine_seed_count)
            ]
            future = executor.submit(
                evaluate_candidate_total_trials,
                P,
                reference_rows,
                mode,
                params,
                args.refine_trials,
                seeds,
                eval_args(worker_args, n_workers=1),
                "refine",
                int(row["candidate_id"]),
            )
            futures[future] = idx
        for future in as_completed(futures):
            record = json_ready(future.result())
            refined.append(record)
            print(f"{mode} refine {len(refined)}/{len(top)}: score={record['score']:.4g}", flush=True)
    return rank_records(refined)


def final_eval(P, reference_rows, mode, best_record, args):
    params = {field: best_record[field] for field in PARAM_FIELDS}
    seeds = [
        mode_seed(args, mode, idx + 1, 500_000 + idx * 103)
        for idx in range(args.final_seed_count)
    ]
    return evaluate_record(
        P,
        reference_rows,
        mode,
        params,
        args.final_trials,
        seeds,
        args,
        "final",
        int(best_record["candidate_id"]),
    )


def rank_records(records):
    ranked = sorted(records, key=lambda row: (float(row["score"]), float(row["ber_score"])))
    for idx, row in enumerate(ranked, start=1):
        row["rank"] = idx
    return ranked


def ebno_key(ebno):
    return f"{float(ebno):.1f}"


def search_summary_row(record):
    row = {
        "rank": record.get("rank", ""),
        "mode": record["mode"],
        "stage": record["stage"],
        "candidate_id": record["candidate_id"],
        "score": record["score"],
        "ber_score": record["ber_score"],
        "fer_score": record["fer_score"],
        "syndrome_penalty": record["syndrome_penalty"],
        "evaluation_trials": record["evaluation_trials"],
        "evaluation_seeds": record["evaluation_seeds"],
        "EbNo_dB_values": record["EbNo_dB_values"],
        "BP_BER_by_EbNo": record["BP_BER_by_EbNo"],
        "candidate_BER_by_EbNo": record["candidate_BER_by_EbNo"],
        "BP_FER_by_EbNo": record["BP_FER_by_EbNo"],
        "candidate_FER_by_EbNo": record["candidate_FER_by_EbNo"],
        "syndrome_violation_by_EbNo": record["syndrome_violation_by_EbNo"],
    }
    for field in PARAM_FIELDS:
        row[field] = record.get(field, "")
    for item in record.get("by_ebno", []):
        key = ebno_key(item["EbNo_dB"])
        row[f"BER_EbNo_{key}"] = item["candidate_BER"]
        row[f"FER_EbNo_{key}"] = item["candidate_FER"]
        row[f"syndrome_violation_EbNo_{key}"] = item["syndrome_violation_rate"]
    return row


def write_search_summary(records, path):
    rows = [search_summary_row(row) for row in records]
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    write_csv(rows, path, fields)


def make_by_ebno_rows(reference_rows, best_by_mode):
    rows = []
    for ref in reference_rows:
        rows.append({
            "mode": "BP",
            "EbNo_dB": ref["EbNo_dB"],
            "BER": ref["BP_BER"],
            "FER": ref["BP_FER"],
            "score": 0.0,
            "syndrome_violation": 0.0,
        })
    for mode, record in best_by_mode.items():
        for item in sorted(record["by_ebno"], key=lambda row: row["EbNo_dB"]):
            rows.append({
                "mode": mode,
                "EbNo_dB": item["EbNo_dB"],
                "BER": item["candidate_BER"],
                "FER": item["candidate_FER"],
                "score": record["score"],
                "syndrome_violation": item["syndrome_violation_rate"],
            })
    return rows


def make_hyperparameter_rows(best_by_mode):
    rows = []
    for mode, record in best_by_mode.items():
        row = {
            "mode": mode,
            "score": record["score"],
            "ber_score": record["ber_score"],
            "fer_score": record["fer_score"],
            "syndrome_penalty": record["syndrome_penalty"],
            "evaluation_trials": record["evaluation_trials"],
        }
        for field in PARAM_FIELDS:
            row[field] = record.get(field, "")
        rows.append(row)
    return rows


def plot_comparison(reference_rows, best_by_mode, path):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ebno = np.array([row["EbNo_dB"] for row in reference_rows], dtype=float)
    bp_ber = np.array([row["BP_BER"] for row in reference_rows], dtype=float)
    bp_fer = np.array([row["BP_FER"] for row in reference_rows], dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2), dpi=160)
    axes[0].plot(ebno, bp_ber, marker="x", linewidth=1.6, label="BP")
    axes[1].plot(ebno, bp_fer, marker="x", linewidth=1.6, label="BP")

    for mode, record in best_by_mode.items():
        by_ebno = sorted(record["by_ebno"], key=lambda row: row["EbNo_dB"])
        x = np.array([row["EbNo_dB"] for row in by_ebno], dtype=float)
        axes[0].plot(x, [row["candidate_BER"] for row in by_ebno], marker="o", label=mode)
        axes[1].plot(x, [row["candidate_FER"] for row in by_ebno], marker="o", label=mode)
        axes[2].plot(x, [row["syndrome_violation_rate"] for row in by_ebno], marker="o", label=mode)

    axes[0].set_yscale("log")
    axes[0].set_title("BER")
    axes[0].set_xlabel("Eb/N0 [dB]")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].set_title("FER")
    axes[1].set_xlabel("Eb/N0 [dB]")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)

    axes[2].set_title("Syndrome violation")
    axes[2].set_xlabel("Eb/N0 [dB]")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path




def parse_lambda_values(value):
    return [float(item.strip()) for item in str(value).split(",") if item.strip()]


def fallback_sweep_params(mode):
    if mode == "lambda_pSA":
        return clamp_params({
            "kw": 2.066701297881316,
            "kr": 2.46793710784665,
            "alpha": 0.6623839086960397,
            "alpha_mode": "llr",
            "nrnd": 0.0,
            "psa_p": 0.17279330093725856,
            "lambda_mem": 0.8523968297288208,
            "lambda_out": 0.0,
            "I0_min": 0.05137239019146209,
            "I0_max": 0.8726181754134458,
            "I0_schedule_type": "exponential",
            "I0_schedule_shape": 0.9843129223328104,
            "I0_hold_fraction": 0.08301009657337094,
            "decision_method": "best_state",
            "burn_in": 5821,
            "sample_window": 3770,
            "n_cycles": 9600,
        }, mode)
    if mode == "lambda_SSA":
        return clamp_params({
            "kw": 0.5170080347282644,
            "kr": 0.7505686756355289,
            "alpha": 0.21414779981963764,
            "alpha_mode": "llr",
            "nrnd": 0.9572438770413181,
            "psa_p": 0.0,
            "lambda_mem": 0.6654687083888319,
            "lambda_out": 0.0,
            "I0_min": 1.4624127898995685,
            "I0_max": 5.234758127859346,
            "I0_schedule_type": "linear",
            "I0_schedule_shape": 1.2262115343373272,
            "I0_hold_fraction": 0.3608887846443949,
            "decision_method": "best_state",
            "burn_in": 3822,
            "sample_window": 1253,
            "n_cycles": 6400,
        }, mode)
    if mode == "tau_pSA":
        return clamp_params({
            "kw": 2.0,
            "kr": 2.5,
            "alpha": 0.6,
            "alpha_mode": "llr",
            "nrnd": 0.0,
            "psa_p": 0.2,
            "lambda_mem": 0.0,
            "response_lambda": 0.7,
            "lambda_out": 0.0,
            "I0_min": 0.05,
            "I0_max": 1.0,
            "I0_schedule_type": "linear",
            "I0_schedule_shape": 1.0,
            "I0_hold_fraction": 0.1,
            "decision_method": "best_state",
            "burn_in": 800,
            "sample_window": 2400,
            "n_cycles": 3200,
        }, mode)
    raise ValueError(f"Unsupported sweep mode: {mode}")


def parse_best_param_row(row):
    params = {}
    for field in PARAM_FIELDS:
        value = row.get(field, "")
        if value == "":
            continue
        if field in {"alpha_mode", "I0_schedule_type", "decision_method"}:
            params[field] = value
        elif field in {"burn_in", "sample_window", "n_cycles"}:
            params[field] = int(float(value))
        else:
            params[field] = float(value)
    return params


def score_value(row):
    try:
        score = float(row.get("score", ""))
    except (TypeError, ValueError):
        return math.inf
    return score if math.isfinite(score) else math.inf


def select_best_summary_row(rows):
    scored_rows = [row for row in rows if score_value(row) < math.inf]
    if not scored_rows:
        return None
    for stage in ("final", "refine"):
        stage_rows = [
            row for row in scored_rows
            if str(row.get("stage", "")).strip().lower() == stage
        ]
        if stage_rows:
            return min(stage_rows, key=score_value)
    return min(scored_rows, key=score_value)


def print_loaded_best_params(mode, info, params):
    print(f"Loaded best params for {mode}:", flush=True)
    print(f"  source = {info['source']}", flush=True)
    print(f"  stage  = {info['source_stage']}", flush=True)
    print(f"  score  = {info['source_score']}", flush=True)
    for field in ("lambda_mem", "response_lambda", "kw", "kr", "alpha", "alpha_mode", "n_cycles"):
        print(f"  {field} = {params.get(field, '')}", flush=True)


def load_best_params_from_summary(mode, args):
    # Prefer the deep-search summaries in results/.  If they do not exist,
    # fall back to the hard-coded best parameters obtained in the previous run.
    candidates = []
    base = Path(args.results_dir)
    candidates += [Path(path) for path in getattr(args, "sweep_param_csv", []) or []]
    if mode == "lambda_pSA":
        candidates += [
            base / "lambda_psa_48x96_deep_search_summary.csv",
            base / "lambda_psa_48x96_deep" / "lambda_psa_48x96_deep_search_summary.csv",
            base / "lambda_psa_48x96_deep" / "search_summary.csv",
            Path("results/lambda_psa_48x96_deep_search_summary.csv"),
        ]
        candidates += sorted(base.rglob("*lambda_psa*deep*summary*.csv"))
    elif mode == "tau_pSA":
        candidates += [
            base / "tau_psa_48x96_deep_search_summary.csv",
            base / "tau_psa_48x96_deep" / "tau_psa_48x96_deep_search_summary.csv",
            base / "tau_psa_48x96_deep" / "search_summary.csv",
            Path("results/tau_psa_48x96_deep_search_summary.csv"),
        ]
        candidates += sorted(base.rglob("*tau_psa*deep*summary*.csv"))
    elif mode == "lambda_SSA":
        candidates += [
            base / "lambda_ssa_48x96_deep_search_summary.csv",
            base / "lambda_ssa_48x96_deep" / "lambda_ssa_48x96_deep_search_summary.csv",
            base / "lambda_ssa_48x96_deep" / "search_summary.csv",
            Path("results/lambda_ssa_48x96_deep_search_summary.csv"),
        ]
        candidates += sorted(base.rglob("*lambda_ssa*deep*summary*.csv"))

    seen = set()
    for path in candidates:
        path = Path(path)
        path_key = str(path)
        if path_key in seen:
            continue
        seen.add(path_key)
        if not path.exists():
            continue
        rows = read_csv_rows(path)
        mode_rows = [row for row in rows if row.get("mode") == mode]
        if not mode_rows:
            continue
        row = select_best_summary_row(mode_rows)
        if row is None:
            continue
        params = clamp_params(parse_best_param_row(row), mode)
        info = {
            "source": str(path),
            "source_stage": str(row.get("stage", "")).strip(),
            "source_score": score_value(row),
        }
        print_loaded_best_params(mode, info, params)
        return params, info

    params = fallback_sweep_params(mode)
    info = {
        "source": "fallback",
        "source_stage": "fallback",
        "source_score": "",
    }
    print_loaded_best_params(mode, info, params)
    return params, info


def sweep_output_paths(args):
    base = Path(args.results_dir) / args.prefix
    return {
        "bp_reference": base / f"{args.prefix}_bp_reference.csv",
        "summary": base / f"{args.prefix}_summary.csv",
        "by_ebno": base / f"{args.prefix}_by_ebno.csv",
        "params": base / f"{args.prefix}_fixed_params.csv",
        "plot": base / f"{args.prefix}_plot.png",
    }


def plot_lambda_sweep(summary_rows, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    modes = []
    for row in summary_rows:
        if row["mode"] not in modes:
            modes.append(row["mode"])
    x_field = "response_lambda" if modes and all(mode == "tau_pSA" for mode in modes) else "lambda_mem"
    x_label = "response_lambda" if x_field == "response_lambda" else "lambda_mem"

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2), dpi=160)
    for mode in modes:
        rows = [row for row in summary_rows if row["mode"] == mode]
        rows.sort(key=lambda row: float(row[x_field]))
        x = np.array([float(row[x_field]) for row in rows], dtype=float)
        axes[0].plot(x, [float(row["avg_BER_ratio_to_BP"]) for row in rows], marker="o", label=mode)
        axes[1].plot(x, [float(row["avg_FER_ratio_to_BP"]) for row in rows], marker="o", label=mode)
        axes[2].plot(x, [float(row["avg_syndrome_violation"]) for row in rows], marker="o", label=mode)

    axes[0].set_title("BER ratio to BP")
    axes[0].set_xlabel(x_label)
    axes[0].set_ylabel("candidate BER / BP BER")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].set_title("FER ratio to BP")
    axes[1].set_xlabel(x_label)
    axes[1].set_ylabel("candidate FER / BP FER")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)

    axes[2].set_title("Syndrome violation")
    axes[2].set_xlabel(x_label)
    axes[2].set_ylabel("average violation rate")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def print_sanity_check(mode, record):
    print("Sanity check:", flush=True)
    print(f"  mode={mode}", flush=True)
    print(f"  lambda_mem={record['lambda_mem']}", flush=True)
    print(f"  response_lambda={record.get('response_lambda', '')}", flush=True)
    print(f"  BER_by_EbNo={record['candidate_BER_by_EbNo']}", flush=True)
    print(f"  FER_by_EbNo={record['candidate_FER_by_EbNo']}", flush=True)
    print(f"  syndrome={record['syndrome_violation_by_EbNo']}", flush=True)


def run_lambda_mem_sweep(P, reference_rows, args):
    lambda_values = args.lambda_values
    out = sweep_output_paths(args)
    summary_rows = []
    by_ebno_rows = []
    fixed_rows = []

    for mode in args.modes:
        if mode not in {"lambda_pSA", "lambda_SSA", "tau_pSA"}:
            raise ValueError("sweep supports only lambda_pSA, lambda_SSA, and tau_pSA")
        sweep_field = "response_lambda" if mode == "tau_pSA" else "lambda_mem"
        base_params, source_info = load_best_params_from_summary(mode, args)
        fixed_row = {
            "mode": mode,
            "source": source_info["source"],
            "source_stage": source_info["source_stage"],
            "source_score": source_info["source_score"],
        }
        for field in PARAM_FIELDS:
            fixed_row[field] = base_params.get(field, "")
        fixed_rows.append(fixed_row)
        print(f"\n=== {mode} {sweep_field} sweep ===", flush=True)
        print(f"fixed parameter source: {source_info['source']}", flush=True)
        sanity_seeds = [
            mode_seed(args, mode, idx + 1, 60_000 + idx * 103)
            for idx in range(args.final_seed_count)
        ]
        sanity_record = evaluate_record(
            P=P,
            reference_rows=reference_rows,
            mode=mode,
            params=base_params,
            total_trials=args.final_trials,
            seeds=sanity_seeds,
            args=args,
            stage="lambda_sweep_sanity",
            candidate_id=0,
            n_workers=args.n_workers,
        )
        print_sanity_check(mode, sanity_record)

        max_value = 0.95 if mode == "lambda_pSA" else 0.99 if mode == "tau_pSA" else 1.0
        params_list = []
        for value in lambda_values:
            sweep_value = clamp(value, 0.0, max_value)
            params = dict(base_params)
            params[sweep_field] = sweep_value
            params = clamp_params(params, mode)
            params_list.append(params)

        records = evaluate_params_list(
            P=P,
            reference_rows=reference_rows,
            mode=mode,
            params_list=params_list,
            args=args,
            stage="tau_sweep" if mode == "tau_pSA" else "lambda_sweep",
            total_trials=args.final_trials,
            stage_offset=70_000,
        )

        # evaluate_params_list() returns records ranked by score.
        # Therefore, do not zip requested sweep values with records.
        # Use the sweep value stored in each evaluated record.
        records = sorted(records, key=lambda row: float(row[sweep_field]))

        for record in records:
            lm = float(record["lambda_mem"])
            response_lambda = float(record.get("response_lambda", 0.0))
            sweep_value = float(record[sweep_field])
            by_ebno = sorted(record["by_ebno"], key=lambda row: row["EbNo_dB"])
            ber_ratios = []
            fer_ratios = []
            syndromes = []
            row = {
                "mode": mode,
                "lambda_mem": lm,
                "response_lambda": response_lambda,
                "score": record["score"],
                "ber_score": record["ber_score"],
                "fer_score": record["fer_score"],
                "syndrome_penalty": record["syndrome_penalty"],
                "evaluation_trials": record["evaluation_trials"],
                "evaluation_seeds": record["evaluation_seeds"],
            }
            for item, ref in zip(by_ebno, reference_rows):
                key = ebno_key(item["EbNo_dB"])
                cand_ber = item["candidate_BER"]
                cand_fer = item["candidate_FER"]
                bp_ber = ref["BP_BER"]
                bp_fer = ref["BP_FER"]
                syndrome_v = item["syndrome_violation_rate"]
                row[f"BER_EbNo_{key}"] = cand_ber
                row[f"FER_EbNo_{key}"] = cand_fer
                row[f"syndrome_violation_EbNo_{key}"] = syndrome_v
                row[f"BER_ratio_EbNo_{key}"] = cand_ber / max(bp_ber, args.eps)
                row[f"FER_ratio_EbNo_{key}"] = cand_fer / max(bp_fer, args.eps)
                ber_ratios.append(row[f"BER_ratio_EbNo_{key}"])
                fer_ratios.append(row[f"FER_ratio_EbNo_{key}"])
                syndromes.append(syndrome_v)
                by_ebno_rows.append({
                    "mode": mode,
                    "lambda_mem": lm,
                    "response_lambda": response_lambda,
                    "EbNo_dB": item["EbNo_dB"],
                    "BP_BER": bp_ber,
                    "BER": cand_ber,
                    "BER_ratio_to_BP": row[f"BER_ratio_EbNo_{key}"],
                    "BP_FER": bp_fer,
                    "FER": cand_fer,
                    "FER_ratio_to_BP": row[f"FER_ratio_EbNo_{key}"],
                    "syndrome_violation": syndrome_v,
                })
            row["avg_BER_ratio_to_BP"] = float(np.mean(ber_ratios))
            row["avg_FER_ratio_to_BP"] = float(np.mean(fer_ratios))
            row["avg_syndrome_violation"] = float(np.mean(syndromes))
            summary_rows.append(row)
            print(
                f"{mode} {sweep_field}={sweep_value:.3f}: score={record['score']:.5g}, "
                f"avg BER ratio={row['avg_BER_ratio_to_BP']:.4g}, "
                f"avg FER ratio={row['avg_FER_ratio_to_BP']:.4g}",
                flush=True,
            )

    write_csv(fixed_rows, out["params"], FIXED_PARAM_FIELDS)
    write_csv(summary_rows, out["summary"])
    write_csv(by_ebno_rows, out["by_ebno"])
    plot_lambda_sweep(summary_rows, out["plot"])
    print("\nWrote:")
    for key, path in out.items():
        if key != "bp_reference":
            print(f"  {key}: {path}")
    return summary_rows

def paths(args):
    base = Path(args.results_dir)
    return {
        "bp_reference": base / f"{args.prefix}_bp_reference.csv",
        "search_summary": base / f"{args.prefix}_search_summary.csv",
        "best_json": base / f"{args.prefix}_best.json",
        "by_ebno": base / f"{args.prefix}_by_ebno.csv",
        "comparison_png": base / f"{args.prefix}_comparison.png",
        "hyperparameters": base / f"{args.prefix}_hyperparameters.csv",
    }


def run_mode(P, reference_rows, mode, args):
    print(f"\n=== {mode} search ===", flush=True)
    if args.search_algorithm == "optuna":
        try:
            search_records = run_optuna_search(P, reference_rows, mode, args)
        except ImportError:
            print("Optuna is not installed; falling back to random search.", flush=True)
            search_records = run_random_search(P, reference_rows, mode, args)
    else:
        search_records = run_random_search(P, reference_rows, mode, args)

    refined = refine_top(P, reference_rows, mode, search_records, args)
    best_for_final = (refined or search_records)[0]
    final = final_eval(P, reference_rows, mode, best_for_final, args)
    all_records = rank_records(search_records + refined + [final])
    print(
        f"{mode} final: score={final['score']:.5g}, "
        f"BER={final['candidate_BER_by_EbNo']}",
        flush=True,
    )
    return all_records, final


def apply_preset_defaults(args):
    cfg = default_config(args.preset)
    for key, value in cfg.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    if args.prefix is None:
        args.prefix = cfg["prefix"]
    if args.modes is None:
        if args.preset in {"lambda_mem_sweep48x96", "lambda_mem_sweep48x96_fine"}:
            args.modes = ["lambda_pSA", "lambda_SSA"]
        elif args.preset == "dual24x48":
            args.modes = DUAL_MEMORY_MODES
        else:
            args.modes = DEFAULT_MODES
    return args


def parse_args():
    parser = argparse.ArgumentParser(description="Compare pSA/SSA lambda memory variants against BP.")
    parser.add_argument("--preset", choices=["24x48", "48x96", "lambda_psa48x96_deep", "lambda_ssa48x96_deep", "lambda_mem_sweep48x96", "lambda_mem_sweep48x96_fine", "dual24x48", "smoke"], default="24x48")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--prefix", default=None)
    parser.add_argument("--p-matrix", choices=["toy", "random_regular"], default=None)
    parser.add_argument("--n-bits", type=int, default=None)
    parser.add_argument("--n-checks", type=int, default=None)
    parser.add_argument("--variable-degree", type=int, default=None)
    parser.add_argument("--check-degree", type=int, default=None)
    parser.add_argument("--matrix-seed", type=int, default=None)
    parser.add_argument("--ebno-values", type=parse_float_list, default=None)
    parser.add_argument("--n-cycle-choices", type=parse_int_list, default=None)
    parser.add_argument("--n-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--search-algorithm", choices=["optuna", "random"], default="optuna")
    parser.add_argument("--bp-trials", type=int, default=None)
    parser.add_argument("--search-candidates", type=int, default=None)
    parser.add_argument("--search-trials", type=int, default=None)
    parser.add_argument("--refine-top-k", type=int, default=None)
    parser.add_argument("--refine-trials", type=int, default=None)
    parser.add_argument("--refine-seed-count", type=int, default=None)
    parser.add_argument("--final-trials", type=int, default=None)
    parser.add_argument("--final-seed-count", type=int, default=None)
    parser.add_argument("--eps", type=float, default=1e-12)
    parser.add_argument("--fer-weight", type=float, default=0.2)
    parser.add_argument("--syndrome-weight", type=float, default=0.5)
    parser.add_argument("--fixed-bit-width", type=int, default=8)
    parser.add_argument("--channel-input-mode", choices=["float", "fixed"], default="float")
    parser.add_argument("--bp-max-iter", type=int, default=50)
    parser.add_argument("--bp-llr-clip", type=float, default=50.0)
    parser.add_argument("--modes", type=lambda value: [item.strip() for item in value.split(",") if item.strip()], default=None)
    parser.add_argument("--lambda-values", type=parse_lambda_values, default=None)
    parser.add_argument("--sweep-param-csv", action="append", default=[])
    return apply_preset_defaults(parser.parse_args())


def main():
    args = parse_args()
    invalid_modes = [mode for mode in args.modes if mode not in MODES]
    if invalid_modes:
        raise ValueError(f"Unknown mode(s): {invalid_modes}")

    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    P = make_matrix(args)

    if args.preset in {"lambda_mem_sweep48x96", "lambda_mem_sweep48x96_fine"}:
        out = sweep_output_paths(args)
        reference_rows = load_or_create_reference(args, P, out["bp_reference"])
        run_lambda_mem_sweep(P, reference_rows, args)
        return

    out = paths(args)
    reference_rows = load_or_create_reference(args, P, out["bp_reference"])

    all_records = []
    best_by_mode = {}

    print("BP reference:")
    for row in reference_rows:
        print(f"  Eb/N0={row['EbNo_dB']}: BER={row['BP_BER']:.6g}, FER={row['BP_FER']:.6g}")

    for mode in args.modes:
        records, best = run_mode(P, reference_rows, mode, args)
        all_records.extend(records)
        best_by_mode[mode] = best
        write_search_summary(all_records, out["search_summary"])

    by_ebno_rows = make_by_ebno_rows(reference_rows, best_by_mode)
    hyper_rows = make_hyperparameter_rows(best_by_mode)
    write_search_summary(all_records, out["search_summary"])
    write_csv(by_ebno_rows, out["by_ebno"])
    write_csv(hyper_rows, out["hyperparameters"])
    plot_comparison(reference_rows, best_by_mode, out["comparison_png"])

    best_json = {
        "code": {
            "p_matrix": args.p_matrix,
            "n_bits": args.n_bits,
            "n_checks": args.n_checks,
            "variable_degree": args.variable_degree,
            "check_degree": args.check_degree,
            "matrix_seed": args.matrix_seed,
        },
        "objective": {
            "definition": "mean abs log10 BER gap + fer_weight * mean abs log10 FER gap + syndrome_weight * mean syndrome violation",
            "eps": args.eps,
            "fer_weight": args.fer_weight,
            "syndrome_weight": args.syndrome_weight,
        },
        "ebno_values": args.ebno_values,
        "n_cycle_choices": args.n_cycle_choices,
        "best_by_mode": best_by_mode,
        "outputs": {key: str(value) for key, value in out.items()},
    }
    out["best_json"].write_text(json.dumps(json_ready(best_json), indent=2))

    print("\nSaved:")
    for key, value in out.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()

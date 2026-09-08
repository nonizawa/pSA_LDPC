#!/usr/bin/env python3
"""Integrated paper experiments for pSA, lambda_pSA, and tau_pSA.

This script is intentionally a thin paper-facing layer.  LDPC evaluations reuse
the existing LDPC_pbit implementation, while MAX-CUT and 2-SAT use small local
Jacobi/parallel pSA evaluators so the original COP source remains unchanged.
"""

from __future__ import print_function

import argparse
import csv
import json
import math
import os
import sys

if sys.version_info[0] < 3:
    here = os.path.dirname(os.path.abspath(__file__))
    arm64 = "/usr/bin/arch"
    m3max_python = os.path.join(here, "LDPC_pbit", "m3max", "bin", "python")
    if os.path.exists(arm64) and os.path.exists(m3max_python):
        try:
            os.execv(arm64, [arm64, "-arm64", m3max_python] + sys.argv)
        except OSError:
            pass
    for python3 in ("python3",):
        try:
            os.execvp(python3, [python3] + sys.argv)
        except OSError:
            pass
    sys.stderr.write("run_paper_experiments.py requires Python 3.\n")
    sys.exit(1)

from concurrent.futures import ProcessPoolExecutor, as_completed
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
PAPER_ROOT = REPO_ROOT / "paper_results"
PAPER_MODES = ["pSA", "lambda_pSA", "tau_pSA"]
EBNO_VALUES = [2.0, 2.5, 3.0]

_MAXCUT_WORKER_ARGS = None
_MAXCUT_WORKER_SIZE = None
_MAXCUT_WORKER_INSTANCES = None
_TWOSAT_WORKER_ARGS = None
_TWOSAT_WORKER_SIZE = None
_TWOSAT_WORKER_INSTANCES = None
_TWOSAT_PHASE_WORKER_ARGS = None
_TWOSAT_PHASE_WORKER_SIZE = None
_TWOSAT_PHASE_WORKER_INSTANCES = None

LDPC_SIZES = {
    "N96_M48": {"n_bits": 96, "n_checks": 48},
    "N192_M96": {"n_bits": 192, "n_checks": 96},
    "N288_M144": {"n_bits": 288, "n_checks": 144},
}

MAXCUT_SIZES = {
    "N100": {"n_nodes": 100},
    "N200": {"n_nodes": 200},
    "N500": {"n_nodes": 500},
    "N1000": {"n_nodes": 1000},
}

MAXCUT_STRUCTURE_SIZES = {
    "N500_ERp0p03": {"n_nodes": 500, "graph_type": "erdos_renyi", "edge_probability": 0.03},
    "N500_ERp0p04": {"n_nodes": 500, "graph_type": "erdos_renyi", "edge_probability": 0.04},
    "N500_ERp0p05": {"n_nodes": 500, "graph_type": "erdos_renyi", "edge_probability": 0.05},
    "N500_ERp0p06": {"n_nodes": 500, "graph_type": "erdos_renyi", "edge_probability": 0.06},
    "N500_ERp0p07": {"n_nodes": 500, "graph_type": "erdos_renyi", "edge_probability": 0.07},
    "N500_ERp0p08": {"n_nodes": 500, "graph_type": "erdos_renyi", "edge_probability": 0.08},
    "N500_ERp0p09": {"n_nodes": 500, "graph_type": "erdos_renyi", "edge_probability": 0.09},
    "N500_ERp0p10": {"n_nodes": 500, "graph_type": "erdos_renyi", "edge_probability": 0.10},
    "N1000_ERp0p01": {"n_nodes": 1000, "graph_type": "erdos_renyi", "edge_probability": 0.01},
    "N1000_ERp0p015": {"n_nodes": 1000, "graph_type": "erdos_renyi", "edge_probability": 0.015},
    "N1000_ERp0p02": {"n_nodes": 1000, "graph_type": "erdos_renyi", "edge_probability": 0.02},
    "N1000_ERp0p025": {"n_nodes": 1000, "graph_type": "erdos_renyi", "edge_probability": 0.025},
    "N1000_ERp0p03": {"n_nodes": 1000, "graph_type": "erdos_renyi", "edge_probability": 0.03},
    "N500_BA_m2": {"n_nodes": 500, "graph_type": "barabasi_albert", "ba_m": 2},
    "N500_BA_m3": {"n_nodes": 500, "graph_type": "barabasi_albert", "ba_m": 3},
    "N500_BA_m4": {"n_nodes": 500, "graph_type": "barabasi_albert", "ba_m": 4},
    "N500_WS_k4_p0p10": {"n_nodes": 500, "graph_type": "watts_strogatz", "ws_k": 4, "ws_rewire_p": 0.10},
    "N500_WS_k6_p0p10": {"n_nodes": 500, "graph_type": "watts_strogatz", "ws_k": 6, "ws_rewire_p": 0.10},
    "N500_WS_k8_p0p10": {"n_nodes": 500, "graph_type": "watts_strogatz", "ws_k": 8, "ws_rewire_p": 0.10},
}

TWOSAT_SIZES = {
    "N100": {"n_vars": 100},
    "N200": {"n_vars": 200},
    "N500": {"n_vars": 500},
    "N1000": {"n_vars": 1000},
}

TWOSAT_PHASE_SIZES = {
    "N500_A0p80": {"n_vars": 500, "alpha": 0.80},
    "N500_A0p90": {"n_vars": 500, "alpha": 0.90},
    "N500_A0p95": {"n_vars": 500, "alpha": 0.95},
    "N500_A0p98": {"n_vars": 500, "alpha": 0.98},
    "N500_A1p00": {"n_vars": 500, "alpha": 1.00},
    "N500_A1p02": {"n_vars": 500, "alpha": 1.02},
    "N500_A1p05": {"n_vars": 500, "alpha": 1.05},
    "N500_A1p10": {"n_vars": 500, "alpha": 1.10},
    "N500_A1p20": {"n_vars": 500, "alpha": 1.20},
    "N1000_A0p90": {"n_vars": 1000, "alpha": 0.90},
    "N1000_A0p95": {"n_vars": 1000, "alpha": 0.95},
    "N1000_A0p98": {"n_vars": 1000, "alpha": 0.98},
    "N1000_A1p00": {"n_vars": 1000, "alpha": 1.00},
    "N1000_A1p02": {"n_vars": 1000, "alpha": 1.02},
    "N1000_A1p05": {"n_vars": 1000, "alpha": 1.05},
    "N1000_A1p10": {"n_vars": 1000, "alpha": 1.10},
}

TWOSAT_PHASE_REPRO_SIZES = {
    "N500_A0p95": TWOSAT_PHASE_SIZES["N500_A0p95"],
    "N500_A0p98": TWOSAT_PHASE_SIZES["N500_A0p98"],
    "N500_A1p00": TWOSAT_PHASE_SIZES["N500_A1p00"],
    "N500_A1p02": TWOSAT_PHASE_SIZES["N500_A1p02"],
    "N500_A1p05": TWOSAT_PHASE_SIZES["N500_A1p05"],
    "N500_A1p10": TWOSAT_PHASE_SIZES["N500_A1p10"],
    "N500_A1p20": TWOSAT_PHASE_SIZES["N500_A1p20"],
}

COMMON_COLUMNS = [
    "problem",
    "size",
    "size_label",
    "mode",
    "stage",
    "candidate_id",
    "rank",
    "graph_type",
    "graph_param",
    "edge_probability",
    "ba_m",
    "ws_k",
    "ws_rewire_p",
    "graph_seed",
    "instance_seed",
    "N",
    "M",
    "EbNo_dB",
    "kw",
    "kr",
    "alpha",
    "alpha_mode",
    "psa_p",
    "lambda_mem",
    "response_lambda",
    "I0_min",
    "I0_max",
    "I0_schedule_type",
    "I0_schedule_shape",
    "I0_hold_fraction",
    "decision_method",
    "burn_in",
    "sample_window",
    "n_cycles",
    "BER",
    "FER",
    "syndrome_violation",
    "BP_BER",
    "BP_FER",
    "BER_ratio_to_BP",
    "FER_ratio_to_BP",
    "best_cut",
    "normalized_cut",
    "mean_cut",
    "mean_normalized_cut",
    "best_normalized_cut",
    "approximation_ratio",
    "best_unsat",
    "mean_unsat",
    "mean_unsat_rate",
    "mean_unsatisfied_clauses",
    "best_unsat_rate",
    "best_unsatisfied_clauses",
    "std_unsat",
    "valid_assignment_rate",
    "success_rate",
    "success_rate_all",
    "success_rate_sat_only",
    "unsat_instance_rate",
    "is_satisfiable",
    "mean_convergence_cycle",
    "median_convergence_cycle",
    "final_energy",
    "best_energy",
    "mean_energy",
    "time_to_first_solution",
    "n_trials",
    "score",
    "seed",
    "best_mode",
    "best_lambda_mem",
    "best_response_lambda",
    "improvement_over_pSA",
]

PARAM_COLUMNS = [
    "kw",
    "kr",
    "alpha",
    "alpha_mode",
    "psa_p",
    "lambda_mem",
    "response_lambda",
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


def parse_csv_list(value     , item_type=str)             :
    return [item_type(item.strip()) for item in str(value).split(",") if item.strip()]


def parse_size_list(value            , choices                           )             :
    if value is None or value == "all":
        return list(choices)
    sizes = parse_csv_list(value, str)
    missing = [size for size in sizes if size not in choices]
    if missing:
        raise ValueError("Unknown size(s): {}; choices are {}".format(missing, list(choices)))
    return sizes


def decimal_grid(start       , stop       , step       )               :
    start_d = Decimal(str(start))
    stop_d = Decimal(str(stop))
    step_d = Decimal(str(step))
    out              = []
    value = start_d
    while value <= stop_d + Decimal("1e-12"):
        out.append(float(value))
        value += step_d
    return out


def memory_grid(best_value       , max_value       , smoke      )               :
    if smoke:
        values = [0.0, float(best_value), min(0.95, max_value)]
    else:
        coarse = decimal_grid(0.0, 0.95, 0.05)
        low = max(0.0, float(best_value) - 0.15)
        high = min(max_value, float(best_value) + 0.15)
        fine = decimal_grid(round(low, 2), round(high, 2), 0.02)
        values = coarse + fine
    return sorted({round(min(max(value, 0.0), max_value), 10) for value in values})


def maxcut_structure_tau_grid(best_value       , smoke      )               :
    max_value = 0.99
    if smoke:
        values = [0.0, float(best_value), 0.5]
    else:
        coarse = decimal_grid(0.0, 0.95, 0.05)
        focused = decimal_grid(0.0, 0.50, 0.02)
        low = max(0.0, float(best_value) - 0.10)
        high = min(max_value, float(best_value) + 0.10)
        fine = decimal_grid(round(low, 2), round(high, 2), 0.01)
        values = coarse + focused + fine
    return sorted({round(min(max(value, 0.0), max_value), 10) for value in values})


def label_float(value       )      :
    value = float(value)
    if abs(value - round(value)) < 1e-12:
        text = str(int(round(value)))
    elif abs(value - round(value, 2)) < 1e-12:
        text = "{:.2f}".format(value)
    elif abs(value - round(value, 3)) < 1e-12:
        text = "{:.3f}".format(value)
    else:
        text = "{:.6g}".format(value)
    return text.replace("-", "m").replace(".", "p")


def label_alpha(value       )      :
    return "{:.2f}".format(float(value)).replace("-", "m").replace(".", "p")


def sanitize_label(text     )      :
    cleaned = []
    for ch in str(text):
        if ch.isalnum() or ch in {"_", "-"}:
            cleaned.append(ch)
        else:
            cleaned.append("_")
    return "".join(cleaned).strip("_")


def maxcut_output_label(args                    , size     )        :
    graph_types = parse_csv_list(args.graph_types, str)
    parts = []
    for graph_type in graph_types:
        if graph_type == "random_regular":
            parts.append("RR{}".format(int(args.regular_degree)))
        elif graph_type == "erdos_renyi":
            if args.edge_prob is None:
                parts.append("ERdeg{}".format(label_float(float(args.regular_degree))))
            else:
                parts.append("ERp{}".format(label_float(float(args.edge_prob))))
        else:
            parts.append(sanitize_label(graph_type))
    suffix = "_".join(parts)
    return sanitize_label("{}_{}".format(size, suffix)) if suffix else sanitize_label(size)


def twosat_output_label(args                    , size     )        :
    return sanitize_label("{}_CD{}".format(size, label_float(float(args.clause_density))))


def twosat_phase_output_label(size     )        :
    spec = TWOSAT_PHASE_SIZES[size]
    return sanitize_label("N{}_A{}".format(int(spec["n_vars"]), label_alpha(float(spec["alpha"]))))


def ensure_tree(root      )        :
    for problem in ("ldpc", "maxcut", "maxcut_structure", "2sat", "2sat_phase", "2sat_phase_repro"):
        (root / problem).mkdir(parents=True, exist_ok=True)
    for leaf in ["figures", "tables", "summaries"]:
        (root / leaf).mkdir(parents=True, exist_ok=True)


def row_template(problem     , size     , mode     )                  :
    row = {key: "" for key in COMMON_COLUMNS}
    row.update({"problem": problem, "size": size, "size_label": size, "mode": mode})
    return row


def write_csv(rows                      , path      , fieldnames                   = None)        :
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(COMMON_COLUMNS)
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def read_csv(path      )                        :
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def safe_float(value     , default        = math.nan)         :
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def safe_mean(values             )         :
    clean = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(clean)) if clean else math.nan


def metric_is_better(problem     , candidate       , baseline       )        :
    if not math.isfinite(candidate) or not math.isfinite(baseline):
        return False
    if problem == "ldpc":
        return candidate < baseline
    if problem == "2sat":
        return candidate < baseline
    return candidate > baseline


def format_json(path      , data                )        :
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True))


# ---------------------------------------------------------------------------
# LDPC integration
# ---------------------------------------------------------------------------


def load_ldpc_compare():
    ldpc_dir = str(REPO_ROOT / "LDPC_pbit")
    if ldpc_dir not in sys.path:
        sys.path.insert(0, ldpc_dir)
    import compare_lambda_memory as ldpc_compare  # type: ignore

    return ldpc_compare


def ldpc_cycle_choices(n_bits     , smoke_cycles             = None)             :
    if smoke_cycles is not None:
        return [max(1, int(smoke_cycles))]
    if n_bits <= 96:
        return [9600, 12800, 19200, 25600]
    if n_bits <= 192:
        return [6400, 9600, 12800, 19200]
    return [9600, 12800, 19200, 25600]


def ldpc_args_for(args                    , size     )                   :
    spec = LDPC_SIZES[size]
    smoke = args.mode == "smoke"
    trials = int(args.n_trials)
    return SimpleNamespace(
        preset="paper",
        results_dir=str(Path(args.output_dir) / "ldpc" / size / "comparison"),
        prefix="{}_paper".format(size),
        p_matrix="random_regular",
        n_bits=int(spec["n_bits"]),
        n_checks=int(spec["n_checks"]),
        variable_degree=3,
        check_degree=6,
        matrix_seed=int(args.matrix_seed),
        ebno_values=parse_csv_list(args.ebno_values, float),
        n_cycle_choices=ldpc_cycle_choices(int(spec["n_bits"]), args.n_cycles if smoke else None),
        n_workers=1 if smoke else int(args.n_workers),
        seed=int(args.seed),
        resume=bool(args.resume),
        search_algorithm="random" if smoke else args.search_algorithm,
        bp_trials=trials if smoke else int(args.bp_trials),
        search_candidates=max(1, int(args.search_candidates if args.search_candidates is not None else (2 if smoke else 256))),
        search_trials=trials if smoke else int(args.search_trials),
        refine_top_k=max(1, int(args.refine_top_k if args.refine_top_k is not None else (1 if smoke else 12))),
        refine_trials=trials if smoke else int(args.refine_trials),
        refine_seed_count=1 if smoke else int(args.refine_seed_count),
        final_trials=trials if smoke else int(args.final_trials),
        final_seed_count=1 if smoke else int(args.final_seed_count),
        eps=float(args.eps),
        fer_weight=float(args.fer_weight),
        syndrome_weight=float(args.syndrome_weight),
        fixed_bit_width=int(args.fixed_bit_width),
        channel_input_mode=str(args.channel_input_mode),
        bp_max_iter=int(args.bp_max_iter),
        bp_llr_clip=float(args.bp_llr_clip),
        modes=list(PAPER_MODES),
        lambda_values=[],
        sweep_param_csv=[],
    )


def ldpc_make_matrix(ldpc_compare     , ldpc_args                 )              :
    return ldpc_compare.random_regular_ldpc_parity_check(
        n_bits=ldpc_args.n_bits,
        n_checks=ldpc_args.n_checks,
        variable_degree=ldpc_args.variable_degree,
        check_degree=ldpc_args.check_degree,
        seed=ldpc_args.matrix_seed,
    )


def ldpc_param_row(record                , problem     , size     , mode     )                  :
    row = row_template(problem, size, mode)
    for field in PARAM_COLUMNS:
        row[field] = record.get(field, "")
    row.update(
        {
            "stage": record.get("stage", ""),
            "candidate_id": record.get("candidate_id", ""),
            "rank": record.get("rank", ""),
            "score": record.get("score", ""),
            "seed": record.get("evaluation_seeds", ""),
        }
    )
    by_ebno = list(record.get("by_ebno", []) or [])
    if by_ebno:
        bers = [safe_float(item.get("candidate_BER")) for item in by_ebno]
        fers = [safe_float(item.get("candidate_FER")) for item in by_ebno]
        synd = [safe_float(item.get("syndrome_violation_rate")) for item in by_ebno]
        bp_bers = [safe_float(item.get("BP_BER")) for item in by_ebno]
        bp_fers = [safe_float(item.get("BP_FER")) for item in by_ebno]
        row["BER"] = safe_mean(bers)
        row["FER"] = safe_mean(fers)
        row["syndrome_violation"] = safe_mean(synd)
        row["BP_BER"] = safe_mean(bp_bers)
        row["BP_FER"] = safe_mean(bp_fers)
        row["BER_ratio_to_BP"] = safe_float(row["BER"]) / max(safe_float(row["BP_BER"]), 1e-12)
        row["FER_ratio_to_BP"] = safe_float(row["FER"]) / max(safe_float(row["BP_FER"]), 1e-12)
    return row


def ldpc_by_ebno_rows(record                , problem     , size     , mode     )                        :
    rows                       = []
    for item in sorted(record.get("by_ebno", []) or [], key=lambda row: safe_float(row.get("EbNo_dB"), 0.0)):
        row = row_template(problem, size, mode)
        for field in PARAM_COLUMNS:
            row[field] = record.get(field, "")
        bp_ber = safe_float(item.get("BP_BER"))
        bp_fer = safe_float(item.get("BP_FER"))
        ber = safe_float(item.get("candidate_BER"))
        fer = safe_float(item.get("candidate_FER"))
        row.update(
            {
                "stage": record.get("stage", ""),
                "candidate_id": record.get("candidate_id", ""),
                "rank": record.get("rank", ""),
                "EbNo_dB": item.get("EbNo_dB", ""),
                "BER": ber,
                "FER": fer,
                "syndrome_violation": item.get("syndrome_violation_rate", ""),
                "BP_BER": bp_ber,
                "BP_FER": bp_fer,
                "BER_ratio_to_BP": ber / max(bp_ber, 1e-12),
                "FER_ratio_to_BP": fer / max(bp_fer, 1e-12),
                "score": record.get("score", ""),
                "seed": record.get("evaluation_seeds", ""),
            }
        )
        rows.append(row)
    return rows


def ldpc_bp_rows(reference_rows                      , size     )                        :
    rows = []
    for ref in sorted(reference_rows, key=lambda row: safe_float(row.get("EbNo_dB"), 0.0)):
        row = row_template("ldpc", size, "BP")
        row.update(
            {
                "stage": "bp_baseline",
                "EbNo_dB": ref.get("EbNo_dB", ""),
                "BER": ref.get("BP_BER", ""),
                "FER": ref.get("BP_FER", ""),
                "syndrome_violation": 0.0,
                "BP_BER": ref.get("BP_BER", ""),
                "BP_FER": ref.get("BP_FER", ""),
                "BER_ratio_to_BP": 1.0,
                "FER_ratio_to_BP": 1.0,
                "score": 0.0,
            }
        )
        rows.append(row)
    return rows


def write_ldpc_deep_outputs(root      , size     , mode     , records                      , best                )        :
    mode_dir = root / "ldpc" / size / mode
    write_csv([ldpc_param_row(record, "ldpc", size, mode) for record in records], mode_dir / "deep_search_summary.csv")
    by_rows                       = []
    for record in records:
        by_rows.extend(ldpc_by_ebno_rows(record, "ldpc", size, mode))
    write_csv(by_rows, mode_dir / "deep_search_by_ebno.csv")
    write_csv([ldpc_param_row(best, "ldpc", size, mode)], mode_dir / "best_params.csv")


def write_ldpc_memory_outputs(
    root      ,
    size     ,
    mode     ,
    records                      ,
    best_record                ,
)        :
    mode_dir = root / "ldpc" / size / mode
    write_csv([ldpc_param_row(record, "ldpc", size, mode) for record in records], mode_dir / "memory_sweep_summary.csv")
    by_rows                       = []
    for record in records:
        by_rows.extend(ldpc_by_ebno_rows(record, "ldpc", size, mode))
    write_csv(by_rows, mode_dir / "memory_sweep_by_ebno.csv")
    write_csv(ldpc_by_ebno_rows(best_record, "ldpc", size, mode), mode_dir / "final_comparison.csv")


def check_ldpc_tau_zero_matches_psa(ldpc_compare     , P            )                  :
    from ldpc_pbit import update_pbits  # type: ignore

    params = {
        "kw": 1.2,
        "kr": 1.5,
        "n_cycles": 5,
        "I0_min": 0.1,
        "I0_max": 0.5,
        "psa_p": 0.0,
        "I0_schedule_type": "linear",
        "I0_schedule_shape": 1.0,
        "I0_hold_fraction": 0.0,
        "decision_method": "last",
        "burn_in": 0,
        "sample_window": 0,
    }
    channel_values = np.linspace(-0.5, 0.5, P.shape[1])
    out_psa = update_pbits(
        P=P,
        channel_values=channel_values,
        mode="pSA",
        rng=np.random.default_rng(12345),
        **params
    )
    out_tau = update_pbits(
        P=P,
        channel_values=channel_values,
        mode="tau_pSA",
        response_lambda=0.0,
        rng=np.random.default_rng(12345),
        **params
    )
    return {
        "jacobi_parallel_update": True,
        "parity_sum_from_w_prev": True,
        "H_eff_from_Itanh_hold_prev": True,
        "response_lambda_zero_matches_pSA": bool(np.array_equal(out_psa[1], out_tau[1])),
    }


def run_ldpc_size(args                    , root      , size     )        :
    ldpc_compare = load_ldpc_compare()
    ldpc_args = ldpc_args_for(args, size)
    P = ldpc_make_matrix(ldpc_compare, ldpc_args)
    comparison_dir = root / "ldpc" / size / "comparison"
    reference_path = comparison_dir / "bp_baseline.csv"
    reference_rows = ldpc_compare.load_or_create_reference(ldpc_args, P, reference_path)

    best_by_mode                            = {}
    for mode in PAPER_MODES:
        print("[LDPC {}] deep search: {}".format(size, mode), flush=True)
        mode_args = SimpleNamespace(**vars(ldpc_args))
        mode_args.modes = [mode]
        records, best = ldpc_compare.run_mode(P, reference_rows, mode, mode_args)
        write_ldpc_deep_outputs(root, size, mode, records, best)
        best_by_mode[mode] = best

    final_by_mode                            = {}
    for mode in PAPER_MODES:
        if mode == "pSA":
            records = [best_by_mode[mode]]
            final_by_mode[mode] = best_by_mode[mode]
            write_ldpc_memory_outputs(root, size, mode, records, best_by_mode[mode])
            continue

        sweep_field = "response_lambda" if mode == "tau_pSA" else "lambda_mem"
        max_value = 0.99 if mode == "tau_pSA" else 0.95
        best_value = safe_float(best_by_mode[mode].get(sweep_field), 0.0)
        values = memory_grid(best_value, max_value=max_value, smoke=args.mode == "smoke")
        params_list = []
        for value in values:
            params = {field: best_by_mode[mode].get(field, "") for field in ldpc_compare.PARAM_FIELDS}
            params[sweep_field] = value
            params = ldpc_compare.clamp_params(params, mode)
            params_list.append(params)

        print("[LDPC {}] memory sweep: {} ({})".format(size, mode, sweep_field), flush=True)
        sweep_args = SimpleNamespace(**vars(ldpc_args))
        records = ldpc_compare.evaluate_params_list(
            P=P,
            reference_rows=reference_rows,
            mode=mode,
            params_list=params_list,
            args=sweep_args,
            stage="tau_sweep" if mode == "tau_pSA" else "lambda_sweep",
            total_trials=sweep_args.final_trials,
            stage_offset=70000,
        )
        # evaluate_params_list() ranks by score, so sort by the stored sweep field.
        records = sorted(records, key=lambda row: float(row[sweep_field]))
        best_record = min(records, key=lambda row: (float(row["score"]), float(row["ber_score"])))
        final_by_mode[mode] = best_record
        write_ldpc_memory_outputs(root, size, mode, records, best_record)

    comparison_rows = ldpc_bp_rows(reference_rows, size)
    for mode in PAPER_MODES:
        comparison_rows.extend(ldpc_by_ebno_rows(final_by_mode[mode], "ldpc", size, mode))
    write_csv(comparison_rows, comparison_dir / "final_comparison.csv")

    tau_check = check_ldpc_tau_zero_matches_psa(ldpc_compare, P)
    format_json(comparison_dir / "tau_pSA_checks.json", tau_check)


# ---------------------------------------------------------------------------
# MAX-CUT paper evaluator
# ---------------------------------------------------------------------------


def make_schedule(
    start       ,
    stop       ,
    n_cycles     ,
    schedule_type     ,
    shape       ,
    hold_fraction       ,
)              :
    n_cycles = max(1, int(n_cycles))
    if n_cycles == 1:
        return np.array([stop], dtype=float)
    shape = max(1e-9, float(shape))
    hold_fraction = min(max(float(hold_fraction), 0.0), 0.95)
    t = np.linspace(0.0, 1.0, n_cycles)
    if schedule_type == "constant":
        return np.full(n_cycles, start, dtype=float)
    if schedule_type == "exponential":
        progress = (np.exp(shape * t) - 1.0) / max(np.exp(shape) - 1.0, 1e-12)
    elif schedule_type == "cosine":
        progress = ((1.0 - np.cos(np.pi * t)) / 2.0) ** shape
    elif schedule_type == "piecewise":
        progress = np.zeros_like(t)
        active = t > hold_fraction
        if np.any(active):
            local_t = (t[active] - hold_fraction) / max(1.0 - hold_fraction, 1e-12)
            progress[active] = local_t**shape
    else:
        progress = t**shape
    return start + (stop - start) * progress


def maxcut_value(spins            , adj            )         :
    diff = spins[:, None] != spins[None, :]
    return float(np.sum(adj * diff) / 2.0)


def random_regular_adj(n_nodes     , degree     , seed     )              :
    if degree < 2:
        raise ValueError("degree must be at least 2")
    if degree != 3:
        raise ValueError("local fallback supports degree=3")
    rng = np.random.default_rng(seed)
    adj = np.zeros((n_nodes, n_nodes), dtype=float)
    for node in range(n_nodes):
        nxt = (node + 1) % n_nodes
        adj[node, nxt] = 1.0
        adj[nxt, node] = 1.0
    existing = {tuple(sorted((node, (node + 1) % n_nodes))) for node in range(n_nodes)}
    for _ in range(2000):
        perm = rng.permutation(n_nodes)
        pairs = [(int(perm[idx]), int(perm[idx + 1])) for idx in range(0, n_nodes, 2)]
        if all(a != b and tuple(sorted((a, b))) not in existing for a, b in pairs):
            for a, b in pairs:
                adj[a, b] = 1.0
                adj[b, a] = 1.0
            return adj
    raise RuntimeError("failed to generate random 3-regular graph")


def erdos_renyi_adj(n_nodes     , edge_prob       , seed     )              :
    rng = np.random.default_rng(seed)
    upper = rng.random((n_nodes, n_nodes)) < float(edge_prob)
    upper = np.triu(upper, 1)
    adj = upper.astype(float)
    return adj + adj.T


def _networkx_to_adj(graph     , n_nodes     )              :
    adj = np.zeros((n_nodes, n_nodes), dtype=float)
    for u, v in graph.edges():
        adj[int(u), int(v)] = 1.0
        adj[int(v), int(u)] = 1.0
    return adj


def barabasi_albert_adj(n_nodes     , m     , seed     )              :
    n_nodes = int(n_nodes)
    m = int(m)
    if m < 1 or m >= n_nodes:
        raise ValueError("BA m must satisfy 1 <= m < n_nodes")
    try:
        import networkx as nx  # type: ignore

        return _networkx_to_adj(nx.barabasi_albert_graph(n_nodes, m, seed=int(seed)), n_nodes)
    except ImportError:
        rng = np.random.default_rng(seed)
        adj = np.zeros((n_nodes, n_nodes), dtype=float)
        degrees = np.zeros(n_nodes, dtype=float)
        for i in range(m + 1):
            for j in range(i + 1, m + 1):
                adj[i, j] = 1.0
                adj[j, i] = 1.0
                degrees[i] += 1.0
                degrees[j] += 1.0
        for node in range(m + 1, n_nodes):
            probs = degrees[:node].copy()
            if np.sum(probs) <= 0:
                probs[:] = 1.0
            probs = probs / np.sum(probs)
            targets = rng.choice(node, size=m, replace=False, p=probs)
            for target in targets:
                adj[node, int(target)] = 1.0
                adj[int(target), node] = 1.0
                degrees[node] += 1.0
                degrees[int(target)] += 1.0
        return adj


def watts_strogatz_adj(n_nodes     , k     , rewire_p       , seed     )              :
    n_nodes = int(n_nodes)
    k = int(k)
    if k < 2 or k >= n_nodes or k % 2 != 0:
        raise ValueError("WS k must be even and satisfy 2 <= k < n_nodes")
    try:
        import networkx as nx  # type: ignore

        return _networkx_to_adj(nx.watts_strogatz_graph(n_nodes, k, float(rewire_p), seed=int(seed)), n_nodes)
    except ImportError:
        rng = np.random.default_rng(seed)
        adj = np.zeros((n_nodes, n_nodes), dtype=float)
        half = k // 2
        for node in range(n_nodes):
            for offset in range(1, half + 1):
                target = (node + offset) % n_nodes
                adj[node, target] = 1.0
                adj[target, node] = 1.0
        for node in range(n_nodes):
            for offset in range(1, half + 1):
                old = (node + offset) % n_nodes
                if node < old or old < node and (old + offset) % n_nodes == node:
                    if rng.random() >= float(rewire_p):
                        continue
                    adj[node, old] = 0.0
                    adj[old, node] = 0.0
                    candidates = np.flatnonzero((adj[node] == 0.0) & (np.arange(n_nodes) != node))
                    if candidates.size == 0:
                        adj[node, old] = 1.0
                        adj[old, node] = 1.0
                        continue
                    new = int(rng.choice(candidates))
                    adj[node, new] = 1.0
                    adj[new, node] = 1.0
        return adj


def maxcut_graph_param(spec                )      :
    graph_type = spec.get("graph_type", "")
    if graph_type == "erdos_renyi":
        return "p={:.6g}".format(float(spec.get("edge_probability", 0.0)))
    if graph_type == "barabasi_albert":
        return "m={}".format(int(spec.get("ba_m", 0)))
    if graph_type == "watts_strogatz":
        return "k={},p={:.6g}".format(int(spec.get("ws_k", 0)), float(spec.get("ws_rewire_p", 0.0)))
    if graph_type == "random_regular":
        return "degree={}".format(int(spec.get("regular_degree", 0)))
    return str(graph_type)


def build_maxcut_instance(graph_type     , n_nodes     , seed     , spec                )                  :
    if graph_type == "random_regular":
        adj = random_regular_adj(n_nodes, int(spec.get("regular_degree", 3)), seed)
    elif graph_type == "erdos_renyi":
        p = float(spec.get("edge_probability", 0.0))
        adj = erdos_renyi_adj(n_nodes, p, seed)
    elif graph_type == "barabasi_albert":
        adj = barabasi_albert_adj(n_nodes, int(spec.get("ba_m", 2)), seed)
    elif graph_type == "watts_strogatz":
        adj = watts_strogatz_adj(n_nodes, int(spec.get("ws_k", 4)), float(spec.get("ws_rewire_p", 0.10)), seed)
    else:
        raise ValueError("Unknown graph type: {}".format(graph_type))
    meta = {
        "graph_type": graph_type,
        "graph_param": maxcut_graph_param(spec),
        "edge_probability": spec.get("edge_probability", ""),
        "ba_m": spec.get("ba_m", ""),
        "ws_k": spec.get("ws_k", ""),
        "ws_rewire_p": spec.get("ws_rewire_p", ""),
        "graph_seed": seed,
        "adj": adj,
    }
    return meta


def maxcut_instances(args                    , size     )                        :
    n_nodes = int(MAXCUT_SIZES[size]["n_nodes"])
    graph_types = parse_csv_list(args.graph_types, str)
    instance_seeds = [0] if args.mode == "smoke" else parse_csv_list(args.instance_seeds, int)
    instances                       = []
    for graph_type in graph_types:
        for instance_seed in instance_seeds:
            seed = int(args.seed) + 1009 * int(instance_seed) + n_nodes
            if graph_type == "random_regular":
                spec = {"graph_type": graph_type, "regular_degree": int(args.regular_degree)}
            elif graph_type == "erdos_renyi":
                p = float(args.edge_prob) if args.edge_prob is not None else float(args.regular_degree) / max(n_nodes - 1, 1)
                spec = {"graph_type": graph_type, "edge_probability": p}
            else:
                raise ValueError("Unknown graph type: {}".format(graph_type))
            instance = build_maxcut_instance(graph_type, n_nodes, seed, spec)
            instance["instance_seed"] = instance_seed
            instances.append(instance)
    return instances


def maxcut_structure_instances(args                    , size     )                        :
    spec = dict(MAXCUT_STRUCTURE_SIZES[size])
    n_nodes = int(spec["n_nodes"])
    graph_type = str(spec["graph_type"])
    instance_seeds = [0] if args.mode == "smoke" else parse_csv_list(args.instance_seeds, int)
    instances                       = []
    for instance_seed in instance_seeds:
        seed = int(args.seed) + 1009 * int(instance_seed) + n_nodes
        instance = build_maxcut_instance(graph_type, n_nodes, seed, spec)
        instance["instance_seed"] = instance_seed
        instances.append(instance)
    return instances


def maxcut_default_params(mode     , n_cycles     )                  :
    params = {
        "kw": 1.0,
        "kr": 0.0,
        "alpha": 1.0,
        "alpha_mode": "fixed",
        "psa_p": 0.0,
        "lambda_mem": 0.0,
        "response_lambda": 0.0,
        "I0_min": 0.1,
        "I0_max": 3.0,
        "I0_schedule_type": "linear",
        "I0_schedule_shape": 1.0,
        "I0_hold_fraction": 0.0,
        "decision_method": "best_state",
        "burn_in": 0,
        "sample_window": 0,
        "n_cycles": int(n_cycles),
    }
    if mode == "lambda_pSA":
        params["lambda_mem"] = 0.3
    if mode == "tau_pSA":
        params["response_lambda"] = 0.5
    return params


def clamp_maxcut_params(params                , mode     , kw_max        = 8.0)                  :
    n_cycles = max(1, int(safe_float(params.get("n_cycles"), 100)))
    kw_max = max(0.05, safe_float(kw_max, 8.0))
    i0_min = min(max(safe_float(params.get("I0_min"), 0.1), 0.03), 3.0)
    i0_max = min(max(safe_float(params.get("I0_max"), 3.0), i0_min), 12.0)
    out = {
        "kw": min(max(safe_float(params.get("kw"), 1.0), 0.05), kw_max),
        "kr": 0.0,
        "alpha": 1.0,
        "alpha_mode": "fixed",
        "psa_p": min(max(safe_float(params.get("psa_p"), 0.0), 0.0), 1.0),
        "lambda_mem": 0.0,
        "response_lambda": 0.0,
        "I0_min": i0_min,
        "I0_max": i0_max,
        "I0_schedule_type": params.get("I0_schedule_type", "linear")
        if params.get("I0_schedule_type", "linear") in {"linear", "constant", "exponential", "cosine", "piecewise"}
        else "linear",
        "I0_schedule_shape": min(max(safe_float(params.get("I0_schedule_shape"), 1.0), 0.2), 10.0),
        "I0_hold_fraction": min(max(safe_float(params.get("I0_hold_fraction"), 0.0), 0.0), 0.9),
        "decision_method": params.get("decision_method", "best_state")
        if params.get("decision_method", "best_state") in {"last", "best_state"}
        else "best_state",
        "burn_in": min(max(int(safe_float(params.get("burn_in"), 0)), 0), n_cycles - 1),
        "sample_window": min(max(int(safe_float(params.get("sample_window"), 0)), 0), n_cycles),
        "n_cycles": n_cycles,
    }
    if mode == "lambda_pSA":
        out["lambda_mem"] = min(max(safe_float(params.get("lambda_mem"), 0.0), 0.0), 0.95)
    if mode == "tau_pSA":
        out["response_lambda"] = min(max(safe_float(params.get("response_lambda"), 0.0), 0.0), 0.99)
    out["sample_window"] = min(out["sample_window"], out["n_cycles"] - out["burn_in"])
    return out


def random_maxcut_params(rng                     , mode     , n_cycle_choices           , kw_max        = 8.0)                  :
    n_cycles = int(rng.choice(n_cycle_choices))
    burn_in = int(rng.integers(0, max(1, n_cycles)))
    max_window = max(0, n_cycles - burn_in)
    kw_max = max(0.1, safe_float(kw_max, 8.0))
    i0_min = float(np.exp(rng.uniform(np.log(0.03), np.log(2.0))))
    params = {
        "kw": float(np.exp(rng.uniform(np.log(0.1), np.log(kw_max)))),
        "psa_p": float(rng.uniform(0.0, 0.8 if mode == "tau_pSA" else 1.0)),
        "lambda_mem": 0.0,
        "response_lambda": 0.0,
        "I0_min": i0_min,
        "I0_max": float(np.exp(rng.uniform(np.log(i0_min), np.log(10.0)))),
        "I0_schedule_type": str(rng.choice(["linear", "constant", "exponential", "cosine", "piecewise"])),
        "I0_schedule_shape": float(np.exp(rng.uniform(np.log(0.2), np.log(10.0)))),
        "I0_hold_fraction": float(rng.uniform(0.0, 0.9)),
        "decision_method": str(rng.choice(["last", "best_state"])),
        "burn_in": burn_in,
        "sample_window": int(rng.integers(0, max_window + 1)),
        "n_cycles": n_cycles,
    }
    if mode == "lambda_pSA":
        params["lambda_mem"] = float(rng.uniform(0.0, 0.95))
    if mode == "tau_pSA":
        params["response_lambda"] = float(rng.uniform(0.0, 0.99))
    return clamp_maxcut_params(params, mode, kw_max=kw_max)


def maxcut_single_trial(
    adj            ,
    mode     ,
    params                ,
    rng                     ,
)                  :
    n_nodes = adj.shape[0]
    n_cycles = int(params["n_cycles"])
    spins = rng.choice(np.array([-1, 1], dtype=int), size=n_nodes)
    hold_state = np.zeros(n_nodes, dtype=float)
    best_spins = spins.copy()
    best_cut = maxcut_value(spins, adj)
    best_cycle = 0
    schedule = make_schedule(
        params["I0_min"],
        params["I0_max"],
        n_cycles,
        params["I0_schedule_type"],
        params["I0_schedule_shape"],
        params["I0_hold_fraction"],
    )
    for cycle in range(n_cycles):
        spins_prev = spins.copy()
        hold_prev = hold_state.copy()
        local_field = adj.dot(spins_prev)
        I_vec = -float(params["kw"]) * local_field
        H_new = np.tanh(float(schedule[cycle]) * I_vec)
        rnd = rng.uniform(-1.0, 1.0, size=n_nodes)

        if mode == "pSA":
            H_eff = H_new
        elif mode == "lambda_pSA":
            H_eff = float(params["lambda_mem"]) * hold_prev + H_new
        elif mode == "tau_pSA":
            H_eff = float(params["response_lambda"]) * hold_prev + (1.0 - float(params["response_lambda"])) * H_new
        else:
            raise ValueError("Unknown MAX-CUT mode: {}".format(mode))

        candidate = np.where(H_eff + rnd >= 0.0, 1, -1).astype(int)
        if float(params["psa_p"]) > 0.0:
            hold_mask = rng.random(n_nodes) < float(params["psa_p"])
            candidate[hold_mask] = spins_prev[hold_mask]
            if mode == "tau_pSA":
                H_eff[hold_mask] = hold_prev[hold_mask]
            elif mode in {"pSA", "lambda_pSA"}:
                H_new[hold_mask] = hold_prev[hold_mask]
        spins = candidate
        hold_state = H_eff if mode == "tau_pSA" else H_new

        cut = maxcut_value(spins, adj)
        if cut > best_cut:
            best_cut = cut
            best_spins = spins.copy()
            best_cycle = cycle + 1

    if params["decision_method"] == "last":
        final_spins = spins
        final_cut = maxcut_value(final_spins, adj)
        convergence_cycle = n_cycles
    else:
        final_spins = best_spins
        final_cut = best_cut
        convergence_cycle = best_cycle

    return {"cut": float(final_cut), "convergence_cycle": int(convergence_cycle)}


def simulate_maxcut(
    adj            ,
    mode     ,
    params                ,
    n_trials     ,
    seed     ,
    reference_cut               = None,
)                  :
    rng = np.random.default_rng(seed)
    trials = [maxcut_single_trial(adj, mode, params, rng) for _ in range(max(1, int(n_trials)))]
    cuts = np.array([trial["cut"] for trial in trials], dtype=float)
    convergence = np.array([trial["convergence_cycle"] for trial in trials], dtype=float)
    edges = max(float(np.sum(adj) / 2.0), 1.0)
    ref = float(reference_cut) if reference_cut is not None else float(np.max(cuts))
    success = cuts >= ref - 1e-12 if ref > 0 else cuts == np.max(cuts)
    best_cut = float(np.max(cuts))
    mean_cut = float(np.mean(cuts))
    return {
        "N": int(adj.shape[0]),
        "best_cut": best_cut,
        "mean_cut": mean_cut,
        "normalized_cut": float(best_cut / edges),
        "best_normalized_cut": float(best_cut / edges),
        "mean_normalized_cut": float(mean_cut / edges),
        "approximation_ratio": float(best_cut / ref) if ref > 0 else 1.0,
        "success_rate": float(np.mean(success)),
        "convergence_cycle": float(np.mean(convergence)),
        "mean_convergence_cycle": float(np.mean(convergence)),
        "median_convergence_cycle": float(np.median(convergence)),
        "reference_cut": ref,
        "n_edges": edges,
        "mean_energy": float(-mean_cut),
        "best_energy": float(-best_cut),
    }


def maxcut_record_row(
    size     ,
    mode     ,
    params                ,
    result                ,
    stage     ,
    candidate_id     ,
    rank     ,
    seed     ,
    graph_type      = "",
    instance_seed            = "",
    score               = None,
    graph_param      = "",
    edge_probability            = "",
    ba_m            = "",
    ws_k            = "",
    ws_rewire_p            = "",
    graph_seed            = "",
    problem_label      = "maxcut",
)                  :
    row = row_template(problem_label, size, mode)
    for field in PARAM_COLUMNS:
        row[field] = params.get(field, "")
    row.update(
        {
            "stage": stage,
            "candidate_id": candidate_id,
            "rank": rank,
            "graph_type": graph_type,
            "graph_param": graph_param,
            "edge_probability": edge_probability,
            "ba_m": ba_m,
            "ws_k": ws_k,
            "ws_rewire_p": ws_rewire_p,
            "graph_seed": graph_seed,
            "instance_seed": instance_seed,
            "N": result.get("N", ""),
            "best_cut": result.get("best_cut", ""),
            "normalized_cut": result.get("normalized_cut", ""),
            "best_normalized_cut": result.get("best_normalized_cut", result.get("normalized_cut", "")),
            "approximation_ratio": result.get("approximation_ratio", ""),
            "success_rate": result.get("success_rate", ""),
            "score": result.get("score", score if score is not None else ""),
            "seed": seed,
            "mean_cut": result.get("mean_cut", ""),
            "mean_normalized_cut": result.get("mean_normalized_cut", ""),
            "convergence_cycle": result.get("convergence_cycle", ""),
            "mean_convergence_cycle": result.get("mean_convergence_cycle", result.get("convergence_cycle", "")),
            "median_convergence_cycle": result.get("median_convergence_cycle", ""),
            "reference_cut": result.get("reference_cut", ""),
            "n_edges": result.get("n_edges", ""),
            "mean_energy": result.get("mean_energy", ""),
            "best_energy": result.get("best_energy", ""),
        }
    )
    return row


def update_maxcut_references(rows                      )        :
    best_by_instance                                    = {}
    for row in rows:
        key = (str(row.get("size", "")), str(row.get("graph_type", "")), str(row.get("instance_seed", "")))
        best_by_instance[key] = max(best_by_instance.get(key, 0.0), safe_float(row.get("best_cut"), 0.0))
    for row in rows:
        key = (str(row.get("size", "")), str(row.get("graph_type", "")), str(row.get("instance_seed", "")))
        ref = best_by_instance.get(key, safe_float(row.get("best_cut"), 0.0))
        best_cut = safe_float(row.get("best_cut"), 0.0)
        row["reference_cut"] = ref
        row["approximation_ratio"] = best_cut / ref if ref > 0 else 1.0


def aggregate_maxcut_record(
    size     ,
    mode     ,
    params                ,
    instance_rows                      ,
    stage     ,
    candidate_id     ,
    rank     ,
    seed     ,
    success_weight        = 0.0,
    problem_label      = "maxcut",
)                  :
    normalized = [safe_float(row.get("normalized_cut")) for row in instance_rows]
    best_normalized = [safe_float(row.get("best_normalized_cut")) for row in instance_rows]
    mean_normalized = [safe_float(row.get("mean_normalized_cut")) for row in instance_rows]
    approx = [safe_float(row.get("approximation_ratio")) for row in instance_rows]
    success = [safe_float(row.get("success_rate")) for row in instance_rows]
    best_cut = [safe_float(row.get("best_cut")) for row in instance_rows]
    mean_cut = [safe_float(row.get("mean_cut")) for row in instance_rows]
    mean_energy = [safe_float(row.get("mean_energy")) for row in instance_rows]
    best_energy = [safe_float(row.get("best_energy")) for row in instance_rows]
    mean_convergence = [safe_float(row.get("mean_convergence_cycle")) for row in instance_rows]
    median_convergence = [safe_float(row.get("median_convergence_cycle")) for row in instance_rows]
    mean_quality = safe_mean(mean_normalized)
    success_mean = safe_mean(success)
    score = -mean_quality - float(success_weight) * success_mean
    first_row = instance_rows[0] if instance_rows else {}
    result = {
        "N": first_row.get("N", ""),
        "best_cut": safe_mean(best_cut),
        "mean_cut": safe_mean(mean_cut),
        "normalized_cut": safe_mean(normalized),
        "best_normalized_cut": safe_mean(best_normalized),
        "mean_normalized_cut": mean_quality,
        "approximation_ratio": safe_mean(approx),
        "success_rate": success_mean,
        "score": score,
        "mean_energy": safe_mean(mean_energy),
        "best_energy": safe_mean(best_energy),
        "mean_convergence_cycle": safe_mean(mean_convergence),
        "median_convergence_cycle": safe_mean(median_convergence),
    }
    return maxcut_record_row(
        size,
        mode,
        params,
        result,
        stage,
        candidate_id,
        rank,
        seed,
        score=score,
        graph_type=first_row.get("graph_type", ""),
        graph_param=first_row.get("graph_param", ""),
        edge_probability=first_row.get("edge_probability", ""),
        ba_m=first_row.get("ba_m", ""),
        ws_k=first_row.get("ws_k", ""),
        ws_rewire_p=first_row.get("ws_rewire_p", ""),
        graph_seed=first_row.get("graph_seed", ""),
        problem_label=problem_label,
    )


def evaluate_maxcut_candidate(
    args                    ,
    size     ,
    mode     ,
    params                ,
    instances                      ,
    stage     ,
    candidate_id     ,
    seed     ,
)                                               :
    rows = []
    problem_label = getattr(args, "_maxcut_problem_label", "maxcut")
    for instance in instances:
        result = simulate_maxcut(
            adj=instance["adj"],
            mode=mode,
            params=params,
            n_trials=int(args.n_trials),
            seed=seed + 7919 * (int(instance["instance_seed"]) + 1),
        )
        rows.append(
            maxcut_record_row(
                size=size,
                mode=mode,
                params=params,
                result=result,
                stage=stage,
                candidate_id=candidate_id,
                rank=0,
                seed=seed,
                graph_type=instance["graph_type"],
                instance_seed=instance["instance_seed"],
                graph_param=instance.get("graph_param", ""),
                edge_probability=instance.get("edge_probability", ""),
                ba_m=instance.get("ba_m", ""),
                ws_k=instance.get("ws_k", ""),
                ws_rewire_p=instance.get("ws_rewire_p", ""),
                graph_seed=instance.get("graph_seed", ""),
                problem_label=problem_label,
            )
        )
    update_maxcut_references(rows)
    summary = aggregate_maxcut_record(
        size,
        mode,
        params,
        rows,
        stage,
        candidate_id,
        0,
        seed,
        success_weight=safe_float(getattr(args, "maxcut_success_weight", 0.0), 0.0),
        problem_label=problem_label,
    )
    return summary, rows


def init_maxcut_worker(args                    , size     , instances                      )        :
    global _MAXCUT_WORKER_ARGS, _MAXCUT_WORKER_SIZE, _MAXCUT_WORKER_INSTANCES
    _MAXCUT_WORKER_ARGS = args
    _MAXCUT_WORKER_SIZE = size
    _MAXCUT_WORKER_INSTANCES = instances


def evaluate_maxcut_task(task                )                                               :
    if _MAXCUT_WORKER_ARGS is None or _MAXCUT_WORKER_INSTANCES is None:
        raise RuntimeError("MAX-CUT worker was not initialized")
    return evaluate_maxcut_candidate(
        _MAXCUT_WORKER_ARGS,
        _MAXCUT_WORKER_SIZE,
        task["mode"],
        task["params"],
        _MAXCUT_WORKER_INSTANCES,
        task["stage"],
        task["candidate_id"],
        task["seed"],
    )


def maxcut_worker_count(args                    , n_tasks     )        :
    if n_tasks <= 1 or args.mode == "smoke":
        return 1
    return min(max(1, int(getattr(args, "n_workers", 1))), n_tasks)


def maxcut_instance_sort_key(row                )        :
    return (
        int(safe_float(row.get("candidate_id"), 0.0)),
        str(row.get("graph_type", "")),
        int(safe_float(row.get("instance_seed"), 0.0)),
    )


def evaluate_maxcut_tasks(
    args                    ,
    size     ,
    instances                      ,
    tasks                      ,
    progress_label     ,
)                                               :
    n_tasks = len(tasks)
    n_workers = maxcut_worker_count(args, n_tasks)
    results = []
    if n_workers <= 1:
        for task in tasks:
            summary, rows = evaluate_maxcut_candidate(
                args,
                size,
                task["mode"],
                task["params"],
                instances,
                task["stage"],
                task["candidate_id"],
                task["seed"],
            )
            results.append((summary, rows))
            print(
                "{} {}/{}: score={:.4g}".format(
                    progress_label,
                    len(results),
                    n_tasks,
                    safe_float(summary.get("score"), math.inf),
                ),
                flush=True,
            )
        return results

    print("{} using {} workers".format(progress_label, n_workers), flush=True)
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=init_maxcut_worker,
        initargs=(args, size, instances),
    ) as executor:
        futures = {executor.submit(evaluate_maxcut_task, task): task for task in tasks}
        for future in as_completed(futures):
            summary, rows = future.result()
            results.append((summary, rows))
            print(
                "{} {}/{}: score={:.4g}".format(
                    progress_label,
                    len(results),
                    n_tasks,
                    safe_float(summary.get("score"), math.inf),
                ),
                flush=True,
            )
    return results


def run_maxcut_deep(
    args                    ,
    root      ,
    size     ,
    instances                      ,
)                             :
    maxcut_dir = getattr(args, "_maxcut_dir", "maxcut")
    n_cycles = int(args.n_cycles) if args.mode == "smoke" else int(args.maxcut_cycles)
    n_cycle_choices = [n_cycles] if args.mode == "smoke" else parse_csv_list(args.maxcut_cycle_choices, int)
    search_candidates = max(1, int(args.search_candidates if args.search_candidates is not None else (2 if args.mode == "smoke" else 128)))
    best_by_mode                            = {}
    all_instance_rows                       = []

    for mode in PAPER_MODES:
        rng = np.random.default_rng(int(args.seed) + 101003 * (PAPER_MODES.index(mode) + 1))
        summaries                       = []
        instance_rows                       = []
        candidates = [maxcut_default_params(mode, n_cycles)]
        candidates.extend(random_maxcut_params(rng, mode, n_cycle_choices) for _ in range(max(0, search_candidates - 1)))

        print("[MAX-CUT {}] deep search: {}".format(size, mode), flush=True)
        tasks = []
        for idx, params in enumerate(candidates, start=1):
            seed = int(args.seed) + 100000 * (PAPER_MODES.index(mode) + 1) + 7919 * idx
            tasks.append({
                "mode": mode,
                "params": params,
                "stage": "search",
                "candidate_id": idx,
                "seed": seed,
            })
        for summary, rows in evaluate_maxcut_tasks(
            args,
            size,
            instances,
            tasks,
            "[MAX-CUT {} {}] search".format(size, mode),
        ):
            summaries.append(summary)
            instance_rows.extend(rows)

        ranked = sorted(summaries, key=lambda row: safe_float(row.get("score"), math.inf))
        for rank, row in enumerate(ranked, start=1):
            row["rank"] = rank
        rank_by_candidate = {int(row["candidate_id"]): int(row["rank"]) for row in ranked}
        for row in instance_rows:
            row["rank"] = rank_by_candidate.get(int(row["candidate_id"]), "")
        instance_rows = sorted(instance_rows, key=maxcut_instance_sort_key)

        best = ranked[0]
        best_by_mode[mode] = best
        mode_dir = root / maxcut_dir / size / mode
        write_csv(ranked, mode_dir / "deep_search_summary.csv")
        write_csv(instance_rows, mode_dir / "deep_search_by_instance.csv")
        write_csv([best], mode_dir / "best_params.csv")
        all_instance_rows.extend(instance_rows)

    update_maxcut_references(all_instance_rows)
    return best_by_mode


def run_maxcut_sweeps(
    args                    ,
    root      ,
    size     ,
    instances                      ,
    best_by_mode                           ,
)                             :
    maxcut_dir = getattr(args, "_maxcut_dir", "maxcut")
    final_by_mode                            = {}
    all_sweep_rows                       = []

    for mode in PAPER_MODES:
        base_params = {field: best_by_mode[mode].get(field, "") for field in PARAM_COLUMNS}
        base_params = clamp_maxcut_params(base_params, mode)
        if mode == "pSA":
            values = [0.0]
            sweep_field = "lambda_mem"
            max_value = 0.0
        else:
            sweep_field = "response_lambda" if mode == "tau_pSA" else "lambda_mem"
            max_value = 0.99 if mode == "tau_pSA" else 0.95
            if mode == "tau_pSA" and bool(getattr(args, "_maxcut_structure", False)):
                values = maxcut_structure_tau_grid(safe_float(base_params.get(sweep_field), 0.0), args.mode == "smoke")
            else:
                values = memory_grid(safe_float(base_params.get(sweep_field), 0.0), max_value, args.mode == "smoke")

        summaries                       = []
        instance_rows                       = []
        print("[MAX-CUT {}] memory sweep: {}".format(size, mode), flush=True)
        tasks = []
        for idx, value in enumerate(values, start=1):
            params = dict(base_params)
            params[sweep_field] = min(max(float(value), 0.0), max_value)
            params = clamp_maxcut_params(params, mode)
            seed = int(args.seed) + 700000 + 100000 * (PAPER_MODES.index(mode) + 1) + 7919 * idx
            tasks.append({
                "mode": mode,
                "params": params,
                "stage": "memory_sweep",
                "candidate_id": idx,
                "seed": seed,
            })
        for summary, rows in evaluate_maxcut_tasks(
            args,
            size,
            instances,
            tasks,
            "[MAX-CUT {} {}] memory sweep".format(size, mode),
        ):
            summaries.append(summary)
            instance_rows.extend(rows)
        instance_rows = sorted(instance_rows, key=maxcut_instance_sort_key)

        # Sort by the memory value stored in the record, not by zip order.
        if mode == "lambda_pSA":
            summaries = sorted(summaries, key=lambda row: safe_float(row.get("lambda_mem"), 0.0))
        elif mode == "tau_pSA":
            summaries = sorted(summaries, key=lambda row: safe_float(row.get("response_lambda"), 0.0))

        ranked = sorted(summaries, key=lambda row: safe_float(row.get("score"), math.inf))
        best = ranked[0]
        final_by_mode[mode] = best
        mode_dir = root / maxcut_dir / size / mode
        write_csv(summaries, mode_dir / "memory_sweep_summary.csv")
        write_csv(instance_rows, mode_dir / "memory_sweep_by_instance.csv")
        write_csv([best], mode_dir / "final_comparison.csv")
        all_sweep_rows.extend(instance_rows)

    update_maxcut_references(all_sweep_rows)
    return final_by_mode


def check_maxcut_tau_zero_matches_psa(size     , instances                      , n_cycles     )                  :
    params = maxcut_default_params("pSA", n_cycles)
    params["psa_p"] = 0.0
    adj = instances[0]["adj"]
    psa = [maxcut_single_trial(adj, "pSA", params, np.random.default_rng(123))]
    tau_params = dict(params)
    tau_params["response_lambda"] = 0.0
    tau = [maxcut_single_trial(adj, "tau_pSA", tau_params, np.random.default_rng(123))]
    return {
        "jacobi_parallel_update": True,
        "local_field_from_spins_prev": True,
        "H_eff_from_hold_prev": True,
        "response_lambda_zero_matches_pSA": bool(psa[0]["cut"] == tau[0]["cut"]),
    }


def write_maxcut_final_by_instance(root      , maxcut_dir      , size     , final_by_mode                           )        :
    rows                       = []
    for mode in PAPER_MODES:
        path = root / maxcut_dir / size / mode / "memory_sweep_by_instance.csv"
        selected_candidate = str(final_by_mode[mode].get("candidate_id", ""))
        for row in read_csv(path):
            if row.get("mode") == mode and str(row.get("candidate_id", "")) == selected_candidate:
                rows.append(row)
    rows = sorted(rows, key=maxcut_instance_sort_key)
    update_maxcut_references(rows)
    write_csv(rows, root / maxcut_dir / size / "comparison" / "by_instance.csv")


def run_maxcut_size(args                    , root      , size     )        :
    if not hasattr(args, "_maxcut_dir"):
        args._maxcut_dir = "maxcut"
    if not hasattr(args, "_maxcut_problem_label"):
        args._maxcut_problem_label = "maxcut"
    output_size = maxcut_output_label(args, size)
    if output_size != size:
        print("[MAX-CUT {}] output label: {}".format(size, output_size), flush=True)
    instances = maxcut_instances(args, size)
    best_by_mode = run_maxcut_deep(args, root, output_size, instances)
    final_by_mode = run_maxcut_sweeps(args, root, output_size, instances, best_by_mode)
    comparison_rows = [final_by_mode[mode] for mode in PAPER_MODES]
    update_maxcut_references(comparison_rows)
    write_csv(comparison_rows, root / args._maxcut_dir / output_size / "comparison" / "final_comparison.csv")
    write_maxcut_final_by_instance(root, args._maxcut_dir, output_size, final_by_mode)
    checks = check_maxcut_tau_zero_matches_psa(output_size, instances, int(args.n_cycles if args.mode == "smoke" else args.maxcut_cycles))
    format_json(root / args._maxcut_dir / output_size / "comparison" / "tau_pSA_checks.json", checks)


def run_maxcut_structure_size(args                    , root      , size     )        :
    local_args = SimpleNamespace(**vars(args))
    local_args._maxcut_dir = "maxcut_structure"
    local_args._maxcut_problem_label = "maxcut_structure"
    local_args._maxcut_structure = True
    output_size = size
    instances = maxcut_structure_instances(local_args, size)
    best_by_mode = run_maxcut_deep(local_args, root, output_size, instances)
    final_by_mode = run_maxcut_sweeps(local_args, root, output_size, instances, best_by_mode)
    comparison_rows = [final_by_mode[mode] for mode in PAPER_MODES]
    update_maxcut_references(comparison_rows)
    write_csv(comparison_rows, root / "maxcut_structure" / output_size / "comparison" / "final_comparison.csv")
    write_maxcut_final_by_instance(root, "maxcut_structure", output_size, final_by_mode)
    checks = check_maxcut_tau_zero_matches_psa(output_size, instances, int(local_args.n_cycles if local_args.mode == "smoke" else local_args.maxcut_cycles))
    format_json(root / "maxcut_structure" / output_size / "comparison" / "tau_pSA_checks.json", checks)


# ---------------------------------------------------------------------------
# 2-SAT paper evaluator
# ---------------------------------------------------------------------------


def generate_planted_2sat(n_vars     , clause_density       , seed     = 0)                  :
    rng = np.random.default_rng(seed)
    n_vars = int(n_vars)
    n_clauses = max(1, int(round(float(clause_density) * n_vars)))
    planted = rng.choice(np.array([-1, 1], dtype=int), size=n_vars)
    clauses = []

    while len(clauses) < n_clauses:
        i, j = rng.choice(n_vars, size=2, replace=False)
        a = int(rng.choice(np.array([-1, 1], dtype=int)))
        b = int(rng.choice(np.array([-1, 1], dtype=int)))
        if a * planted[i] == -1 and b * planted[j] == -1:
            continue
        clauses.append((int(i), a, int(j), b))

    return {
        "n_vars": n_vars,
        "clauses": np.array(clauses, dtype=int),
        "planted_assignment": planted,
        "clause_density": float(clause_density),
        "n_clauses": int(n_clauses),
        "sat_planted": True,
    }


def generate_random_2sat(n_vars     , alpha       , seed     = 0)                  :
    rng = np.random.default_rng(seed)
    n_vars = int(n_vars)
    alpha = float(alpha)
    n_clauses = max(1, int(round(alpha * n_vars)))
    clauses = []

    for _ in range(n_clauses):
        i, j = rng.choice(n_vars, size=2, replace=False)
        a = int(rng.choice(np.array([-1, 1], dtype=int)))
        b = int(rng.choice(np.array([-1, 1], dtype=int)))
        clauses.append((int(i), a, int(j), b))

    clause_array = np.array(clauses, dtype=int)
    return {
        "n_vars": n_vars,
        "clauses": clause_array,
        "planted_assignment": None,
        "clause_density": alpha,
        "alpha": alpha,
        "n_clauses": int(n_clauses),
        "sat_planted": False,
        "is_satisfiable": twosat_is_satisfiable(clause_array, n_vars),
    }


def twosat_literal_node(var_index     , sign     )        :
    return 2 * int(var_index) + (0 if int(sign) == 1 else 1)


def twosat_is_satisfiable(clauses            , n_vars     )        :
    n_vars = int(n_vars)
    n_nodes = 2 * n_vars
    graph = [[] for _ in range(n_nodes)]
    reverse = [[] for _ in range(n_nodes)]

    def add_edge(src     , dst     )        :
        graph[src].append(dst)
        reverse[dst].append(src)

    for i, a, j, b in np.asarray(clauses, dtype=int):
        left = twosat_literal_node(i, a)
        right = twosat_literal_node(j, b)
        add_edge(left ^ 1, right)
        add_edge(right ^ 1, left)

    seen = [False] * n_nodes
    order = []
    for start in range(n_nodes):
        if seen[start]:
            continue
        stack = [(start, 0)]
        seen[start] = True
        while stack:
            node, idx = stack[-1]
            if idx < len(graph[node]):
                nxt = graph[node][idx]
                stack[-1] = (node, idx + 1)
                if not seen[nxt]:
                    seen[nxt] = True
                    stack.append((nxt, 0))
            else:
                order.append(node)
                stack.pop()

    comp = [-1] * n_nodes
    comp_id = 0
    for start in reversed(order):
        if comp[start] != -1:
            continue
        stack = [start]
        comp[start] = comp_id
        while stack:
            node = stack.pop()
            for nxt in reverse[node]:
                if comp[nxt] == -1:
                    comp[nxt] = comp_id
                    stack.append(nxt)
        comp_id += 1

    for var_index in range(n_vars):
        if comp[2 * var_index] == comp[2 * var_index + 1]:
            return False
    return True


def twosat_ising(clauses            , n_vars     )                                 :
    clauses = np.array(clauses, dtype=int)
    h = np.zeros(int(n_vars), dtype=float)
    J = np.zeros((int(n_vars), int(n_vars)), dtype=float)
    const = 0.0
    for i, a, j, b in clauses:
        i = int(i)
        j = int(j)
        a = int(a)
        b = int(b)
        const += 0.25
        h[i] += -0.25 * a
        h[j] += -0.25 * b
        coupling = 0.25 * a * b
        J[i, j] += coupling
        J[j, i] += coupling
    return h, J, const


def twosat_unsat_count(spins            , clauses            )         :
    spins = np.asarray(spins, dtype=int)
    clauses = np.asarray(clauses, dtype=int)
    if clauses.size == 0:
        return 0
    i = clauses[:, 0]
    a = clauses[:, 1]
    j = clauses[:, 2]
    b = clauses[:, 3]
    left_true = a * spins[i] == 1
    right_true = b * spins[j] == 1
    return int(np.count_nonzero(~(left_true | right_true)))


def twosat_energy(spins            , h            , J            , const       )         :
    spins = np.asarray(spins, dtype=float)
    return float(const + h.dot(spins) + 0.5 * spins.dot(J).dot(spins))


def twosat_instances(args                    , size     )                        :
    n_vars = int(TWOSAT_SIZES[size]["n_vars"])
    instance_seeds = [0] if args.mode == "smoke" else parse_csv_list(args.instance_seeds, int)
    instances                       = []
    for instance_seed in instance_seeds:
        seed = int(args.seed) + 1009 * int(instance_seed) + n_vars
        formula = generate_planted_2sat(n_vars, float(args.clause_density), seed=seed)
        h, J, const = twosat_ising(formula["clauses"], n_vars)
        formula["h"] = h
        formula["J"] = J
        formula["const"] = const
        instances.append({"graph_type": "planted_2sat", "instance_seed": instance_seed, "formula": formula})
    return instances


def twosat_default_params(mode     , n_cycles     )                  :
    return maxcut_default_params(mode, n_cycles)


def clamp_twosat_params(params                , mode     , kw_max        = 8.0)                  :
    return clamp_maxcut_params(params, mode, kw_max=kw_max)


def random_twosat_params(rng                     , mode     , n_cycle_choices           , kw_max        = 8.0)                  :
    return random_maxcut_params(rng, mode, n_cycle_choices, kw_max=kw_max)


def twosat_single_trial(
    formula                ,
    mode     ,
    params                ,
    rng                     ,
)                  :
    clauses = np.asarray(formula["clauses"], dtype=int)
    n_vars = int(formula["n_vars"])
    h = np.asarray(formula["h"], dtype=float)
    J = np.asarray(formula["J"], dtype=float)
    const = float(formula["const"])
    n_cycles = int(params["n_cycles"])
    spins = rng.choice(np.array([-1, 1], dtype=int), size=n_vars)
    hold_state = np.zeros(n_vars, dtype=float)
    best_spins = spins.copy()
    best_unsat = twosat_unsat_count(spins, clauses)
    best_energy = twosat_energy(spins, h, J, const)
    best_cycle = 0
    first_solution_cycle = 0 if best_unsat == 0 else None
    last_cycle = 0
    schedule = make_schedule(
        params["I0_min"],
        params["I0_max"],
        n_cycles,
        params["I0_schedule_type"],
        params["I0_schedule_shape"],
        params["I0_hold_fraction"],
    )

    for cycle in range(n_cycles):
        last_cycle = cycle + 1
        spins_prev = spins.copy()
        hold_prev = hold_state.copy()
        local_field = h + J.dot(spins_prev)
        I_vec = -float(params["kw"]) * local_field
        H_new = np.tanh(float(schedule[cycle]) * I_vec)
        rnd = rng.uniform(-1.0, 1.0, size=n_vars)

        if mode == "pSA":
            H_eff = H_new
        elif mode == "lambda_pSA":
            H_eff = float(params["lambda_mem"]) * hold_prev + H_new
        elif mode == "tau_pSA":
            H_eff = float(params["response_lambda"]) * hold_prev + (1.0 - float(params["response_lambda"])) * H_new
        else:
            raise ValueError("Unknown 2-SAT mode: {}".format(mode))

        candidate = np.where(H_eff + rnd >= 0.0, 1, -1).astype(int)
        if float(params["psa_p"]) > 0.0:
            hold_mask = rng.random(n_vars) < float(params["psa_p"])
            candidate[hold_mask] = spins_prev[hold_mask]
            if mode == "tau_pSA":
                H_eff[hold_mask] = hold_prev[hold_mask]
            elif mode in {"pSA", "lambda_pSA"}:
                H_new[hold_mask] = hold_prev[hold_mask]
        spins = candidate
        hold_state = H_eff if mode == "tau_pSA" else H_new

        unsat = twosat_unsat_count(spins, clauses)
        if unsat == 0 and first_solution_cycle is None:
            first_solution_cycle = cycle + 1
        if unsat < best_unsat:
            best_unsat = unsat
            best_energy = twosat_energy(spins, h, J, const)
            best_spins = spins.copy()
            best_cycle = cycle + 1
            if best_unsat == 0:
                break

    if params["decision_method"] == "last":
        final_spins = spins
        final_unsat = twosat_unsat_count(final_spins, clauses)
        final_energy = twosat_energy(final_spins, h, J, const)
        convergence_cycle = last_cycle
    else:
        final_spins = best_spins
        final_unsat = best_unsat
        final_energy = best_energy
        convergence_cycle = best_cycle

    return {
        "unsat": int(final_unsat),
        "energy": float(final_energy),
        "best_energy": float(best_energy),
        "convergence_cycle": int(convergence_cycle),
        "time_to_first_solution": "" if first_solution_cycle is None else int(first_solution_cycle),
        "state": final_spins,
    }


def simulate_2sat(
    formula                ,
    mode     ,
    params                ,
    n_trials     ,
    seed     ,
)                  :
    rng = np.random.default_rng(seed)
    trials = [twosat_single_trial(formula, mode, params, rng) for _ in range(max(1, int(n_trials)))]
    unsat = np.array([trial["unsat"] for trial in trials], dtype=float)
    energies = np.array([trial["energy"] for trial in trials], dtype=float)
    convergence = np.array([trial["convergence_cycle"] for trial in trials], dtype=float)
    n_clauses = max(float(formula["n_clauses"]), 1.0)
    return {
        "best_unsat": int(np.min(unsat)),
        "mean_unsat": float(np.mean(unsat)),
        "std_unsat": float(np.std(unsat)),
        "best_unsat_rate": float(np.min(unsat) / n_clauses),
        "mean_unsat_rate": float(np.mean(unsat) / n_clauses),
        "success_rate": float(np.mean(unsat == 0.0)),
        "valid_assignment_rate": float(np.mean(unsat == 0.0)),
        "convergence_cycle": float(np.mean(convergence)),
        "best_energy": float(np.min(energies)),
        "mean_energy": float(np.mean(energies)),
        "clause_density": float(formula["clause_density"]),
        "n_clauses": int(formula["n_clauses"]),
    }


def twosat_record_row(
    size     ,
    mode     ,
    params                ,
    result                ,
    stage     ,
    candidate_id     ,
    rank     ,
    seed     ,
    graph_type      = "",
    instance_seed            = "",
    score               = None,
)                  :
    row = row_template("2sat", size, mode)
    for field in PARAM_COLUMNS:
        row[field] = params.get(field, "")
    row.update(
        {
            "stage": stage,
            "candidate_id": candidate_id,
            "rank": rank,
            "graph_type": graph_type,
            "instance_seed": instance_seed,
            "best_unsat": result.get("best_unsat", ""),
            "mean_unsat": result.get("mean_unsat", ""),
            "mean_unsat_rate": result.get("mean_unsat_rate", ""),
            "best_unsat_rate": result.get("best_unsat_rate", ""),
            "std_unsat": result.get("std_unsat", ""),
            "success_rate": result.get("success_rate", ""),
            "valid_assignment_rate": result.get("valid_assignment_rate", ""),
            "score": result.get("score", score if score is not None else ""),
            "seed": seed,
            "convergence_cycle": result.get("convergence_cycle", ""),
            "best_energy": result.get("best_energy", ""),
            "mean_energy": result.get("mean_energy", ""),
            "clause_density": result.get("clause_density", ""),
            "n_clauses": result.get("n_clauses", ""),
        }
    )
    return row


def aggregate_twosat_record(
    size     ,
    mode     ,
    params                ,
    instance_rows                      ,
    stage     ,
    candidate_id     ,
    rank     ,
    seed     ,
    success_weight        = 0.10,
    best_unsat_weight        = 0.25,
    convergence_weight        = 0.001,
)                  :
    mean_unsat_rate = safe_mean([safe_float(row.get("mean_unsat_rate")) for row in instance_rows])
    best_unsat_rate = safe_mean([safe_float(row.get("best_unsat_rate")) for row in instance_rows])
    success_mean = safe_mean([safe_float(row.get("success_rate")) for row in instance_rows])
    convergence = safe_mean([safe_float(row.get("convergence_cycle")) for row in instance_rows])
    n_cycles = max(safe_float(params.get("n_cycles"), 1.0), 1.0)
    convergence_frac = convergence / n_cycles if math.isfinite(convergence) else 0.0
    score = (
        mean_unsat_rate
        + float(best_unsat_weight) * best_unsat_rate
        - float(success_weight) * success_mean
        + float(convergence_weight) * convergence_frac
    )
    result = {
        "best_unsat": safe_mean([safe_float(row.get("best_unsat")) for row in instance_rows]),
        "mean_unsat": safe_mean([safe_float(row.get("mean_unsat")) for row in instance_rows]),
        "std_unsat": safe_mean([safe_float(row.get("std_unsat")) for row in instance_rows]),
        "best_unsat_rate": best_unsat_rate,
        "mean_unsat_rate": mean_unsat_rate,
        "success_rate": success_mean,
        "valid_assignment_rate": safe_mean([safe_float(row.get("valid_assignment_rate")) for row in instance_rows]),
        "convergence_cycle": convergence,
        "best_energy": safe_mean([safe_float(row.get("best_energy")) for row in instance_rows]),
        "mean_energy": safe_mean([safe_float(row.get("mean_energy")) for row in instance_rows]),
        "clause_density": safe_mean([safe_float(row.get("clause_density")) for row in instance_rows]),
        "n_clauses": safe_mean([safe_float(row.get("n_clauses")) for row in instance_rows]),
        "score": score,
    }
    return twosat_record_row(size, mode, params, result, stage, candidate_id, rank, seed, score=score)


def evaluate_twosat_candidate(
    args                    ,
    size     ,
    mode     ,
    params                ,
    instances                      ,
    stage     ,
    candidate_id     ,
    seed     ,
)                                               :
    rows = []
    for instance in instances:
        result = simulate_2sat(
            formula=instance["formula"],
            mode=mode,
            params=params,
            n_trials=int(args.n_trials),
            seed=seed + 7919 * (int(instance["instance_seed"]) + 1),
        )
        rows.append(
            twosat_record_row(
                size=size,
                mode=mode,
                params=params,
                result=result,
                stage=stage,
                candidate_id=candidate_id,
                rank=0,
                seed=seed,
                graph_type=instance["graph_type"],
                instance_seed=instance["instance_seed"],
            )
        )
    summary = aggregate_twosat_record(
        size,
        mode,
        params,
        rows,
        stage,
        candidate_id,
        0,
        seed,
        success_weight=safe_float(getattr(args, "twosat_success_weight", 0.10), 0.10),
        best_unsat_weight=safe_float(getattr(args, "twosat_best_unsat_weight", 0.25), 0.25),
        convergence_weight=safe_float(getattr(args, "twosat_convergence_weight", 0.001), 0.001),
    )
    return summary, rows


def init_twosat_worker(args                    , size     , instances                      )        :
    global _TWOSAT_WORKER_ARGS, _TWOSAT_WORKER_SIZE, _TWOSAT_WORKER_INSTANCES
    _TWOSAT_WORKER_ARGS = args
    _TWOSAT_WORKER_SIZE = size
    _TWOSAT_WORKER_INSTANCES = instances


def evaluate_twosat_task(task                )                                               :
    if _TWOSAT_WORKER_ARGS is None or _TWOSAT_WORKER_INSTANCES is None:
        raise RuntimeError("2-SAT worker was not initialized")
    return evaluate_twosat_candidate(
        _TWOSAT_WORKER_ARGS,
        _TWOSAT_WORKER_SIZE,
        task["mode"],
        task["params"],
        _TWOSAT_WORKER_INSTANCES,
        task["stage"],
        task["candidate_id"],
        task["seed"],
    )


def twosat_worker_count(args                    , n_tasks     )        :
    if n_tasks <= 1 or args.mode == "smoke":
        return 1
    return min(max(1, int(getattr(args, "n_workers", 1))), n_tasks)


def twosat_instance_sort_key(row                )        :
    return (
        int(safe_float(row.get("candidate_id"), 0.0)),
        str(row.get("graph_type", "")),
        int(safe_float(row.get("instance_seed"), 0.0)),
    )


def evaluate_twosat_tasks(
    args                    ,
    size     ,
    instances                      ,
    tasks                      ,
    progress_label     ,
)                                               :
    n_tasks = len(tasks)
    n_workers = twosat_worker_count(args, n_tasks)
    results = []
    if n_workers <= 1:
        for task in tasks:
            summary, rows = evaluate_twosat_candidate(
                args,
                size,
                task["mode"],
                task["params"],
                instances,
                task["stage"],
                task["candidate_id"],
                task["seed"],
            )
            results.append((summary, rows))
            print(
                "{} {}/{}: score={:.4g}".format(
                    progress_label,
                    len(results),
                    n_tasks,
                    safe_float(summary.get("score"), math.inf),
                ),
                flush=True,
            )
        return results

    print("{} using {} workers".format(progress_label, n_workers), flush=True)
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=init_twosat_worker,
        initargs=(args, size, instances),
    ) as executor:
        futures = {executor.submit(evaluate_twosat_task, task): task for task in tasks}
        for future in as_completed(futures):
            summary, rows = future.result()
            results.append((summary, rows))
            print(
                "{} {}/{}: score={:.4g}".format(
                    progress_label,
                    len(results),
                    n_tasks,
                    safe_float(summary.get("score"), math.inf),
                ),
                flush=True,
            )
    return results


def run_twosat_deep(
    args                    ,
    root      ,
    size     ,
    instances                      ,
)                             :
    n_cycles = int(args.n_cycles) if args.mode == "smoke" else int(args.twosat_cycles)
    n_cycle_choices = [n_cycles] if args.mode == "smoke" else parse_csv_list(args.twosat_cycle_choices, int)
    search_candidates = max(1, int(args.search_candidates if args.search_candidates is not None else (2 if args.mode == "smoke" else 128)))
    best_by_mode                            = {}

    for mode in PAPER_MODES:
        rng = np.random.default_rng(int(args.seed) + 301003 * (PAPER_MODES.index(mode) + 1))
        summaries                       = []
        instance_rows                       = []
        candidates = [twosat_default_params(mode, n_cycles)]
        candidates.extend(random_twosat_params(rng, mode, n_cycle_choices) for _ in range(max(0, search_candidates - 1)))

        print("[2-SAT {}] deep search: {}".format(size, mode), flush=True)
        tasks = []
        for idx, params in enumerate(candidates, start=1):
            seed = int(args.seed) + 300000 * (PAPER_MODES.index(mode) + 1) + 7919 * idx
            tasks.append({
                "mode": mode,
                "params": params,
                "stage": "search",
                "candidate_id": idx,
                "seed": seed,
            })
        for summary, rows in evaluate_twosat_tasks(
            args,
            size,
            instances,
            tasks,
            "[2-SAT {} {}] search".format(size, mode),
        ):
            summaries.append(summary)
            instance_rows.extend(rows)

        ranked = sorted(summaries, key=lambda row: safe_float(row.get("score"), math.inf))
        for rank, row in enumerate(ranked, start=1):
            row["rank"] = rank
        rank_by_candidate = {int(row["candidate_id"]): int(row["rank"]) for row in ranked}
        for row in instance_rows:
            row["rank"] = rank_by_candidate.get(int(row["candidate_id"]), "")
        instance_rows = sorted(instance_rows, key=twosat_instance_sort_key)

        best = ranked[0]
        best_by_mode[mode] = best
        mode_dir = root / "2sat" / size / mode
        write_csv(ranked, mode_dir / "deep_search_summary.csv")
        write_csv(instance_rows, mode_dir / "deep_search_by_instance.csv")
        write_csv([best], mode_dir / "best_params.csv")

    return best_by_mode


def run_twosat_sweeps(
    args                    ,
    root      ,
    size     ,
    instances                      ,
    best_by_mode                           ,
)                             :
    final_by_mode                            = {}

    for mode in PAPER_MODES:
        base_params = {field: best_by_mode[mode].get(field, "") for field in PARAM_COLUMNS}
        base_params = clamp_twosat_params(base_params, mode)
        if mode == "pSA":
            values = [0.0]
            sweep_field = "lambda_mem"
            max_value = 0.0
        else:
            sweep_field = "response_lambda" if mode == "tau_pSA" else "lambda_mem"
            max_value = 0.99 if mode == "tau_pSA" else 0.95
            values = memory_grid(safe_float(base_params.get(sweep_field), 0.0), max_value, args.mode == "smoke")

        summaries                       = []
        instance_rows                       = []
        print("[2-SAT {}] memory sweep: {}".format(size, mode), flush=True)
        tasks = []
        for idx, value in enumerate(values, start=1):
            params = dict(base_params)
            params[sweep_field] = min(max(float(value), 0.0), max_value)
            params = clamp_twosat_params(params, mode)
            seed = int(args.seed) + 900000 + 300000 * (PAPER_MODES.index(mode) + 1) + 7919 * idx
            tasks.append({
                "mode": mode,
                "params": params,
                "stage": "memory_sweep",
                "candidate_id": idx,
                "seed": seed,
            })
        for summary, rows in evaluate_twosat_tasks(
            args,
            size,
            instances,
            tasks,
            "[2-SAT {} {}] memory sweep".format(size, mode),
        ):
            summaries.append(summary)
            instance_rows.extend(rows)
        instance_rows = sorted(instance_rows, key=twosat_instance_sort_key)

        # Sort by the memory value stored in the record, not by zip order.
        if mode == "lambda_pSA":
            summaries = sorted(summaries, key=lambda row: safe_float(row.get("lambda_mem"), 0.0))
        elif mode == "tau_pSA":
            summaries = sorted(summaries, key=lambda row: safe_float(row.get("response_lambda"), 0.0))

        ranked = sorted(summaries, key=lambda row: safe_float(row.get("score"), math.inf))
        best = ranked[0]
        final_by_mode[mode] = best
        mode_dir = root / "2sat" / size / mode
        write_csv(summaries, mode_dir / "memory_sweep_summary.csv")
        write_csv(instance_rows, mode_dir / "memory_sweep_by_instance.csv")
        write_csv([best], mode_dir / "final_comparison.csv")

    return final_by_mode


def check_twosat_tau_zero_matches_psa(size     , instances                      , n_cycles     )                  :
    params = twosat_default_params("pSA", n_cycles)
    params["psa_p"] = 0.0
    formula = instances[0]["formula"]
    psa = [twosat_single_trial(formula, "pSA", params, np.random.default_rng(123))]
    tau_params = dict(params)
    tau_params["response_lambda"] = 0.0
    tau = [twosat_single_trial(formula, "tau_pSA", tau_params, np.random.default_rng(123))]
    return {
        "jacobi_parallel_update": True,
        "local_field_from_spins_prev": True,
        "H_eff_from_hold_prev": True,
        "response_lambda_zero_matches_pSA": bool(
            psa[0]["unsat"] == tau[0]["unsat"] and np.array_equal(psa[0]["state"], tau[0]["state"])
        ),
    }


def run_twosat_size(args                    , root      , size     )        :
    output_size = twosat_output_label(args, size)
    if output_size != size:
        print("[2-SAT {}] output label: {}".format(size, output_size), flush=True)
    instances = twosat_instances(args, size)
    best_by_mode = run_twosat_deep(args, root, output_size, instances)
    final_by_mode = run_twosat_sweeps(args, root, output_size, instances, best_by_mode)
    comparison_rows = [final_by_mode[mode] for mode in PAPER_MODES]
    write_csv(comparison_rows, root / "2sat" / output_size / "comparison" / "final_comparison.csv")
    checks = check_twosat_tau_zero_matches_psa(output_size, instances, int(args.n_cycles if args.mode == "smoke" else args.twosat_cycles))
    format_json(root / "2sat" / output_size / "comparison" / "tau_pSA_checks.json", checks)


# ---------------------------------------------------------------------------
# 2-SAT phase-transition evaluator
# ---------------------------------------------------------------------------


def is_twosat_phase_repro(args                    )        :
    return str(getattr(args, "problem", "")) == "2sat-phase-repro"


def twosat_phase_dir_name(args                    )      :
    return "2sat_phase_repro" if is_twosat_phase_repro(args) else "2sat_phase"


def twosat_phase_problem_label(args                    )      :
    return "2sat-phase-repro" if is_twosat_phase_repro(args) else "2sat-phase"


def twosat_phase_log_label(args                    )      :
    return "2-SAT phase repro" if is_twosat_phase_repro(args) else "2-SAT phase"


def twosat_phase_kw_max(args                    , mode     )        :
    if mode == "tau_pSA":
        return max(0.05, safe_float(getattr(args, "tau_kw_max", 8.0), 8.0))
    return 8.0


def truthy(value     )        :
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def twosat_phase_instance_score(row                , params                , args                    )         :
    mean_unsat_rate = safe_float(row.get("mean_unsat_rate"), math.inf)
    best_unsat_rate = safe_float(row.get("best_unsat_rate"), 0.0)
    success = safe_float(row.get("success_rate_all"), 0.0) if truthy(row.get("is_satisfiable")) else 0.0
    convergence = safe_float(row.get("mean_convergence_cycle"), 0.0)
    n_cycles = max(safe_float(params.get("n_cycles"), safe_float(row.get("n_cycles"), 1.0)), 1.0)
    return (
        mean_unsat_rate
        + safe_float(getattr(args, "twosat_best_unsat_weight", 0.25), 0.25) * best_unsat_rate
        - safe_float(getattr(args, "twosat_success_weight", 0.10), 0.10) * success
        + safe_float(getattr(args, "twosat_convergence_weight", 0.001), 0.001) * (convergence / n_cycles)
    )


def twosat_phase_instances(args                    , size     )                        :
    spec = TWOSAT_PHASE_SIZES[size]
    n_vars = int(spec["n_vars"])
    alpha = float(spec["alpha"])
    instance_seeds = [0] if args.mode == "smoke" else parse_csv_list(args.instance_seeds, int)
    instances                       = []
    for instance_seed in instance_seeds:
        seed = int(args.seed) + 2003 * int(instance_seed) + n_vars + int(round(1000.0 * alpha))
        formula = generate_random_2sat(n_vars, alpha, seed=seed)
        h, J, const = twosat_ising(formula["clauses"], n_vars)
        formula["h"] = h
        formula["J"] = J
        formula["const"] = const
        instances.append({"graph_type": "random_2sat", "instance_seed": instance_seed, "formula": formula})
    return instances


def simulate_2sat_phase(
    formula                ,
    mode     ,
    params                ,
    n_trials     ,
    seed     ,
)                  :
    rng = np.random.default_rng(seed)
    trials = [twosat_single_trial(formula, mode, params, rng) for _ in range(max(1, int(n_trials)))]
    unsat = np.array([trial["unsat"] for trial in trials], dtype=float)
    final_energies = np.array([trial["energy"] for trial in trials], dtype=float)
    best_energies = np.array([trial["best_energy"] for trial in trials], dtype=float)
    convergence = np.array([trial["convergence_cycle"] for trial in trials], dtype=float)
    first_solution_cycles = [
        safe_float(trial.get("time_to_first_solution"))
        for trial in trials
        if math.isfinite(safe_float(trial.get("time_to_first_solution")))
    ]
    n_clauses = max(float(formula["n_clauses"]), 1.0)
    is_sat = bool(formula.get("is_satisfiable", True))
    success = float(np.mean(unsat == 0.0))
    return {
        "best_unsat": int(np.min(unsat)),
        "mean_unsat": float(np.mean(unsat)),
        "std_unsat": float(np.std(unsat)),
        "best_unsat_rate": float(np.min(unsat) / n_clauses),
        "mean_unsat_rate": float(np.mean(unsat) / n_clauses),
        "best_unsatisfied_clauses": int(np.min(unsat)),
        "mean_unsatisfied_clauses": float(np.mean(unsat)),
        "success_rate": success,
        "success_rate_all": success,
        "success_rate_sat_only": success if is_sat else "",
        "valid_assignment_rate": success if is_sat else "",
        "unsat_instance_rate": 0.0 if is_sat else 1.0,
        "is_satisfiable": is_sat,
        "mean_convergence_cycle": float(np.mean(convergence)),
        "median_convergence_cycle": float(np.median(convergence)),
        "convergence_cycle": float(np.mean(convergence)),
        "final_energy": float(np.mean(final_energies)),
        "best_energy": float(np.min(best_energies)),
        "mean_energy": float(np.mean(final_energies)),
        "time_to_first_solution": safe_mean(first_solution_cycles) if first_solution_cycles else "",
        "clause_density": float(formula["clause_density"]),
        "alpha": float(formula["alpha"]),
        "N": int(formula["n_vars"]),
        "M": int(formula["n_clauses"]),
        "n_clauses": int(formula["n_clauses"]),
    }


def twosat_phase_record_row(
    size     ,
    mode     ,
    params                ,
    result                ,
    stage     ,
    candidate_id     ,
    rank     ,
    seed     ,
    graph_type      = "",
    instance_seed            = "",
    score               = None,
    n_trials               = "",
    problem_label      = "2sat-phase",
)                  :
    row = row_template(problem_label, size, mode)
    for field in PARAM_COLUMNS:
        row[field] = params.get(field, "")
    row["pbit_alpha"] = params.get("alpha", "")
    row.update(
        {
            "size_label": size,
            "stage": stage,
            "candidate_id": candidate_id,
            "rank": rank,
            "graph_type": graph_type,
            "instance_seed": instance_seed,
            "N": result.get("N", ""),
            "M": result.get("M", result.get("n_clauses", "")),
            "alpha": result.get("alpha", result.get("clause_density", "")),
            "best_unsat": result.get("best_unsat", result.get("best_unsatisfied_clauses", "")),
            "mean_unsat": result.get("mean_unsat", result.get("mean_unsatisfied_clauses", "")),
            "mean_unsat_rate": result.get("mean_unsat_rate", ""),
            "mean_unsatisfied_clauses": result.get("mean_unsatisfied_clauses", ""),
            "best_unsat_rate": result.get("best_unsat_rate", ""),
            "best_unsatisfied_clauses": result.get("best_unsatisfied_clauses", ""),
            "std_unsat": result.get("std_unsat", ""),
            "success_rate": result.get("success_rate", result.get("success_rate_all", "")),
            "success_rate_all": result.get("success_rate_all", result.get("success_rate", "")),
            "success_rate_sat_only": result.get("success_rate_sat_only", ""),
            "valid_assignment_rate": result.get("valid_assignment_rate", ""),
            "unsat_instance_rate": result.get("unsat_instance_rate", ""),
            "is_satisfiable": result.get("is_satisfiable", ""),
            "mean_convergence_cycle": result.get("mean_convergence_cycle", result.get("convergence_cycle", "")),
            "median_convergence_cycle": result.get("median_convergence_cycle", ""),
            "convergence_cycle": result.get("convergence_cycle", result.get("mean_convergence_cycle", "")),
            "final_energy": result.get("final_energy", ""),
            "best_energy": result.get("best_energy", ""),
            "mean_energy": result.get("mean_energy", ""),
            "time_to_first_solution": result.get("time_to_first_solution", ""),
            "clause_density": result.get("clause_density", result.get("alpha", "")),
            "n_clauses": result.get("n_clauses", result.get("M", "")),
            "n_satisfiable_instances": result.get("n_satisfiable_instances", ""),
            "n_unsatisfiable_instances": result.get("n_unsatisfiable_instances", ""),
            "score": result.get("score", score if score is not None else ""),
            "seed": seed,
            "n_trials": n_trials,
        }
    )
    for key, value in list(row.items()):
        if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
            row[key] = ""
    return row


def aggregate_twosat_phase_record(
    size     ,
    mode     ,
    params                ,
    instance_rows                      ,
    stage     ,
    candidate_id     ,
    rank     ,
    seed     ,
    n_trials     ,
    success_weight        = 0.10,
    best_unsat_weight        = 0.25,
    convergence_weight        = 0.001,
    problem_label      = "2sat-phase",
)                  :
    sat_flags = [truthy(row.get("is_satisfiable")) for row in instance_rows]
    sat_rows = [row for row, is_sat in zip(instance_rows, sat_flags) if is_sat]
    n_instances = max(len(instance_rows), 1)
    n_unsat_instances = sum(1 for is_sat in sat_flags if not is_sat)
    mean_unsat_rate = safe_mean([safe_float(row.get("mean_unsat_rate")) for row in instance_rows])
    best_unsat_rate = safe_mean([safe_float(row.get("best_unsat_rate")) for row in instance_rows])
    success_all = safe_mean([safe_float(row.get("success_rate_all")) for row in instance_rows])
    success_sat = safe_mean([safe_float(row.get("success_rate_all")) for row in sat_rows]) if sat_rows else math.nan
    mean_convergence = safe_mean([safe_float(row.get("mean_convergence_cycle")) for row in instance_rows])
    median_convergence = safe_mean([safe_float(row.get("median_convergence_cycle")) for row in instance_rows])
    n_cycles = max(safe_float(params.get("n_cycles"), 1.0), 1.0)
    convergence_frac = mean_convergence / n_cycles if math.isfinite(mean_convergence) else 0.0
    score_success = success_sat if math.isfinite(success_sat) else 0.0
    score = (
        mean_unsat_rate
        + float(best_unsat_weight) * best_unsat_rate
        - float(success_weight) * score_success
        + float(convergence_weight) * convergence_frac
    )
    result = {
        "N": safe_mean([safe_float(row.get("N")) for row in instance_rows]),
        "M": safe_mean([safe_float(row.get("M")) for row in instance_rows]),
        "alpha": safe_mean([safe_float(row.get("alpha")) for row in instance_rows]),
        "best_unsat": safe_mean([safe_float(row.get("best_unsat")) for row in instance_rows]),
        "mean_unsat": safe_mean([safe_float(row.get("mean_unsat")) for row in instance_rows]),
        "std_unsat": safe_mean([safe_float(row.get("std_unsat")) for row in instance_rows]),
        "best_unsat_rate": best_unsat_rate,
        "mean_unsat_rate": mean_unsat_rate,
        "best_unsatisfied_clauses": safe_mean([safe_float(row.get("best_unsatisfied_clauses")) for row in instance_rows]),
        "mean_unsatisfied_clauses": safe_mean([safe_float(row.get("mean_unsatisfied_clauses")) for row in instance_rows]),
        "success_rate": success_all,
        "success_rate_all": success_all,
        "success_rate_sat_only": success_sat if math.isfinite(success_sat) else "",
        "valid_assignment_rate": success_sat if math.isfinite(success_sat) else "",
        "unsat_instance_rate": float(n_unsat_instances) / float(n_instances),
        "is_satisfiable": "",
        "mean_convergence_cycle": mean_convergence,
        "median_convergence_cycle": median_convergence,
        "convergence_cycle": mean_convergence,
        "final_energy": safe_mean([safe_float(row.get("final_energy")) for row in instance_rows]),
        "best_energy": safe_mean([safe_float(row.get("best_energy")) for row in instance_rows]),
        "mean_energy": safe_mean([safe_float(row.get("mean_energy")) for row in instance_rows]),
        "time_to_first_solution": safe_mean([safe_float(row.get("time_to_first_solution")) for row in instance_rows]),
        "clause_density": safe_mean([safe_float(row.get("clause_density")) for row in instance_rows]),
        "n_clauses": safe_mean([safe_float(row.get("n_clauses")) for row in instance_rows]),
        "n_satisfiable_instances": n_instances - n_unsat_instances,
        "n_unsatisfiable_instances": n_unsat_instances,
        "score": score,
    }
    return twosat_phase_record_row(
        size,
        mode,
        params,
        result,
        stage,
        candidate_id,
        rank,
        seed,
        n_trials=n_trials,
        score=score,
        problem_label=problem_label,
    )


def evaluate_twosat_phase_candidate(
    args                    ,
    size     ,
    mode     ,
    params                ,
    instances                      ,
    stage     ,
    candidate_id     ,
    seed     ,
)                                               :
    rows = []
    problem_label = twosat_phase_problem_label(args)
    for instance in instances:
        result = simulate_2sat_phase(
            formula=instance["formula"],
            mode=mode,
            params=params,
            n_trials=int(args.n_trials),
            seed=seed + 7919 * (int(instance["instance_seed"]) + 1),
        )
        rows.append(
            twosat_phase_record_row(
                size=size,
                mode=mode,
                params=params,
                result=result,
                stage=stage,
                candidate_id=candidate_id,
                rank=0,
                seed=seed,
                graph_type=instance["graph_type"],
                instance_seed=instance["instance_seed"],
                n_trials=int(args.n_trials),
                problem_label=problem_label,
            )
        )
    for row in rows:
        row["score"] = twosat_phase_instance_score(row, params, args)
    summary = aggregate_twosat_phase_record(
        size,
        mode,
        params,
        rows,
        stage,
        candidate_id,
        0,
        seed,
        int(args.n_trials),
        success_weight=safe_float(getattr(args, "twosat_success_weight", 0.10), 0.10),
        best_unsat_weight=safe_float(getattr(args, "twosat_best_unsat_weight", 0.25), 0.25),
        convergence_weight=safe_float(getattr(args, "twosat_convergence_weight", 0.001), 0.001),
        problem_label=problem_label,
    )
    return summary, rows


def init_twosat_phase_worker(args                    , size     , instances                      )        :
    global _TWOSAT_PHASE_WORKER_ARGS, _TWOSAT_PHASE_WORKER_SIZE, _TWOSAT_PHASE_WORKER_INSTANCES
    _TWOSAT_PHASE_WORKER_ARGS = args
    _TWOSAT_PHASE_WORKER_SIZE = size
    _TWOSAT_PHASE_WORKER_INSTANCES = instances


def evaluate_twosat_phase_task(task                )                                               :
    if _TWOSAT_PHASE_WORKER_ARGS is None or _TWOSAT_PHASE_WORKER_INSTANCES is None:
        raise RuntimeError("2-SAT phase worker was not initialized")
    return evaluate_twosat_phase_candidate(
        _TWOSAT_PHASE_WORKER_ARGS,
        _TWOSAT_PHASE_WORKER_SIZE,
        task["mode"],
        task["params"],
        _TWOSAT_PHASE_WORKER_INSTANCES,
        task["stage"],
        task["candidate_id"],
        task["seed"],
    )


def twosat_phase_instance_sort_key(row                )        :
    return (
        int(safe_float(row.get("candidate_id"), 0.0)),
        str(row.get("graph_type", "")),
        int(safe_float(row.get("instance_seed"), 0.0)),
    )


def evaluate_twosat_phase_tasks(
    args                    ,
    size     ,
    instances                      ,
    tasks                      ,
    progress_label     ,
)                                               :
    n_tasks = len(tasks)
    n_workers = twosat_worker_count(args, n_tasks)
    results = []
    if n_workers <= 1:
        for task in tasks:
            summary, rows = evaluate_twosat_phase_candidate(
                args,
                size,
                task["mode"],
                task["params"],
                instances,
                task["stage"],
                task["candidate_id"],
                task["seed"],
            )
            results.append((summary, rows))
            print(
                "{} {}/{}: score={:.4g}".format(
                    progress_label,
                    len(results),
                    n_tasks,
                    safe_float(summary.get("score"), math.inf),
                ),
                flush=True,
            )
        return results

    print("{} using {} workers".format(progress_label, n_workers), flush=True)
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=init_twosat_phase_worker,
        initargs=(args, size, instances),
    ) as executor:
        futures = {executor.submit(evaluate_twosat_phase_task, task): task for task in tasks}
        for future in as_completed(futures):
            summary, rows = future.result()
            results.append((summary, rows))
            print(
                "{} {}/{}: score={:.4g}".format(
                    progress_label,
                    len(results),
                    n_tasks,
                    safe_float(summary.get("score"), math.inf),
                ),
                flush=True,
            )
    return results


def run_twosat_phase_deep(
    args                    ,
    root      ,
    size     ,
    instances                      ,
)                             :
    n_cycles = int(args.n_cycles) if args.mode == "smoke" else int(args.twosat_cycles)
    n_cycle_choices = [n_cycles] if args.mode == "smoke" else parse_csv_list(args.twosat_cycle_choices, int)
    search_candidates = max(1, int(args.search_candidates if args.search_candidates is not None else (2 if args.mode == "smoke" else 128)))
    best_by_mode                            = {}
    phase_dir = twosat_phase_dir_name(args)
    log_label = twosat_phase_log_label(args)

    for mode in PAPER_MODES:
        rng = np.random.default_rng(int(args.seed) + 401003 * (PAPER_MODES.index(mode) + 1))
        summaries                       = []
        instance_rows                       = []
        kw_max = twosat_phase_kw_max(args, mode)
        candidates = [twosat_default_params(mode, n_cycles)]
        candidates.extend(random_twosat_params(rng, mode, n_cycle_choices, kw_max=kw_max) for _ in range(max(0, search_candidates - 1)))

        print("[{} {}] deep search: {}".format(log_label, size, mode), flush=True)
        tasks = []
        for idx, params in enumerate(candidates, start=1):
            seed = int(args.seed) + 400000 * (PAPER_MODES.index(mode) + 1) + 7919 * idx
            tasks.append({
                "mode": mode,
                "params": params,
                "stage": "search",
                "candidate_id": idx,
                "seed": seed,
            })
        for summary, rows in evaluate_twosat_phase_tasks(
            args,
            size,
            instances,
            tasks,
            "[{} {} {}] search".format(log_label, size, mode),
        ):
            summaries.append(summary)
            instance_rows.extend(rows)

        ranked = sorted(summaries, key=lambda row: safe_float(row.get("score"), math.inf))
        for rank, row in enumerate(ranked, start=1):
            row["rank"] = rank
        rank_by_candidate = {int(row["candidate_id"]): int(row["rank"]) for row in ranked}
        for row in instance_rows:
            row["rank"] = rank_by_candidate.get(int(row["candidate_id"]), "")
        instance_rows = sorted(instance_rows, key=twosat_phase_instance_sort_key)

        best = ranked[0]
        best_by_mode[mode] = best
        mode_dir = root / phase_dir / size / mode
        write_csv(ranked, mode_dir / "deep_search_summary.csv")
        write_csv(instance_rows, mode_dir / "deep_search_by_instance.csv")
        write_csv([best], mode_dir / "best_params.csv")

    return best_by_mode


def run_twosat_phase_sweeps(
    args                    ,
    root      ,
    size     ,
    instances                      ,
    best_by_mode                           ,
)                             :
    final_by_mode                            = {}
    final_instance_rows                       = []
    phase_dir = twosat_phase_dir_name(args)
    log_label = twosat_phase_log_label(args)

    for mode in PAPER_MODES:
        base_params = {field: best_by_mode[mode].get(field, "") for field in PARAM_COLUMNS}
        base_params["alpha"] = best_by_mode[mode].get("pbit_alpha", base_params.get("alpha", 1.0))
        kw_max = twosat_phase_kw_max(args, mode)
        base_params = clamp_twosat_params(base_params, mode, kw_max=kw_max)
        if mode == "pSA":
            values = [0.0]
            sweep_field = "lambda_mem"
            max_value = 0.0
        else:
            sweep_field = "response_lambda" if mode == "tau_pSA" else "lambda_mem"
            max_value = 0.99 if mode == "tau_pSA" else 0.95
            values = memory_grid(safe_float(base_params.get(sweep_field), 0.0), max_value, args.mode == "smoke")

        summaries                       = []
        instance_rows                       = []
        print("[{} {}] memory sweep: {}".format(log_label, size, mode), flush=True)
        tasks = []
        for idx, value in enumerate(values, start=1):
            params = dict(base_params)
            params[sweep_field] = min(max(float(value), 0.0), max_value)
            params = clamp_twosat_params(params, mode, kw_max=kw_max)
            seed = int(args.seed) + 1000000 + 400000 * (PAPER_MODES.index(mode) + 1) + 7919 * idx
            tasks.append({
                "mode": mode,
                "params": params,
                "stage": "memory_sweep",
                "candidate_id": idx,
                "seed": seed,
            })
        for summary, rows in evaluate_twosat_phase_tasks(
            args,
            size,
            instances,
            tasks,
            "[{} {} {}] memory sweep".format(log_label, size, mode),
        ):
            summaries.append(summary)
            instance_rows.extend(rows)
        instance_rows = sorted(instance_rows, key=twosat_phase_instance_sort_key)

        if mode == "lambda_pSA":
            summaries = sorted(summaries, key=lambda row: safe_float(row.get("lambda_mem"), 0.0))
        elif mode == "tau_pSA":
            summaries = sorted(summaries, key=lambda row: safe_float(row.get("response_lambda"), 0.0))

        ranked = sorted(summaries, key=lambda row: safe_float(row.get("score"), math.inf))
        best = ranked[0]
        final_by_mode[mode] = best
        best_candidate_id = str(best.get("candidate_id", ""))
        selected_rows = [row for row in instance_rows if str(row.get("candidate_id", "")) == best_candidate_id]
        final_instance_rows.extend(selected_rows)
        mode_dir = root / phase_dir / size / mode
        write_csv(summaries, mode_dir / "memory_sweep_summary.csv")
        write_csv(instance_rows, mode_dir / "memory_sweep_by_instance.csv")
        write_csv(selected_rows, mode_dir / "by_instance.csv")
        write_csv([best], mode_dir / "final_comparison.csv")

    return final_by_mode, final_instance_rows


def check_twosat_phase_tau_zero_matches_psa(size     , instances                      , n_cycles     )                  :
    checks = check_twosat_tau_zero_matches_psa(size, instances, n_cycles)
    formula = instances[0]["formula"]
    checks["implication_graph_scc_satisfiability"] = bool(
        twosat_is_satisfiable(formula["clauses"], int(formula["n_vars"])) == bool(formula.get("is_satisfiable", True))
    )
    return checks


def run_twosat_phase_size(args                    , root      , size     )        :
    output_size = twosat_phase_output_label(size)
    phase_dir = twosat_phase_dir_name(args)
    log_label = twosat_phase_log_label(args)
    if output_size != size:
        print("[{} {}] output label: {}".format(log_label, size, output_size), flush=True)
    instances = twosat_phase_instances(args, size)
    best_by_mode = run_twosat_phase_deep(args, root, output_size, instances)
    final_by_mode, final_instance_rows = run_twosat_phase_sweeps(args, root, output_size, instances, best_by_mode)
    comparison_rows = [final_by_mode[mode] for mode in PAPER_MODES]
    write_csv(comparison_rows, root / phase_dir / output_size / "comparison" / "final_comparison.csv")
    write_csv(final_instance_rows, root / phase_dir / output_size / "comparison" / "by_instance.csv")
    checks = check_twosat_phase_tau_zero_matches_psa(output_size, instances, int(args.n_cycles if args.mode == "smoke" else args.twosat_cycles))
    format_json(root / phase_dir / output_size / "comparison" / "tau_pSA_checks.json", checks)


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------


def postprocess(root      , skip_figures      , skip_tables      )        :
    if not skip_tables:
        import make_paper_tables

        make_paper_tables.build_tables(root)
    if not skip_figures:
        import make_paper_figures

        make_paper_figures.build_figures(root)


def build_parser()                           :
    parser = argparse.ArgumentParser(description="Run paper pSA/lambda_pSA/tau_pSA experiments.")
    parser.add_argument("--problem", choices=["ldpc", "maxcut", "maxcut-structure", "2sat", "2sat-phase", "2sat-phase-repro", "all"], default="all")
    parser.add_argument("--size", default="all", help="Size label or comma-separated labels. Use all for all configured sizes.")
    parser.add_argument("--mode", choices=["smoke", "full", "all"], default="smoke")
    parser.add_argument("--output-dir", default=str(PAPER_ROOT))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-workers", type=int, default=8)
    parser.add_argument("--n-trials", type=int, default=2)
    parser.add_argument("--n-cycles", type=int, default=5)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument("--skip-tables", action="store_true")

    parser.add_argument("--ebno-values", default="2.0,2.5,3.0")
    parser.add_argument("--matrix-seed", type=int, default=0)
    parser.add_argument("--search-algorithm", choices=["optuna", "random"], default="optuna")
    parser.add_argument("--search-candidates", type=int, default=None)
    parser.add_argument("--search-trials", type=int, default=160)
    parser.add_argument("--refine-top-k", type=int, default=None)
    parser.add_argument("--refine-trials", type=int, default=2000)
    parser.add_argument("--refine-seed-count", type=int, default=6)
    parser.add_argument("--final-trials", type=int, default=10000)
    parser.add_argument("--final-seed-count", type=int, default=10)
    parser.add_argument("--bp-trials", type=int, default=5000)
    parser.add_argument("--eps", type=float, default=1e-12)
    parser.add_argument("--fer-weight", type=float, default=0.2)
    parser.add_argument("--syndrome-weight", type=float, default=0.5)
    parser.add_argument("--fixed-bit-width", type=int, default=8)
    parser.add_argument("--channel-input-mode", choices=["float", "fixed"], default="float")
    parser.add_argument("--bp-max-iter", type=int, default=50)
    parser.add_argument("--bp-llr-clip", type=float, default=50.0)

    parser.add_argument("--graph-types", default="random_regular,erdos_renyi")
    parser.add_argument("--instance-seeds", default="0,1,2")
    parser.add_argument("--regular-degree", type=int, default=3)
    parser.add_argument("--edge-prob", type=float, default=None)
    parser.add_argument("--maxcut-cycles", type=int, default=1000)
    parser.add_argument("--maxcut-cycle-choices", default="100,300,1000,3000")
    parser.add_argument("--maxcut-success-weight", type=float, default=0.0)
    parser.add_argument("--clause-density", type=float, default=3.0)
    parser.add_argument("--twosat-cycles", type=int, default=1000)
    parser.add_argument("--twosat-cycle-choices", default="100,300,1000,3000")
    parser.add_argument("--twosat-success-weight", type=float, default=0.10)
    parser.add_argument("--twosat-best-unsat-weight", type=float, default=0.25)
    parser.add_argument("--twosat-convergence-weight", type=float, default=0.001)
    parser.add_argument("--tau-kw-max", type=float, default=8.0)
    return parser


def main()        :
    args = build_parser().parse_args()
    root = Path(args.output_dir).resolve()
    ensure_tree(root)

    if args.mode != "smoke":
        print("Running full paper experiments. This can be expensive.", flush=True)

    if args.problem in {"ldpc", "all"}:
        for size in parse_size_list(args.size, LDPC_SIZES):
            run_ldpc_size(args, root, size)

    if args.problem in {"maxcut", "all"}:
        for size in parse_size_list(args.size, MAXCUT_SIZES):
            run_maxcut_size(args, root, size)

    if args.problem == "maxcut-structure":
        for size in parse_size_list(args.size, MAXCUT_STRUCTURE_SIZES):
            run_maxcut_structure_size(args, root, size)

    if args.problem in {"2sat", "all"}:
        for size in parse_size_list(args.size, TWOSAT_SIZES):
            run_twosat_size(args, root, size)

    if args.problem == "2sat-phase":
        for size in parse_size_list(args.size, TWOSAT_PHASE_SIZES):
            run_twosat_phase_size(args, root, size)

    if args.problem == "2sat-phase-repro":
        for size in parse_size_list(args.size, TWOSAT_PHASE_REPRO_SIZES):
            run_twosat_phase_size(args, root, size)

    postprocess(root, skip_figures=args.skip_figures, skip_tables=args.skip_tables)
    print("Done. Outputs are under {}".format(root), flush=True)


if __name__ == "__main__":
    main()

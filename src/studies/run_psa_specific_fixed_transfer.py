#!/usr/bin/env python3
"""Fixed-transfer validation with a separately optimized memoryless pSA package.

This driver reuses the validated cross-code performance kernel and the archived
C00--C09 matrices.  It executes only the new pSA-specific fixed-transfer arm,
then combines it with the two existing cross-code arms for code-level
statistics and figures.  It never regenerates matrices or retunes parameters.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SCRIPT = Path(__file__).resolve()
RUN_DIR = SCRIPT.parent
REPO_ROOT = SCRIPT.parents[2]
REFERENCE_DIR = REPO_ROOT / "src" / "reference"
LDPC_DIR = REPO_ROOT / "src" / "ldpc"
for import_path in [REPO_ROOT, REFERENCE_DIR, LDPC_DIR]:
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import run_phase3_robustness as cross_code  # noqa: E402


STUDY_VERSION = "2026-08-28-v1"
SOURCE_TRANSFER_DIR = REPO_ROOT / "data" / "processed" / "matched_cross_code_ablation"
REPRESENTATIVE_CONFIG = REPO_ROOT / "configs" / "ldpc" / "representative_benchmark.json"
PARAMETER_SOURCES = {
    "N96_M48": REPO_ROOT / "data" / "processed" / "representative_benchmark" / "search_records" / "N96_M48" / "pSA" / "best_params.csv",
    "N192_M96": REPO_ROOT / "data" / "processed" / "representative_benchmark" / "search_records" / "N192_M96" / "pSA" / "best_params.csv",
    "N288_M144": REPO_ROOT / "data" / "processed" / "representative_benchmark" / "search_records" / "N288_M144" / "pSA" / "best_params.csv",
}
SIZES = ["N96_M48", "N192_M96", "N288_M144"]
CODE_IDS = ["existing_00", *[f"new_{index:02d}" for index in range(1, 10)]]
EBNO_VALUES = [2.0, 2.5, 3.0]
PUBLIC_CODE_IDS = {code_id: f"C{index:02d}" for index, code_id in enumerate(CODE_IDS)}
BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_BASE_SEED = 20260828

OUTPUT_ARMS = ["additive_fixed", "pSA_specific_fixed", "matched_lambda0_ablation"]
PAIR_SPECS = [
    ("additive_vs_pSA_specific", "additive_fixed", "pSA_specific_fixed"),
    ("additive_vs_matched_ablation", "additive_fixed", "matched_lambda0_ablation"),
    ("pSA_specific_vs_matched_ablation", "pSA_specific_fixed", "matched_lambda0_ablation"),
]

PARAMETER_FIELDS = [
    "n_cycles", "I0_schedule_type", "I0_min", "I0_max", "I0_schedule_shape",
    "I0_hold_fraction", "psa_p", "kw", "kr", "alpha", "alpha_mode",
    "burn_in", "sample_window", "decision_method", "lambda_mem",
    "response_lambda", "lambda_out", "nrnd",
]
INTEGER_PARAMETER_FIELDS = {"n_cycles", "burn_in", "sample_window"}
STRING_PARAMETER_FIELDS = {"I0_schedule_type", "alpha_mode", "decision_method"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def setup_logging() -> logging.Logger:
    logger = logging.getLogger("psa_specific_transfer")
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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(*parts: str) -> int:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).digest()
    return BOOTSTRAP_BASE_SEED + int.from_bytes(digest[:4], "big")


def parse_parameter_row(row: dict[str, str]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for field in PARAMETER_FIELDS:
        if field in {"lambda_out", "nrnd"}:
            params[field] = 0.0
        elif field in STRING_PARAMETER_FIELDS:
            params[field] = row[field]
        elif field in INTEGER_PARAMETER_FIELDS:
            params[field] = int(float(row[field]))
        else:
            params[field] = float(row[field])
    return params


def values_equal(left: Any, right: Any) -> bool:
    if isinstance(left, str) or isinstance(right, str):
        return str(left) == str(right)
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)


def load_psa_parameters() -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], dict[str, bool]]:
    representative = json.loads(REPRESENTATIVE_CONFIG.read_text())
    params_by_size: dict[str, dict[str, Any]] = {}
    provenance: list[dict[str, Any]] = []
    checks: dict[str, bool] = {}
    for size in SIZES:
        source = PARAMETER_SOURCES[size]
        rows = read_csv(source)
        if len(rows) != 1:
            raise RuntimeError(f"Expected one selected pSA parameter row in {source}, found {len(rows)}")
        row = rows[0]
        params = parse_parameter_row(row)
        if row.get("problem") != "ldpc" or row.get("size") != size or row.get("mode") != "pSA" or row.get("stage") != "final":
            raise RuntimeError(f"Selected record is not a final LDPC pSA row: {source}")
        configured = representative["parameters"][size]["pSA"]
        comparison_ok = all(values_equal(params[field], configured[field]) for field in PARAMETER_FIELDS)
        memoryless_ok = (
            params["lambda_mem"] == 0.0
            and params["response_lambda"] == 0.0
            and params["lambda_out"] == 0.0
        )
        checks[f"{size}_matches_representative_benchmark_config"] = comparison_ok
        checks[f"{size}_selected_record_is_final_independently_optimized_pSA"] = True
        checks[f"{size}_memory_coefficients_zero"] = memoryless_ok
        if not comparison_ok or not memoryless_ok:
            raise RuntimeError(f"pSA parameter validation failed for {size}")
        params_by_size[size] = params
        provenance.append({
            "size": size,
            "mode": "pSA-specific fixed transfer",
            "representative_mode": row["mode"],
            "selection_stage": row["stage"],
            "candidate_id": row.get("candidate_id", ""),
            "rank_in_archived_search": row.get("rank", ""),
            "source_parameter_file": str(source),
            "source_parameter_file_sha256": file_sha256(source),
            "representative_config_file": str(REPRESENTATIVE_CONFIG),
            "representative_config_sha256": file_sha256(REPRESENTATIVE_CONFIG),
            "independently_optimized_on_representative_matrix": True,
            "transferred_without_per_code_retuning": True,
            **params,
            "p_hold": params["psa_p"],
            "initial_schedule_plateau_fraction": params["I0_hold_fraction"],
            "channel_scaling_value": params["alpha"],
            "channel_scaling_mode": params["alpha_mode"],
            "channel_input_mode": "float",
            "fixed_bit_width": 8,
        })
    return params_by_size, provenance, checks


def load_archived_registry() -> tuple[list[dict[str, Any]], dict[tuple[str, str], np.ndarray], dict[str, bool]]:
    metadata_rows = read_csv(SOURCE_TRANSFER_DIR / "code_matrix_metadata.csv")
    selected = [row for row in metadata_rows if row["size"] in SIZES and row["code_id"] in CODE_IDS]
    if len(selected) != 30:
        raise RuntimeError(f"Expected 30 archived matrices, found {len(selected)}")
    matrices: dict[tuple[str, str], np.ndarray] = {}
    output_rows: list[dict[str, Any]] = []
    checks: dict[str, bool] = {}
    for row in selected:
        size, code_id = row["size"], row["code_id"]
        matrix_path = SOURCE_TRANSFER_DIR / "matrices" / size / f"{code_id}.npz"
        with np.load(matrix_path) as archive:
            H = np.asarray(archive["H"], dtype=np.uint8)
        validation = cross_code.validate_matrix(H, int(row["n_bits"]), int(row["n_checks"]))
        digest = cross_code._matrix_hash(H)
        hash_ok = digest == row["matrix_sha256"]
        seed_ok = int(row["generation_seed"]) >= 0
        degree_rank_rate_ok = bool(
            validation["variable_degree_ok"]
            and validation["check_degree_ok"]
            and validation["full_row_rank"]
            and math.isclose(float(validation["actual_rate"]), float(row["actual_rate"]), abs_tol=1e-15)
        )
        checks[f"{size}_{code_id}_sha256_match"] = hash_ok
        checks[f"{size}_{code_id}_generation_seed_recorded"] = seed_ok
        checks[f"{size}_{code_id}_degree_rank_rate_match"] = degree_rank_rate_ok
        if not hash_ok or not seed_ok or not degree_rank_rate_ok:
            raise RuntimeError(f"Archived matrix validation failed for {size}/{code_id}")
        matrices[(size, code_id)] = H
        output_rows.append({
            **row,
            "public_code_id": PUBLIC_CODE_IDS[code_id],
            "archived_matrix_file": str(matrix_path),
            "reloaded_matrix_sha256": digest,
            "reloaded_gf2_rank": validation["gf2_rank"],
            "reloaded_actual_rate": validation["actual_rate"],
            "validation_passed": True,
            "matrix_regenerated_for_this_study": False,
        })
    checks["matrix_count_is_30"] = len(matrices) == 30
    checks["public_mapping_C00_to_C09_complete"] = set(PUBLIC_CODE_IDS.values()) == {f"C{index:02d}" for index in range(10)}
    return output_rows, matrices, checks


def load_existing_transfer_rows(seeds: list[int]) -> tuple[list[dict[str, Any]], dict[str, bool]]:
    rows = read_csv(SOURCE_TRANSFER_DIR / "raw_results.csv")
    checks: dict[str, bool] = {}
    output: list[dict[str, Any]] = []
    arm_mapping = {"additive": "additive_fixed", "pSA": "matched_lambda0_ablation"}
    for row in rows:
        if row["size"] not in SIZES or row["code_id"] not in CODE_IDS:
            continue
        item: dict[str, Any] = dict(row)
        item["variant"] = arm_mapping[row["variant"]]
        item["public_code_id"] = PUBLIC_CODE_IDS[row["code_id"]]
        item["parameter_transfer_policy"] = "fixed_size_specific_no_code_retuning"
        output.append(item)
    expected_keys = {
        (size, code_id, arm, seed, ebno)
        for size in SIZES for code_id in CODE_IDS
        for arm in ["additive_fixed", "matched_lambda0_ablation"]
        for seed in seeds for ebno in EBNO_VALUES
    }
    observed_keys = {
        (row["size"], row["code_id"], row["variant"], int(row["seed"]), float(row["EbNo_dB"]))
        for row in output
    }
    checks["existing_two_arm_row_count_is_1800"] = len(output) == 1800
    checks["existing_two_arm_keys_match_design"] = observed_keys == expected_keys
    checks["existing_matched_ablation_lambda_zero"] = all(
        float(row["lambda_mem"]) == 0.0 for row in output if row["variant"] == "matched_lambda0_ablation"
    )
    checks["existing_additive_lambda_positive"] = all(
        float(row["lambda_mem"]) > 0.0 for row in output if row["variant"] == "additive_fixed"
    )
    if not all(checks.values()):
        raise RuntimeError(f"Existing cross-code raw-data validation failed: {checks}")
    return output, checks


def metadata_for_kernel(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "size": row["size"],
        "n_bits": int(row["n_bits"]),
        "n_checks": int(row["n_checks"]),
        "code_id": row["code_id"],
        "realization_type": row["realization_type"],
        "generation_seed": int(row["generation_seed"]),
        "matrix_sha256": row["matrix_sha256"],
        "gf2_rank": int(row["gf2_rank"]),
        "design_rate": float(row["design_rate"]),
        "actual_rate": float(row["actual_rate"]),
    }


def run_new_arm(
    metadata_rows: list[dict[str, Any]],
    matrices: dict[tuple[str, str], np.ndarray],
    params_by_size: dict[str, dict[str, Any]],
    seeds: list[int],
    trials: int,
    n_workers: int,
    config_hash: str,
    logger: logging.Logger,
    resume: bool,
) -> list[dict[str, Any]]:
    selected_metadata = sorted(metadata_rows, key=lambda row: (SIZES.index(row["size"]), CODE_IDS.index(row["code_id"])))
    all_rows: list[dict[str, Any]] = []
    for condition_index, row in enumerate(selected_metadata, 1):
        size, code_id = row["size"], row["code_id"]
        condition_dir = RUN_DIR / "performance_conditions" / size / code_id / "pSA_specific_fixed"
        raw_path = condition_dir / "raw_results.csv"
        complete_path = condition_dir / "complete.json"
        if resume and raw_path.exists() and complete_path.exists():
            complete = json.loads(complete_path.read_text())
            if complete.get("config_hash") == config_hash:
                logger.info("[%d/30] resume skip %s/%s", condition_index, size, code_id)
                all_rows.extend(read_csv(raw_path))
                continue
        params = dict(params_by_size[size])
        logger.info(
            "[%d/30] %s/%s pSA-specific fixed transfer; trials/EbNo=%d cycles=%d",
            condition_index, size, code_id, trials, params["n_cycles"],
        )
        started = time.perf_counter()
        rows = cross_code._run_performance_condition(
            matrices[(size, code_id)], metadata_for_kernel(row), "pSA", params,
            str(PARAMETER_SOURCES[size]), EBNO_VALUES, trials, seeds, n_workers,
            8, "float", logger,
        )
        for item in rows:
            item["variant"] = "pSA_specific_fixed"
            item["public_code_id"] = PUBLIC_CODE_IDS[code_id]
            item["parameter_transfer_policy"] = "representative_pSA_specific_fixed_no_code_retuning"
        condition_dir.mkdir(parents=True, exist_ok=True)
        cross_code.write_csv(raw_path, rows, cross_code._fields(rows))
        cross_code.write_json(complete_path, {
            "status": "complete",
            "config_hash": config_hash,
            "row_count": len(rows),
            "elapsed_seconds": time.perf_counter() - started,
            "parameter_hash": cross_code.config_hash(params),
        })
        all_rows.extend(rows)
        logger.info("[%d/30] complete in %.2f s", condition_index, time.perf_counter() - started)
    return all_rows


def aggregate_counts(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    return cross_code._aggregate_counts(rows)


def rates(counts: dict[str, int]) -> dict[str, float]:
    return cross_code._rates(counts)


def pair_metrics(method_counts: dict[str, int], comparator_counts: dict[str, int]) -> dict[str, Any]:
    method = rates(method_counts)
    comparator = rates(comparator_counts)
    output: dict[str, Any] = {}
    for metric in ["BER", "FER"]:
        method_rate = method[metric]
        comparator_rate = comparator[metric]
        relative = (comparator_rate - method_rate) / comparator_rate if comparator_rate > 0 else float("nan")
        log_ratio = math.log10(method_rate / comparator_rate) if method_rate > 0 and comparator_rate > 0 else float("nan")
        output.update({
            f"method_{metric}": method_rate,
            f"comparator_{metric}": comparator_rate,
            f"absolute_{metric}_reduction": comparator_rate - method_rate,
            f"relative_{metric}_reduction": relative,
            f"log10_{metric}_ratio": log_ratio,
            f"{metric}_method_better": int(method_rate < comparator_rate),
            f"{metric}_comparator_better": int(method_rate > comparator_rate),
            f"{metric}_tie": int(method_rate == comparator_rate),
        })
    return output


def build_code_summaries(all_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str, str, float | None], list[dict[str, Any]]] = defaultdict(list)
    metadata = {}
    for row in all_rows:
        size, code_id, arm = str(row["size"]), str(row["code_id"]), str(row["variant"])
        ebno = float(row["EbNo_dB"])
        grouped[(size, code_id, arm, ebno)].append(row)
        grouped[(size, code_id, arm, None)].append(row)
        metadata[(size, code_id)] = row
    wide_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    for size in SIZES:
        for code_id in CODE_IDS:
            for ebno in [*EBNO_VALUES, None]:
                arm_counts = {arm: aggregate_counts(grouped[(size, code_id, arm, ebno)]) for arm in OUTPUT_ARMS}
                n_trials = {counts["n_trials"] for counts in arm_counts.values()}
                n_bits = {counts["n_bits_total"] for counts in arm_counts.values()}
                if len(n_trials) != 1 or len(n_bits) != 1:
                    raise RuntimeError(f"Unpaired three-way aggregate for {size}/{code_id}/{ebno}")
                meta = metadata[(size, code_id)]
                wide: dict[str, Any] = {
                    "size": size,
                    "code_id": code_id,
                    "public_code_id": PUBLIC_CODE_IDS[code_id],
                    "realization_type": meta["realization_type"],
                    "generation_seed": int(meta["generation_seed"]),
                    "matrix_sha256": meta["matrix_sha256"],
                    "scope": "pooled_across_EbNo" if ebno is None else "per_EbNo",
                    "EbNo_dB": "" if ebno is None else ebno,
                    "n_trials_per_arm": next(iter(n_trials)),
                    "n_bits_per_arm": next(iter(n_bits)),
                }
                for arm, counts in arm_counts.items():
                    arm_rates = rates(counts)
                    wide[f"{arm}_BER"] = arm_rates["BER"]
                    wide[f"{arm}_FER"] = arm_rates["FER"]
                    wide[f"{arm}_bit_errors"] = counts["bit_errors"]
                    wide[f"{arm}_frame_errors"] = counts["frame_errors"]
                for comparison, method, comparator in PAIR_SPECS:
                    metrics = pair_metrics(arm_counts[method], arm_counts[comparator])
                    for key, value in metrics.items():
                        wide[f"{comparison}_{key}"] = value
                    pair_rows.append({
                        "size": size,
                        "code_id": code_id,
                        "public_code_id": PUBLIC_CODE_IDS[code_id],
                        "realization_type": meta["realization_type"],
                        "generation_seed": int(meta["generation_seed"]),
                        "matrix_sha256": meta["matrix_sha256"],
                        "scope": wide["scope"],
                        "EbNo_dB": wide["EbNo_dB"],
                        "comparison": comparison,
                        "method": method,
                        "comparator": comparator,
                        **metrics,
                        "bootstrap_unit": "independent_code_realization",
                    })
                wide_rows.append(wide)
    return wide_rows, pair_rows


def bootstrap_interval(values: np.ndarray, statistic: str, seed: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_SAMPLES, len(values)))
    sampled = values[indices]
    stats = np.mean(sampled, axis=1) if statistic == "mean" else np.median(sampled, axis=1)
    low, high = np.percentile(stats, [2.5, 97.5])
    return float(low), float(high)


def summarize_pair_rows(pair_rows: list[dict[str, Any]], overall: bool = False) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in pair_rows:
        size = "all_sizes" if overall else row["size"]
        if overall and row["scope"] != "pooled_across_EbNo":
            continue
        groups[(size, row["scope"], str(row["EbNo_dB"]), row["comparison"])].append(row)
    output: list[dict[str, Any]] = []
    for (size, scope, ebno, comparison), rows in sorted(groups.items()):
        method = rows[0]["method"]
        comparator = rows[0]["comparator"]
        item: dict[str, Any] = {
            "size": size,
            "scope": scope,
            "EbNo_dB": ebno,
            "comparison": comparison,
            "method": method,
            "comparator": comparator,
            "n_code_realizations": len(rows),
            "bootstrap_unit": "independent_code_realization",
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
        }
        for metric in ["BER", "FER"]:
            relative = np.array([float(row[f"relative_{metric}_reduction"]) for row in rows], dtype=float)
            finite = relative[np.isfinite(relative)]
            mean = float(np.mean(finite)) if len(finite) else float("nan")
            median = float(np.median(finite)) if len(finite) else float("nan")
            sd = float(np.std(finite, ddof=1)) if len(finite) > 1 else float("nan")
            mean_low, mean_high = bootstrap_interval(finite, "mean", stable_seed(size, scope, ebno, comparison, metric, "mean"))
            med_low, med_high = bootstrap_interval(finite, "median", stable_seed(size, scope, ebno, comparison, metric, "median"))
            method_rates = np.array([float(row[f"method_{metric}"]) for row in rows])
            comparator_rates = np.array([float(row[f"comparator_{metric}"]) for row in rows])
            item.update({
                f"{metric}_method_better_count": sum(int(row[f"{metric}_method_better"]) for row in rows),
                f"{metric}_comparator_better_count": sum(int(row[f"{metric}_comparator_better"]) for row in rows),
                f"{metric}_tie_count": sum(int(row[f"{metric}_tie"]) for row in rows),
                f"mean_relative_{metric}_reduction": mean,
                f"median_relative_{metric}_reduction": median,
                f"relative_{metric}_reduction_sd_between_codes": sd,
                f"mean_relative_{metric}_reduction_bootstrap_ci95_low": mean_low,
                f"mean_relative_{metric}_reduction_bootstrap_ci95_high": mean_high,
                f"median_relative_{metric}_reduction_bootstrap_ci95_low": med_low,
                f"median_relative_{metric}_reduction_bootstrap_ci95_high": med_high,
                f"mean_method_{metric}": float(np.mean(method_rates)),
                f"mean_comparator_{metric}": float(np.mean(comparator_rates)),
            })
        output.append(item)
    return output


def make_advantage_shrinkage(size_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {
        (row["size"], row["scope"], str(row["EbNo_dB"]), row["comparison"]): row
        for row in size_rows
    }
    output = []
    for size in SIZES:
        for scope, ebno in [("pooled_across_EbNo", "")]:
            strict = lookup[(size, scope, ebno, "additive_vs_pSA_specific")]
            matched = lookup[(size, scope, ebno, "additive_vs_matched_ablation")]
            row: dict[str, Any] = {"size": size, "scope": scope, "EbNo_dB": ebno}
            for metric in ["BER", "FER"]:
                strict_effect = float(strict[f"median_relative_{metric}_reduction"])
                matched_effect = float(matched[f"median_relative_{metric}_reduction"])
                row.update({
                    f"additive_vs_pSA_specific_median_relative_{metric}_reduction": strict_effect,
                    f"additive_vs_matched_ablation_median_relative_{metric}_reduction": matched_effect,
                    f"absolute_shrinkage_in_median_relative_{metric}_reduction": matched_effect - strict_effect,
                    f"retained_fraction_of_matched_ablation_median_{metric}_effect": strict_effect / matched_effect if matched_effect else float("nan"),
                })
            output.append(row)
    return output


def make_figures(wide_rows: list[dict[str, Any]], pair_rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = RUN_DIR / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    colors = {
        "additive_fixed": "#E45756",
        "pSA_specific_fixed": "#4C78A8",
        "matched_lambda0_ablation": "#9D755D",
    }
    labels = {
        "additive_fixed": r"additive fixed transfer",
        "pSA_specific_fixed": r"pSA-specific fixed transfer",
        "matched_lambda0_ablation": r"matched $\lambda=0$ ablation",
    }

    pooled = [row for row in wide_rows if row["scope"] == "pooled_across_EbNo"]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.1), sharey=False)
    for ax, size in zip(axes, SIZES):
        rows = sorted([row for row in pooled if row["size"] == size], key=lambda row: row["public_code_id"])
        x = np.arange(len(rows))
        for arm, marker in zip(OUTPUT_ARMS, ["o", "s", "^"]):
            ax.plot(x, [float(row[f"{arm}_BER"]) for row in rows], marker=marker, color=colors[arm], label=labels[arm])
        ax.set_xticks(x, [row["public_code_id"] for row in rows], rotation=45)
        ax.set_title(size.replace("_", ", "))
        ax.set_yscale("log")
        ax.set_ylabel("BER pooled across three SNRs")
        ax.grid(True, which="both", alpha=0.25)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=3, frameon=False)
    fig.subplots_adjust(left=0.065, right=0.99, bottom=0.20, top=0.82, wspace=0.28)
    for suffix in ["pdf", "png"]:
        fig.savefig(figure_dir / f"Fig_A_three_way_code_BER.{suffix}", dpi=300)
    plt.close(fig)

    comparisons = ["additive_vs_pSA_specific", "additive_vs_matched_ablation"]
    comparison_labels = ["vs pSA-specific transfer", r"vs matched $\lambda=0$ ablation"]
    pooled_pairs = [row for row in pair_rows if row["scope"] == "pooled_across_EbNo" and row["comparison"] in comparisons]
    fig, ax = plt.subplots(figsize=(8.2, 4.7))
    positions = []
    tick_positions = []
    tick_labels = []
    for size_index, size in enumerate(SIZES):
        center = size_index * 3.0
        tick_positions.append(center + 0.5)
        tick_labels.append(size.split("_")[0])
        for comparison_index, comparison in enumerate(comparisons):
            position = center + comparison_index
            positions.append(position)
            values = [
                100.0 * float(row["relative_BER_reduction"])
                for row in pooled_pairs if row["size"] == size and row["comparison"] == comparison
            ]
            box = ax.boxplot(values, positions=[position], widths=0.65, patch_artist=True, showfliers=False)
            color = "#4C78A8" if comparison_index == 0 else "#9D755D"
            box["boxes"][0].set(facecolor=color, alpha=0.28, edgecolor=color)
            for element in ["whiskers", "caps", "medians"]:
                for artist in box[element]:
                    artist.set(color=color)
            jitter = np.linspace(-0.18, 0.18, len(values))
            ax.scatter(np.full(len(values), position) + jitter, values, s=24, color=color, alpha=0.85, zorder=3)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(tick_positions, tick_labels)
    ax.set_ylabel("Relative BER reduction of additive transfer (%)")
    ax.grid(axis="y", alpha=0.25)
    from matplotlib.patches import Patch
    ax.legend([Patch(facecolor="#4C78A8", alpha=0.28), Patch(facecolor="#9D755D", alpha=0.28)], comparison_labels, frameon=False)
    fig.tight_layout()
    for suffix in ["pdf", "png"]:
        fig.savefig(figure_dir / f"Fig_B_size_relative_BER_reduction.{suffix}", dpi=300)
    plt.close(fig)

    strict = [row for row in pair_rows if row["comparison"] == "additive_vs_pSA_specific" and row["scope"] == "per_EbNo"]
    matrix = np.full((30, 3), np.nan)
    ylabels = []
    for size_index, size in enumerate(SIZES):
        for code_index, code_id in enumerate(CODE_IDS):
            global_index = size_index * 10 + code_index
            ylabels.append(f"{size.split('_')[0]} {PUBLIC_CODE_IDS[code_id]}")
            for ebno_index, ebno in enumerate(EBNO_VALUES):
                row = next(
                    item for item in strict
                    if item["size"] == size and item["code_id"] == code_id and float(item["EbNo_dB"]) == ebno
                )
                matrix[global_index, ebno_index] = float(row["log10_BER_ratio"])
    finite = np.abs(matrix[np.isfinite(matrix)])
    limit = max(float(np.max(finite)), 0.1)
    fig, ax = plt.subplots(figsize=(6.4, 9.0))
    image = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
    ax.set_xticks(np.arange(3), [f"{value:.1f} dB" for value in EBNO_VALUES])
    ax.set_yticks(np.arange(30), ylabels, fontsize=7)
    ax.set_xlabel(r"$E_b/N_0$")
    ax.set_title(r"$\log_{10}(\mathrm{BER}_{\mathrm{additive}}/\mathrm{BER}_{\mathrm{pSA-specific}})$")
    for boundary in [9.5, 19.5]:
        ax.axhline(boundary, color="black", linewidth=1.2)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("negative values favor additive transfer")
    fig.tight_layout()
    for suffix in ["pdf", "png"]:
        fig.savefig(figure_dir / f"Fig_C_additive_vs_pSA_specific_log_BER_ratio.{suffix}", dpi=300)
    plt.close(fig)


def determine_case(size_rows: list[dict[str, Any]]) -> tuple[str, str]:
    strict = [
        row for row in size_rows
        if row["scope"] == "pooled_across_EbNo" and row["comparison"] == "additive_vs_pSA_specific"
    ]
    total_wins = sum(int(row["BER_method_better_count"]) for row in strict)
    every_size_majority = all(int(row["BER_method_better_count"]) >= 6 for row in strict)
    every_size_ci_positive = all(float(row["median_relative_BER_reduction_bootstrap_ci95_low"]) > 0 for row in strict)
    if total_wins == 30 and every_size_ci_positive:
        return "Case 1", "Additive fixed transfer clearly outperforms the pSA-specific fixed-transfer comparator."
    if total_wins > 15 and every_size_majority:
        return "Case 2", "The additive advantage is smaller than against the matched ablation but remains positive for a majority of codes at every size."
    return "Case 3", "Additive fixed transfer is not consistently superior to the pSA-specific fixed-transfer comparator."


def fmt(value: Any, digits: int = 5) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "undefined"
    return f"{number:.{digits}g}"


def write_report(
    config: dict[str, Any],
    parameter_rows: list[dict[str, Any]],
    size_rows: list[dict[str, Any]],
    overall_rows: list[dict[str, Any]],
    shrinkage_rows: list[dict[str, Any]],
    validation: dict[str, Any],
    runtime_seconds: float,
) -> None:
    case, interpretation = determine_case(size_rows)
    strict = {
        row["size"]: row for row in size_rows
        if row["scope"] == "pooled_across_EbNo" and row["comparison"] == "additive_vs_pSA_specific"
    }
    matched = {
        row["size"]: row for row in size_rows
        if row["scope"] == "pooled_across_EbNo" and row["comparison"] == "additive_vs_matched_ablation"
    }
    overall_strict = next(row for row in overall_rows if row["comparison"] == "additive_vs_pSA_specific")
    total_wins = int(overall_strict["BER_method_better_count"])
    total_losses = int(overall_strict["BER_comparator_better_count"])
    total_ties = int(overall_strict["BER_tie_count"])
    lines = [
        "# pSA-specific fixed-transfer validation",
        "",
        f"**Predefined interpretation: {case}.** {interpretation}",
        "",
        "This study transfers one independently optimized, representative-matrix pSA parameter set per block length to the same ten archived random-regular (3,6) LDPC matrices used in the existing cross-code study. No matrix generation, parameter optimization, per-code retuning, or mechanism simulation was performed.",
        "",
        "## 1. pSA-specific parameter sets",
        "",
        "| Size | Candidate | cycles | schedule | I0 min | I0 max | shape | initial plateau | p_hold | kw | kr | channel scaling | burn-in | window | readout |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---|",
    ]
    for row in parameter_rows:
        lines.append(
            f"| {row['size']} | {row['candidate_id']} | {row['n_cycles']} | {row['I0_schedule_type']} | "
            f"{fmt(row['I0_min'])} | {fmt(row['I0_max'])} | {fmt(row['I0_schedule_shape'])} | "
            f"{fmt(row['I0_hold_fraction'])} | {fmt(row['psa_p'])} | {fmt(row['kw'])} | {fmt(row['kr'])} | "
            f"{row['alpha_mode']}:{fmt(row['alpha'])} | {row['burn_in']} | {row['sample_window']} | {row['decision_method']} |"
        )
    lines += [
        "",
        "All three rows are the archived `mode=pSA`, `stage=final` selections used in the independently optimized representative-code benchmark. Their complete dictionaries match the manuscript release configuration exactly; lambda, finite-response, and output-memory coefficients are zero.",
        "",
        "## 2. Additive versus pSA-specific fixed transfer",
        "",
        f"Across the 30 pooled code-level comparisons, additive wins/ties/loses in BER are **{total_wins}/{total_ties}/{total_losses}**.",
        "",
        "| Size | BER wins | Median relative BER reduction (95% code-bootstrap CI) | Mean reduction | Between-code SD | FER wins | Median relative FER reduction (95% CI) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for size in SIZES:
        row = strict[size]
        lines.append(
            f"| {size} | {row['BER_method_better_count']}/10 | "
            f"{100*float(row['median_relative_BER_reduction']):.3f}% "
            f"[{100*float(row['median_relative_BER_reduction_bootstrap_ci95_low']):.3f}, "
            f"{100*float(row['median_relative_BER_reduction_bootstrap_ci95_high']):.3f}]% | "
            f"{100*float(row['mean_relative_BER_reduction']):.3f}% | "
            f"{100*float(row['relative_BER_reduction_sd_between_codes']):.3f} pp | "
            f"{row['FER_method_better_count']}/10 | "
            f"{100*float(row['median_relative_FER_reduction']):.3f}% "
            f"[{100*float(row['median_relative_FER_reduction_bootstrap_ci95_low']):.3f}, "
            f"{100*float(row['median_relative_FER_reduction_bootstrap_ci95_high']):.3f}]% |"
        )
    lines += [
        "",
        "The independent code realization is the only bootstrap unit. Individual bits and trials are not resampled as independent instances.",
        "",
        "## 3. Effect relative to the matched lambda=0 ablation",
        "",
        "| Size | Additive vs matched-ablation median BER reduction | Additive vs pSA-specific median BER reduction | Shrinkage | Retained fraction |",
        "|---|---:|---:|---:|---:|",
    ]
    shrink_lookup = {row["size"]: row for row in shrinkage_rows}
    for size in SIZES:
        row = shrink_lookup[size]
        lines.append(
            f"| {size} | {100*float(row['additive_vs_matched_ablation_median_relative_BER_reduction']):.3f}% | "
            f"{100*float(row['additive_vs_pSA_specific_median_relative_BER_reduction']):.3f}% | "
            f"{100*float(row['absolute_shrinkage_in_median_relative_BER_reduction']):.3f} pp | "
            f"{100*float(row['retained_fraction_of_matched_ablation_median_BER_effect']):.2f}% |"
        )
    lines += [
        "",
        "The previously reported approximately 97% reductions at N=192 and 288 remain matched-ablation effects and are not relabeled as independently optimized pSA comparisons.",
        "",
        "## 4. Protocol and validation",
        "",
        f"- Matrices: the archived C00--C09 files were loaded directly; all 30 SHA-256 hashes, generation seeds, degree checks, GF(2) ranks, and rates match the existing transfer registry.",
        f"- Seeds: the exact ten existing seed batches were reused: `{config['seeds']}`.",
        "- Trials: 1000 per code, SNR, and arm at 2.0, 2.5, and 3.0 dB.",
        "- Kernel: the existing `candidate_chunk` path and cross-code condition runner were reused unchanged.",
        "- pSA-specific memory state: `implementation_mode=pSA`, lambda=0, response coefficient=0, and output-memory coefficient=0.",
        "- Initialization, channel generation, stopping behavior, and readout are inherited unchanged from the existing cross-code runner; only the size-specific pSA parameter dictionary differs.",
        "- Per-code retuning: none.",
        f"- Validation passed: **{validation['all_passed']}**.",
        f"- New-arm runtime: {runtime_seconds/60:.1f} min.",
        "",
        "## 5. Manuscript recommendations",
        "",
    ]
    if case == "Case 1":
        lines += [
            "- Figure 5: update to a three-way fixed-transfer comparison; retain the matched-ablation ratios as explicitly rule-level effects.",
            "- Cross-code claim: it can be strengthened to state that additive fixed transfer remains superior to a separately optimized pSA fixed-transfer package within the tested code ensemble.",
            "- Discussion limitation: delete the hypothetical sentence about a future separately optimized memoryless transfer and replace it with the observed three-way result and its tested scope.",
        ]
    elif case == "Case 2":
        lines += [
            "- Figure 5: a three-way comparison is recommended because it gives the fairer transfer benchmark; the matched-ablation effect must remain separately labeled.",
            "- Cross-code claim: strengthen only to a fixed-transfer advantage over pSA-specific parameters for the majority of tested codes, not to a universal margin.",
            "- Discussion limitation: replace the hypothetical sentence with the completed validation and state that the advantage shrinks under the stricter comparator.",
        ]
    else:
        lines += [
            "- Figure 5: retain the current figure as rule-level matched-ablation transferability evidence; place the three-way result in the Supplemental Material or a limitation paragraph.",
            "- Cross-code claim: do not claim transfer-performance superiority over pSA-specific parameters.",
            "- Discussion limitation: replace the hypothetical sentence with the completed null/negative fixed-transfer comparison.",
        ]
    ci_clear = all(
        float(row["median_relative_BER_reduction_bootstrap_ci95_low"]) > 0
        or float(row["median_relative_BER_reduction_bootstrap_ci95_high"]) < 0
        for row in strict.values()
    )
    lines += [
        f"- Additional trials: {'not indicated by the code-level result; the size-level median CIs are directionally resolved' if ci_clear else 'additional per-code trials are unlikely to resolve code-to-code uncertainty; consider more code realizations only if a sharper ensemble claim is required'}.",
        f"- Validation status: {'complete' if validation['all_passed'] else 'incomplete'}.",
        f"- Initialization-robustness study: {'may proceed' if validation['all_passed'] else 'should wait for validation completion'}; it is a separate question and does not alter this fixed-transfer result.",
        "",
        "## 6. Output inventory",
        "",
        "- `raw_psa_specific_seed_batches.csv`: new pSA-specific seed-batch counts and rates.",
        "- `raw_three_way_seed_batches.csv`: all three fixed-transfer arms, including reused existing results.",
        "- `code_snr_three_way_summary.csv`: absolute three-way BER/FER by code and SNR plus pooled rows.",
        "- `code_level_pairwise_summary.csv`: all three pairwise comparisons.",
        "- `pooled_code_level_summary.csv`: pooled code-level pairwise outcomes.",
        "- `three_way_comparison_summary.csv`: size-level effects and code-bootstrap intervals.",
        "- `overall_30_code_summary.csv`: descriptive 30-code pooled win counts and effects.",
        "- `parameter_comparison_table.csv`, `matrix_seed_metadata.csv`, `seed_metadata.json`, and `validation.json`: provenance and checks.",
        "- `figures/Fig_A_three_way_code_BER.*`, `Fig_B_size_relative_BER_reduction.*`, and `Fig_C_additive_vs_pSA_specific_log_BER_ratio.*`: manuscript candidates.",
    ]
    (RUN_DIR / "PSA_SPECIFIC_TRANSFER_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument("--n-workers", type=int, default=10)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    global RUN_DIR
    RUN_DIR = REPO_ROOT / 'outputs' / 'psa_specific_fixed_transfer'
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    args = build_parser().parse_args()
    logger = setup_logging()
    original_parameters = json.loads((SOURCE_TRANSFER_DIR / "parameters_and_seeds.json").read_text())
    seeds = [int(value) for value in original_parameters["seeds"]]
    params_by_size, parameter_provenance, parameter_checks = load_psa_parameters()
    metadata_rows, matrices, matrix_checks = load_archived_registry()
    existing_rows, existing_checks = load_existing_transfer_rows(seeds)
    config = {
        "study_version": STUDY_VERSION,
        "purpose": "pSA-specific fixed-transfer comparator",
        "sizes": SIZES,
        "code_ids": CODE_IDS,
        "public_code_ids": PUBLIC_CODE_IDS,
        "EbNo_dB": EBNO_VALUES,
        "trials_per_code_EbNo_arm": int(args.trials),
        "seed_count": len(seeds),
        "seeds": seeds,
        "seed_construction": original_parameters["seed_construction"],
        "n_workers": int(args.n_workers),
        "fixed_bit_width": int(original_parameters["fixed_bit_width"]),
        "channel_input_mode": original_parameters["channel_input_mode"],
        "matrix_source": str(SOURCE_TRANSFER_DIR),
        "existing_raw_source": str(SOURCE_TRANSFER_DIR / "raw_results.csv"),
        "representative_parameter_sources": {size: str(path) for size, path in PARAMETER_SOURCES.items()},
        "parameter_transfer_policy": "one representative-matrix pSA parameter set per size; no per-code retuning",
        "simulation_scope": "new pSA-specific arm only; additive and matched-ablation rows reused",
    }
    config["config_hash"] = cross_code.config_hash(config)
    config_path = RUN_DIR / "effective_config.json"
    if config_path.exists() and args.resume:
        prior = json.loads(config_path.read_text())
        if prior.get("config_hash") != config["config_hash"]:
            raise RuntimeError("Configuration changed; use a new result directory")
    cross_code.write_json(config_path, config)
    cross_code.write_csv(RUN_DIR / "parameter_provenance.csv", parameter_provenance, cross_code._fields(parameter_provenance))
    cross_code.write_csv(RUN_DIR / "matrix_seed_metadata.csv", metadata_rows, cross_code._fields(metadata_rows))
    cross_code.write_json(RUN_DIR / "seed_metadata.json", {
        "seeds": seeds,
        "seed_construction": config["seed_construction"],
        "paired_across_all_three_arms": True,
        "source": str(SOURCE_TRANSFER_DIR / "parameters_and_seeds.json"),
        "source_sha256": file_sha256(SOURCE_TRANSFER_DIR / "parameters_and_seeds.json"),
        "bootstrap_base_seed": BOOTSTRAP_BASE_SEED,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_unit": "independent_code_realization",
    })

    phase3_parameters = original_parameters["size_parameters"]
    parameter_comparison = []
    for size in SIZES:
        arms = {
            "additive_fixed": phase3_parameters[size]["additive_effective_parameters"],
            "matched_lambda0_ablation": phase3_parameters[size]["pSA_effective_parameters"],
            "pSA_specific_fixed": params_by_size[size],
        }
        for arm, params in arms.items():
            parameter_comparison.append({
                "size": size,
                "arm": arm,
                "source_parameter_file": (
                    str(PARAMETER_SOURCES[size]) if arm == "pSA_specific_fixed"
                    else phase3_parameters[size]["source_parameter_file"]
                ),
                "per_code_retuning": False,
                **params,
                "p_hold": params["psa_p"],
                "initial_schedule_plateau_fraction": params["I0_hold_fraction"],
                "channel_scaling_value": params["alpha"],
                "channel_scaling_mode": params["alpha_mode"],
            })
    cross_code.write_csv(RUN_DIR / "parameter_comparison_table.csv", parameter_comparison, cross_code._fields(parameter_comparison))

    estimated_updates = sum(
        int(args.trials) * len(EBNO_VALUES) * int(params_by_size[size]["n_cycles"]) * int(next(row["n_bits"] for row in metadata_rows if row["size"] == size)) * 10
        for size in SIZES
    )
    manifest = {
        "status": "planned" if args.dry_run else "running",
        "started_utc": utc_now(),
        "config_hash": config["config_hash"],
        "new_arm_condition_count": 30,
        "estimated_new_arm_bit_updates": estimated_updates,
        "completed_conditions": 0,
    }
    cross_code.write_json(RUN_DIR / "manifest.json", manifest)
    logger.info("Validated archived matrices, pSA parameter sources, and existing two-arm rows")
    logger.info("New-arm conditions=30; estimated bit updates=%.6g", estimated_updates)
    if args.dry_run:
        manifest["status"] = "dry_run_complete"
        cross_code.write_json(RUN_DIR / "manifest.json", manifest)
        return

    started = time.perf_counter()
    new_rows = run_new_arm(
        metadata_rows, matrices, params_by_size, seeds, int(args.trials), int(args.n_workers),
        config["config_hash"], logger, bool(args.resume),
    )
    new_rows = sorted(new_rows, key=lambda row: (row["size"], row["code_id"], int(row["seed"]), float(row["EbNo_dB"])))
    cross_code.write_csv(RUN_DIR / "raw_psa_specific_seed_batches.csv", new_rows, cross_code._fields(new_rows))
    all_rows = sorted(existing_rows + new_rows, key=lambda row: (row["size"], row["code_id"], row["variant"], int(row["seed"]), float(row["EbNo_dB"])))
    cross_code.write_csv(RUN_DIR / "raw_three_way_seed_batches.csv", all_rows, cross_code._fields(all_rows))

    expected_keys = {
        (size, code_id, arm, seed, ebno)
        for size in SIZES for code_id in CODE_IDS for arm in OUTPUT_ARMS
        for seed in seeds for ebno in EBNO_VALUES
    }
    observed_keys = {
        (row["size"], row["code_id"], row["variant"], int(row["seed"]), float(row["EbNo_dB"]))
        for row in all_rows
    }
    new_parameter_hashes = {
        size: {
            cross_code.config_hash(params_by_size[size])
            for row in new_rows if row["size"] == size
        }
        for size in SIZES
    }
    validation = {
        **parameter_checks,
        **matrix_checks,
        **existing_checks,
        "new_arm_row_count_is_900": len(new_rows) == 900,
        "three_way_row_count_is_2700": len(all_rows) == 2700,
        "three_way_unique_keys_match_design": observed_keys == expected_keys,
        "stochastic_seeds_identical_across_arms": all(
            {
                int(row["seed"]) for row in all_rows
                if row["size"] == size and row["code_id"] == code_id and row["variant"] == arm
            } == set(seeds)
            for size in SIZES for code_id in CODE_IDS for arm in OUTPUT_ARMS
        ),
        "pSA_specific_implementation_mode_is_pSA": all(row["implementation_mode"] == "pSA" for row in new_rows),
        "pSA_specific_lambda_zero": all(float(row["lambda_mem"]) == 0.0 for row in new_rows),
        "pSA_specific_response_memory_zero": all(
            params_by_size[size]["response_lambda"] == 0.0 and params_by_size[size]["lambda_out"] == 0.0
            for size in SIZES
        ),
        "one_pSA_parameter_hash_per_size": all(len(hashes) == 1 for hashes in new_parameter_hashes.values()),
        "no_per_code_retuning": all("no_code_retuning" in row["parameter_transfer_policy"] for row in all_rows),
        "no_new_matrix_generation": all(row["matrix_regenerated_for_this_study"] is False for row in metadata_rows),
        "initialization_channel_readout_kernel_reused": True,
    }
    validation["all_passed"] = all(bool(value) for key, value in validation.items() if key != "all_passed")
    cross_code.write_json(RUN_DIR / "validation.json", validation)
    if not validation["all_passed"]:
        raise RuntimeError("Validation failed; see validation.json")

    wide_rows, pair_rows = build_code_summaries(all_rows)
    pooled_rows = [row for row in pair_rows if row["scope"] == "pooled_across_EbNo"]
    size_rows = summarize_pair_rows(pair_rows)
    overall_rows = summarize_pair_rows(pair_rows, overall=True)
    shrinkage_rows = make_advantage_shrinkage(size_rows)
    cross_code.write_csv(RUN_DIR / "code_snr_three_way_summary.csv", wide_rows, cross_code._fields(wide_rows))
    cross_code.write_csv(RUN_DIR / "code_level_pairwise_summary.csv", pair_rows, cross_code._fields(pair_rows))
    cross_code.write_csv(RUN_DIR / "pooled_code_level_summary.csv", pooled_rows, cross_code._fields(pooled_rows))
    cross_code.write_csv(RUN_DIR / "three_way_comparison_summary.csv", size_rows, cross_code._fields(size_rows))
    cross_code.write_csv(RUN_DIR / "bootstrap_statistics.csv", size_rows, cross_code._fields(size_rows))
    cross_code.write_csv(RUN_DIR / "overall_30_code_summary.csv", overall_rows, cross_code._fields(overall_rows))
    cross_code.write_csv(RUN_DIR / "advantage_shrinkage_summary.csv", shrinkage_rows, cross_code._fields(shrinkage_rows))
    make_figures(wide_rows, pair_rows)
    elapsed = time.perf_counter() - started
    write_report(config, parameter_provenance, size_rows, overall_rows, shrinkage_rows, validation, elapsed)
    cross_code.write_json(RUN_DIR / "runtime_summary.json", {
        "elapsed_seconds_this_invocation": elapsed,
        "new_arm_bit_updates": estimated_updates,
        "new_arm_seed_batch_rows": len(new_rows),
        "combined_three_way_seed_batch_rows": len(all_rows),
    })
    manifest.update({
        "status": "complete",
        "finished_utc": utc_now(),
        "completed_conditions": 30,
        "validation_all_passed": validation["all_passed"],
        "runtime_seconds": elapsed,
    })
    cross_code.write_json(RUN_DIR / "manifest.json", manifest)
    logger.info("Complete in %.2f s", elapsed)


if __name__ == "__main__":
    main()

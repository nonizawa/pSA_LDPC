#!/usr/bin/env python3
"""Matched-parameter Phase 1 controls for the LDPC lambda-pSA claim.

The production paper runner and its pSA/lambda-pSA/tau-pSA definitions are left
unchanged.  This driver loads completed LDPC parameter sets and evaluates three
one-factor-at-a-time controls with common seeds:

  additive:   q(t) + lambda q(t-1)              (the published lambda-pSA rule)
  normalized: [q(t) + lambda q(t-1)]/(1+lambda)
  gain_only:  (1+lambda) q(t)                   (no temporal memory)

For additive lambda <= 0.95 the existing lambda_pSA implementation is called
directly.  The dedicated extended mode is used only above the old search bound.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import platform
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
LDPC_DIR = REPO_ROOT / "src" / "ldpc"
if str(LDPC_DIR) not in sys.path:
    sys.path.insert(0, str(LDPC_DIR))

from auto_match_bp import PARAM_FIELDS, evaluate_candidate_total_trials  # noqa: E402
from ldpc_pbit import random_regular_ldpc_parity_check, update_pbits  # noqa: E402


SIZE_SPECS = {
    "N96_M48": {"n_bits": 96, "n_checks": 48},
    "N192_M96": {"n_bits": 192, "n_checks": 96},
    "N288_M144": {"n_bits": 288, "n_checks": 144},
}

PROFILE_DEFAULTS = {
    "smoke": {
        "sizes": ["N96_M48"],
        "lambda_values": [0.0, 0.5, 0.95, 1.10],
        "trials": 2,
        "seed_count": 1,
        "cycles_override": 5,
    },
    "pilot": {
        "sizes": list(SIZE_SPECS),
        "lambda_values": [0.0, 0.50, 0.80, 0.90, 0.95, 1.00, 1.10, 1.25],
        "trials": 200,
        "seed_count": 4,
        "cycles_override": None,
    },
    "full": {
        "sizes": list(SIZE_SPECS),
        "lambda_values": [
            0.0,
            0.25,
            0.50,
            0.70,
            0.80,
            0.85,
            0.90,
            0.92,
            0.94,
            0.95,
            0.975,
            1.00,
            1.05,
            1.10,
            1.20,
            1.40,
        ],
        "trials": 2000,
        "seed_count": 8,
        "cycles_override": None,
    },
}

VARIANT_MODES = {
    "normalized": "lambda_pSA_normalized",
    "gain_only": "gain_only_pSA",
}

SUMMARY_FIELDS = [
    "size",
    "anchor",
    "variant",
    "implementation_mode",
    "lambda_mem",
    "boundary_extension",
    "formula",
    "score",
    "ber_score",
    "fer_score",
    "syndrome_penalty",
    "mean_BER",
    "mean_FER",
    "mean_syndrome_violation",
    "evaluation_trials_per_ebno",
    "evaluation_seeds",
    "elapsed_seconds",
    "source_parameter_file",
    *[field for field in PARAM_FIELDS if field != "lambda_mem"],
]

BY_EBNO_FIELDS = [
    "size",
    "anchor",
    "variant",
    "implementation_mode",
    "lambda_mem",
    "EbNo_dB",
    "BER",
    "FER",
    "syndrome_violation_rate",
    "avg_syndrome_weight",
    "BP_BER",
    "BP_FER",
    "n_trials",
    "n_bits_total",
]

BY_SEED_FIELDS = [
    "size",
    "anchor",
    "variant",
    "implementation_mode",
    "lambda_mem",
    "seed",
    "EbNo_dB",
    "BER",
    "FER",
    "syndrome_violation_rate",
    "avg_syndrome_weight",
    "BP_BER",
    "BP_FER",
    "n_trials",
    "n_bits_total",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_list(value: str, cast=str) -> list[Any]:
    return [cast(item.strip()) for item in str(value).split(",") if item.strip()]


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_ready(data), indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def setup_logging(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger("phase1_controls")
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


def result_roots(extra_roots: list[str] | None) -> list[Path]:
    if extra_roots:
        return [Path(item).expanduser().resolve() for item in extra_roots]
    return [
        REPO_ROOT / "configs" / "ldpc" / "benchmark_sources",
        REPO_ROOT / "paper_results",
        REPO_ROOT / "paper_results_trip",
        REPO_ROOT / "paper_results_N96_M48_full_20260619",
    ]


def locate_completed_size(size: str, roots: list[Path]) -> Path:
    required = [
        Path("comparison/bp_baseline.csv"),
        Path("pSA/best_params.csv"),
        Path("lambda_pSA/memory_sweep_summary.csv"),
    ]
    for root in roots:
        candidate = root / "ldpc" / size
        if all((candidate / relative).exists() for relative in required):
            return candidate
    searched = "\n".join(str(root / "ldpc" / size) for root in roots)
    raise FileNotFoundError(f"No complete source result for {size}. Searched:\n{searched}")


def parse_param_row(row: dict[str, str]) -> dict[str, Any]:
    integer_fields = {"burn_in", "sample_window", "n_cycles"}
    text_fields = {"alpha_mode", "I0_schedule_type", "decision_method"}
    params: dict[str, Any] = {}
    for field in PARAM_FIELDS:
        value = row.get(field, "")
        if field in text_fields:
            params[field] = value
        elif field in integer_fields:
            params[field] = int(float(value))
        else:
            params[field] = float(value or 0.0)
    return params


def load_anchor(source_dir: Path, anchor: str) -> tuple[dict[str, Any], Path]:
    if anchor == "pSA":
        source_path = source_dir / "pSA" / "best_params.csv"
        rows = read_csv(source_path)
        if not rows:
            raise ValueError(f"Empty parameter source: {source_path}")
        row = min(rows, key=lambda item: float(item.get("score") or "inf"))
    elif anchor == "lambda":
        source_path = source_dir / "lambda_pSA" / "memory_sweep_summary.csv"
        rows = read_csv(source_path)
        if not rows:
            raise ValueError(f"Empty parameter source: {source_path}")
        row = min(rows, key=lambda item: float(item.get("score") or "inf"))
    else:
        raise ValueError(f"Unknown anchor: {anchor}")
    return parse_param_row(row), source_path


def load_reference(source_dir: Path) -> list[dict[str, Any]]:
    path = source_dir / "comparison" / "bp_baseline.csv"
    rows = read_csv(path)
    if not rows:
        raise ValueError(f"Empty BP reference: {path}")
    numeric_fields = {
        "EbNo_dB",
        "BP_BER",
        "BP_FER",
        "BP_convergence_rate",
        "BP_avg_iterations",
    }
    integer_fields = {"n_trials", "n_bits_total"}
    parsed = []
    for row in rows:
        out: dict[str, Any] = {}
        for key, value in row.items():
            if key in numeric_fields:
                out[key] = float(value)
            elif key in integer_fields:
                out[key] = int(float(value))
            else:
                out[key] = value
        parsed.append(out)
    return parsed


def formula_for(variant: str) -> str:
    return {
        "pSA": "q_t + xi_t",
        "additive": "q_t + lambda*q_(t-1) + xi_t",
        "normalized": "(q_t + lambda*q_(t-1))/(1+lambda) + xi_t",
        "gain_only": "(1+lambda)*q_t + xi_t",
    }[variant]


def implementation_mode(variant: str, lambda_mem: float) -> str:
    if variant == "pSA":
        return "pSA"
    if variant == "additive":
        return "lambda_pSA" if lambda_mem <= 0.95 + 1e-12 else "lambda_pSA_extended"
    return VARIANT_MODES[variant]


def condition_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row["size"]),
        str(row["anchor"]),
        str(row["variant"]),
        f"{float(row['lambda_mem']):.12g}",
    )


def make_conditions(
    sizes: list[str], anchors: list[str], variants: list[str], lambda_values: list[float]
) -> list[dict[str, Any]]:
    conditions = []
    for size in sizes:
        for anchor in anchors:
            conditions.append({"size": size, "anchor": anchor, "variant": "pSA", "lambda_mem": 0.0})
            for variant in variants:
                for value in lambda_values:
                    conditions.append(
                        {"size": size, "anchor": anchor, "variant": variant, "lambda_mem": float(value)}
                    )
    return conditions


def adjusted_params(base: dict[str, Any], lambda_mem: float, cycles_override: int | None) -> dict[str, Any]:
    params = dict(base)
    params["lambda_mem"] = float(lambda_mem)
    params["response_lambda"] = 0.0
    params["lambda_out"] = 0.0
    params["nrnd"] = 0.0
    if cycles_override is not None:
        params["n_cycles"] = int(cycles_override)
        params["burn_in"] = 0
        params["sample_window"] = 0
        params["decision_method"] = "last"
    else:
        params["burn_in"] = min(int(params["burn_in"]), int(params["n_cycles"]) - 1)
        params["sample_window"] = min(
            int(params["sample_window"]),
            int(params["n_cycles"]) - int(params["burn_in"]),
        )
    return params


def evaluation_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        n_workers=int(args.n_workers),
        fixed_bit_width=int(args.fixed_bit_width),
        channel_input_mode=str(args.channel_input_mode),
        eps=float(args.eps),
        fer_weight=float(args.fer_weight),
        shape_weight=0.0,
        syndrome_weight=float(args.syndrome_weight),
        floor_weight=0.0,
    )


def run_invariants(matrix_seed: int) -> dict[str, Any]:
    P = random_regular_ldpc_parity_check(
        n_bits=48,
        n_checks=24,
        variable_degree=3,
        check_degree=6,
        seed=matrix_seed,
    )
    channel_values = np.linspace(-0.7, 0.7, P.shape[1])
    common = {
        "P": P,
        "channel_values": channel_values,
        "kw": 1.2,
        "kr": 1.5,
        "n_cycles": 12,
        "I0_min": 0.1,
        "I0_max": 0.8,
        "psa_p": 0.25,
        "I0_schedule_type": "linear",
        "I0_schedule_shape": 1.0,
        "I0_hold_fraction": 0.0,
        "decision_method": "last",
        "burn_in": 0,
        "sample_window": 0,
    }

    def trajectory(mode: str, value: float) -> tuple[np.ndarray, np.ndarray]:
        output = update_pbits(
            mode=mode,
            lambda_mem=value,
            rng=np.random.default_rng(12345),
            **common,
        )
        return output[1], output[2]

    baseline = trajectory("pSA", 0.0)
    checks = {}
    for mode in ["lambda_pSA", "lambda_pSA_normalized", "gain_only_pSA", "lambda_pSA_extended"]:
        candidate = trajectory(mode, 0.0)
        checks[f"{mode}_lambda_zero_history_matches_pSA"] = bool(np.array_equal(baseline[0], candidate[0]))
        checks[f"{mode}_lambda_zero_discriminant_matches_pSA"] = bool(np.array_equal(baseline[1], candidate[1]))
    original = trajectory("lambda_pSA", 0.95)
    extended = trajectory("lambda_pSA_extended", 0.95)
    checks["extended_lambda_0p95_history_matches_original"] = bool(np.array_equal(original[0], extended[0]))
    checks["extended_lambda_0p95_discriminant_matches_original"] = bool(np.array_equal(original[1], extended[1]))
    checks["all_passed"] = bool(all(checks.values()))
    return checks


def environment_record() -> dict[str, Any]:
    try:
        import numba

        numba_version = numba.__version__
    except Exception:
        numba_version = None
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version,
        "python_executable": sys.executable,
        "numpy": np.__version__,
        "numba": numba_version,
        "cpu_count": os.cpu_count(),
    }


def config_hash(config: dict[str, Any]) -> str:
    payload = json.dumps(json_ready(config), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run matched-parameter Phase 1 LDPC controls.")
    parser.add_argument("--profile", choices=sorted(PROFILE_DEFAULTS), default="smoke")
    parser.add_argument("--sizes", default=None, help="Comma-separated size labels or 'all'.")
    parser.add_argument("--anchors", default="pSA,lambda", help="Matched non-memory parameter anchors.")
    parser.add_argument("--variants", default="additive,normalized,gain_only")
    parser.add_argument("--lambda-values", default=None, help="Comma-separated lambda grid; allowed range 0..2.")
    parser.add_argument("--trials", type=int, default=None, help="Trials per Eb/N0 and condition.")
    parser.add_argument("--seed-count", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--matrix-seed", type=int, default=0)
    parser.add_argument("--n-workers", type=int, default=8)
    parser.add_argument("--cycles-override", type=int, default=None)
    parser.add_argument("--fixed-bit-width", type=int, default=8)
    parser.add_argument("--channel-input-mode", choices=["float", "fixed"], default="float")
    parser.add_argument("--eps", type=float, default=1e-12)
    parser.add_argument("--fer-weight", type=float, default=0.2)
    parser.add_argument("--syndrome-weight", type=float, default=0.5)
    parser.add_argument("--source-root", action="append", default=None, help="Optional completed result root.")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "phase1_results"))
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
    unknown_sizes = sorted(set(sizes) - set(SIZE_SPECS))
    if unknown_sizes:
        raise ValueError(f"Unknown sizes: {unknown_sizes}")

    anchors = parse_list(args.anchors)
    if not anchors or set(anchors) - {"pSA", "lambda"}:
        raise ValueError("--anchors must contain pSA and/or lambda")
    variants = parse_list(args.variants)
    if not variants or set(variants) - {"additive", "normalized", "gain_only"}:
        raise ValueError("--variants must contain additive, normalized, and/or gain_only")

    lambda_values = (
        parse_list(args.lambda_values, float)
        if args.lambda_values is not None
        else list(defaults["lambda_values"])
    )
    if not lambda_values or min(lambda_values) < 0.0 or max(lambda_values) > 2.0:
        raise ValueError("lambda values must lie in [0, 2]")
    lambda_values = sorted(set(float(value) for value in lambda_values))

    trials = int(args.trials if args.trials is not None else defaults["trials"])
    seed_count = int(args.seed_count if args.seed_count is not None else defaults["seed_count"])
    cycles_override = (
        int(args.cycles_override)
        if args.cycles_override is not None
        else defaults["cycles_override"]
    )
    if trials <= 0 or seed_count <= 0:
        raise ValueError("trials and seed count must be positive")
    seeds = [int(args.seed) + 8022 * index for index in range(seed_count)]

    run_name = args.run_name or f"{args.profile}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(args.output_dir).expanduser().resolve() / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(run_dir)

    roots = result_roots(args.source_root)
    sources = {size: locate_completed_size(size, roots) for size in sizes}
    source_files: dict[str, dict[str, str]] = {}
    base_params: dict[tuple[str, str], dict[str, Any]] = {}
    for size in sizes:
        source_files[size] = {"base": str(sources[size])}
        for anchor in anchors:
            params, path = load_anchor(sources[size], anchor)
            base_params[(size, anchor)] = params
            source_files[size][anchor] = str(path)

    conditions = make_conditions(sizes, anchors, variants, lambda_values)
    effective_config = {
        "profile": args.profile,
        "sizes": sizes,
        "anchors": anchors,
        "variants": variants,
        "lambda_values": lambda_values,
        "trials_per_ebno": trials,
        "seed_count": seed_count,
        "seeds": seeds,
        "matrix_seed": int(args.matrix_seed),
        "n_workers": int(args.n_workers),
        "cycles_override": cycles_override,
        "fixed_bit_width": int(args.fixed_bit_width),
        "channel_input_mode": args.channel_input_mode,
        "eps": float(args.eps),
        "fer_weight": float(args.fer_weight),
        "syndrome_weight": float(args.syndrome_weight),
        "source_files": source_files,
        "condition_count": len(conditions),
        "command": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
    }
    digest = config_hash(effective_config)
    config_path = run_dir / "effective_config.json"
    if config_path.exists() and args.resume:
        previous = json.loads(config_path.read_text())
        if previous.get("config_hash") != digest:
            raise RuntimeError(
                f"Refusing to resume {run_dir}: effective configuration changed. Use a new --run-name."
            )
    write_json(config_path, {**effective_config, "config_hash": digest})

    invariants = run_invariants(int(args.matrix_seed))
    write_json(run_dir / "invariants.json", invariants)
    if not invariants["all_passed"]:
        raise RuntimeError("Phase 1 implementation invariant failed; see invariants.json")

    plan_rows = []
    total_bit_updates = 0
    for index, condition in enumerate(conditions, start=1):
        params = adjusted_params(
            base_params[(condition["size"], condition["anchor"])],
            float(condition["lambda_mem"]),
            cycles_override,
        )
        n_bits = SIZE_SPECS[condition["size"]]["n_bits"]
        updates = trials * len(load_reference(sources[condition["size"]])) * int(params["n_cycles"]) * n_bits
        total_bit_updates += updates
        plan_rows.append(
            {
                "condition_id": index,
                **condition,
                "implementation_mode": implementation_mode(condition["variant"], float(condition["lambda_mem"])),
                "trials_per_ebno": trials,
                "n_cycles": params["n_cycles"],
                "estimated_bit_updates": updates,
            }
        )
    write_csv(
        run_dir / "experiment_plan.csv",
        plan_rows,
        [
            "condition_id",
            "size",
            "anchor",
            "variant",
            "implementation_mode",
            "lambda_mem",
            "trials_per_ebno",
            "n_cycles",
            "estimated_bit_updates",
        ],
    )

    manifest = {
        "status": "planned" if args.dry_run else "running",
        "started_utc": utc_now(),
        "finished_utc": None,
        "run_dir": str(run_dir),
        "config_hash": digest,
        "environment": environment_record(),
        "condition_count": len(conditions),
        "estimated_bit_updates": total_bit_updates,
        "completed_conditions": 0,
    }
    write_json(run_dir / "manifest.json", manifest)
    logger.info("Run directory: %s", run_dir)
    logger.info("Conditions: %d; estimated bit updates: %.6g", len(conditions), total_bit_updates)
    logger.info("Common evaluation seeds: %s", seeds)
    if args.dry_run:
        logger.info("Dry run complete; no simulations executed.")
        return

    summary_path = run_dir / "summary.csv"
    by_ebno_path = run_dir / "by_ebno.csv"
    by_seed_path = run_dir / "by_seed_ebno.csv"
    summaries: list[dict[str, Any]] = read_csv(summary_path) if args.resume else []
    by_ebno_rows: list[dict[str, Any]] = read_csv(by_ebno_path) if args.resume else []
    by_seed_rows: list[dict[str, Any]] = read_csv(by_seed_path) if args.resume else []
    completed = {condition_key(row) for row in summaries}
    eval_namespace = evaluation_args(args)
    start_time = time.perf_counter()

    for index, condition in enumerate(conditions, start=1):
        key = condition_key(condition)
        if key in completed:
            logger.info("[%d/%d] resume skip %s", index, len(conditions), key)
            continue
        size = condition["size"]
        anchor = condition["anchor"]
        variant = condition["variant"]
        lambda_mem = float(condition["lambda_mem"])
        mode = implementation_mode(variant, lambda_mem)
        params = adjusted_params(base_params[(size, anchor)], lambda_mem, cycles_override)
        P = random_regular_ldpc_parity_check(
            n_bits=SIZE_SPECS[size]["n_bits"],
            n_checks=SIZE_SPECS[size]["n_checks"],
            variable_degree=3,
            check_degree=6,
            seed=int(args.matrix_seed),
        )
        reference_rows = load_reference(sources[size])
        eval_namespace.ebno_values = [float(row["EbNo_dB"]) for row in reference_rows]
        logger.info(
            "[%d/%d] %s anchor=%s variant=%s lambda=%.6g mode=%s trials/EbNo=%d cycles=%d",
            index,
            len(conditions),
            size,
            anchor,
            variant,
            lambda_mem,
            mode,
            trials,
            params["n_cycles"],
        )
        condition_start = time.perf_counter()
        record = evaluate_candidate_total_trials(
            P=P,
            reference_rows=reference_rows,
            mode=mode,
            params=params,
            total_trials=trials,
            seeds=seeds,
            args=eval_namespace,
            stage="phase1_control",
            candidate_id=index,
            return_seed_rows=True,
        )
        elapsed = time.perf_counter() - condition_start
        points = list(record["by_ebno"])
        summary = {
            "size": size,
            "anchor": anchor,
            "variant": variant,
            "implementation_mode": mode,
            "lambda_mem": lambda_mem,
            "boundary_extension": bool(variant == "additive" and lambda_mem > 0.95),
            "formula": formula_for(variant),
            "score": record["score"],
            "ber_score": record["ber_score"],
            "fer_score": record["fer_score"],
            "syndrome_penalty": record["syndrome_penalty"],
            "mean_BER": float(np.mean([point["candidate_BER"] for point in points])),
            "mean_FER": float(np.mean([point["candidate_FER"] for point in points])),
            "mean_syndrome_violation": float(
                np.mean([point["syndrome_violation_rate"] for point in points])
            ),
            "evaluation_trials_per_ebno": trials,
            "evaluation_seeds": ";".join(str(seed) for seed in seeds),
            "elapsed_seconds": elapsed,
            "source_parameter_file": source_files[size][anchor],
            **params,
        }
        summaries.append(summary)
        for point in points:
            by_ebno_rows.append(
                {
                    "size": size,
                    "anchor": anchor,
                    "variant": variant,
                    "implementation_mode": mode,
                    "lambda_mem": lambda_mem,
                    "EbNo_dB": point["EbNo_dB"],
                    "BER": point["candidate_BER"],
                    "FER": point["candidate_FER"],
                    "syndrome_violation_rate": point["syndrome_violation_rate"],
                    "avg_syndrome_weight": point["avg_syndrome_weight"],
                    "BP_BER": point["BP_BER"],
                    "BP_FER": point["BP_FER"],
                    "n_trials": point["n_trials"],
                    "n_bits_total": point["n_bits_total"],
                }
            )
        for seed_record in record.get("by_seed", []):
            for point in seed_record["by_ebno"]:
                by_seed_rows.append(
                    {
                        "size": size,
                        "anchor": anchor,
                        "variant": variant,
                        "implementation_mode": mode,
                        "lambda_mem": lambda_mem,
                        "seed": seed_record["seed"],
                        "EbNo_dB": point["EbNo_dB"],
                        "BER": point["candidate_BER"],
                        "FER": point["candidate_FER"],
                        "syndrome_violation_rate": point["syndrome_violation_rate"],
                        "avg_syndrome_weight": point["avg_syndrome_weight"],
                        "BP_BER": point["BP_BER"],
                        "BP_FER": point["BP_FER"],
                        "n_trials": point["n_trials"],
                        "n_bits_total": point["n_bits_total"],
                    }
                )
        write_csv(by_seed_path, by_seed_rows, BY_SEED_FIELDS)
        write_csv(by_ebno_path, by_ebno_rows, BY_EBNO_FIELDS)
        # summary.csv is the resume commit marker and is written last.
        write_csv(summary_path, summaries, SUMMARY_FIELDS)
        completed.add(key)
        manifest["completed_conditions"] = len(completed)
        write_json(run_dir / "manifest.json", manifest)
        logger.info("[%d/%d] complete in %.3f s; mean BER=%.6g", index, len(conditions), elapsed, summary["mean_BER"])

    total_elapsed = time.perf_counter() - start_time
    measured_updates = sum(int(row["estimated_bit_updates"]) for row in plan_rows)
    runtime_summary = {
        "elapsed_seconds_this_invocation": total_elapsed,
        "completed_conditions_total": len(completed),
        "estimated_bit_updates": measured_updates,
        "bit_updates_per_second_including_overhead": measured_updates / max(total_elapsed, 1e-12),
    }
    write_json(run_dir / "runtime_summary.json", runtime_summary)
    manifest.update(
        {
            "status": "complete",
            "finished_utc": utc_now(),
            "completed_conditions": len(completed),
            "runtime_summary": runtime_summary,
        }
    )
    write_json(run_dir / "manifest.json", manifest)
    logger.info("Done in %.3f s. Outputs: %s", total_elapsed, run_dir)


if __name__ == "__main__":
    main()

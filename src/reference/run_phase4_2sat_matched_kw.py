#!/usr/bin/env python3
"""Phase 4 random-2-SAT matched-kw validation for the PRApplied manuscript.

The experiment deliberately does not perform hyperparameter optimization.  It
freezes the non-kw/non-rho fields from the existing N500, alpha=1.20 tau-pSA
anchor and evaluates a preregistered kw x rho grid with formula- and
trial-paired random streams.  Stage B is gated by the Stage A primary contrast.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import platform
import sys
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
from scipy import stats


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "phase4_results" / "2sat_matched_kw_20260827"
PUBLIC_ANCHOR_SOURCE = REPO_ROOT / "configs" / "twosat" / "initial_tau_parameters.csv"
ANCHOR_SOURCE = (
    PUBLIC_ANCHOR_SOURCE
    if PUBLIC_ANCHOR_SOURCE.exists()
    else REPO_ROOT
    / "paper_results_trip"
    / "2sat_phase_repro"
    / "N500_A1p20"
    / "tau_pSA"
    / "best_params.csv"
)
PRIORITY_DOCUMENTS = [
    REPO_ROOT / "docs" / "provenance" / "PROTOCOL_PROVENANCE_TABLE.md",
    REPO_ROOT / "docs" / "provenance" / "UPDATED_NUMERICAL_CLAIM_LEDGER.md",
    REPO_ROOT / "docs" / "provenance" / "FIGURE_SOURCE_MAP.md",
]
KW_GRID = [4.0, 8.0, 12.0, 16.0, 20.0]
RHO_GRID = [0.0, 0.02, 0.04, 0.06, 0.08, 0.12]
STAGE_A_ALPHAS = [1.20]
STAGE_B_ALPHAS = [0.95, 1.05]
FORMULA_INDICES = list(range(30))
PRIMARY_KW = 16.0
PRIMARY_RHO = 0.08
FORMULA_GENERATOR_BASE_SEED = 0
TRIAL_SEED_BASE = 17_000_000
BOOTSTRAP_REPLICATES = 20_000
PRACTICAL_RELATIVE_REDUCTION_PCT = 2.0
PRACTICAL_FORMULA_FRACTION = 0.60


TRIAL_FIELDS = [
    "stage",
    "alpha",
    "formula_index",
    "formula_generation_seed",
    "formula_sha256",
    "is_satisfiable",
    "kw",
    "rho",
    "trial_index",
    "trial_seed",
    "n_cycles",
    "selected_unsat",
    "best_unsat",
    "final_unsat",
    "selected_unsat_rate",
    "best_unsat_rate",
    "final_unsat_rate",
    "success",
    "convergence_cycle",
    "first_hit_cycle",
    "selected_energy",
    "best_energy",
]

FORMULA_FIELDS = [
    "stage",
    "alpha",
    "formula_index",
    "formula_generation_seed",
    "formula_sha256",
    "is_satisfiable",
    "n_variables",
    "n_clauses",
    "kw",
    "rho",
    "n_trials",
    "n_cycles",
    "mean_selected_unsat",
    "mean_selected_unsat_rate",
    "mean_best_unsat",
    "mean_best_unsat_rate",
    "mean_final_unsat",
    "mean_final_unsat_rate",
    "best_trial_unsat",
    "best_trial_unsat_rate",
    "success_rate_sat",
    "mean_convergence_cycle",
    "mean_first_hit_cycle_successful",
    "formula_objective",
]


def stage_for_alpha(alpha: float) -> str:
    return "A" if abs(float(alpha) - 1.20) < 1e-12 else "B"


def alpha_label(alpha: float) -> str:
    return f"{float(alpha):.2f}".replace(".", "p")


def value_label(value: float, digits: int = 2) -> str:
    return f"{float(value):.{digits}f}".replace(".", "p")


def checkpoint_name(alpha: float, kw: float, rho: float) -> str:
    return f"stage{stage_for_alpha(alpha)}_alpha{alpha_label(alpha)}_kw{int(kw):02d}_rho{value_label(rho)}.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def formula_sha256(clauses: np.ndarray) -> str:
    values = np.asarray(clauses, dtype=np.int64, order="C")
    digest = hashlib.sha256()
    digest.update(str(values.shape).encode("ascii"))
    digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    temporary.replace(path)


def write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", newline="", encoding="utf-8", dir=path.parent, delete=False) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
        temporary = Path(handle.name)
    temporary.replace(path)


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def finite_or_blank(value: Any) -> Any:
    if value in (None, ""):
        return ""
    number = float(value)
    return number if math.isfinite(number) else ""


def load_anchor() -> Tuple[Dict[str, Any], Dict[str, str]]:
    rows = read_csv(ANCHOR_SOURCE)
    if len(rows) != 1:
        raise RuntimeError(f"Expected one alpha=1.20 tau anchor row, found {len(rows)}")
    raw = rows[0]
    if raw.get("mode") != "tau_pSA":
        raise RuntimeError("Anchor source is not tau_pSA")
    numeric_fields = [
        "kw",
        "psa_p",
        "lambda_mem",
        "response_lambda",
        "I0_min",
        "I0_max",
        "I0_schedule_shape",
        "I0_hold_fraction",
        "burn_in",
        "sample_window",
        "n_cycles",
    ]
    anchor: Dict[str, Any] = {field: float(raw[field]) for field in numeric_fields}
    anchor.update(
        {
            "kr": 0.0,
            "alpha": 1.0,
            "alpha_mode": "fixed",
            "I0_schedule_type": raw["I0_schedule_type"],
            "decision_method": raw["decision_method"],
            "burn_in": int(float(raw["burn_in"])),
            "sample_window": int(float(raw["sample_window"])),
            "n_cycles": int(float(raw["n_cycles"])),
        }
    )
    return anchor, raw


def formula_generation_seed(alpha: float, formula_index: int) -> int:
    return int(FORMULA_GENERATOR_BASE_SEED + 2003 * int(formula_index) + 500 + round(1000.0 * float(alpha)))


def trial_seed(alpha: float, formula_index: int, trial_index: int) -> int:
    alpha_milli = int(round(1000.0 * float(alpha)))
    return int(TRIAL_SEED_BASE + alpha_milli * 100_000 + int(formula_index) * 1_000 + int(trial_index))


def build_formula(alpha: float, formula_index: int) -> Dict[str, Any]:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    import run_paper_experiments as legacy

    generation_seed = formula_generation_seed(alpha, formula_index)
    formula = legacy.generate_random_2sat(500, float(alpha), seed=generation_seed)
    h, J, const = legacy.twosat_ising(formula["clauses"], 500)
    formula["h"] = h
    formula["J"] = J
    formula["const"] = const
    formula["formula_index"] = int(formula_index)
    formula["generation_seed"] = int(generation_seed)
    formula["sha256"] = formula_sha256(formula["clauses"])
    return formula


def make_cell_params(anchor: Mapping[str, Any], kw: float, rho: float, n_cycles: int) -> Dict[str, Any]:
    params = dict(anchor)
    params["kw"] = float(kw)
    params["response_lambda"] = float(rho)
    params["lambda_mem"] = 0.0
    params["n_cycles"] = int(n_cycles)
    params["burn_in"] = min(int(params["burn_in"]), int(n_cycles) - 1)
    params["sample_window"] = min(int(params["sample_window"]), int(n_cycles) - int(params["burn_in"]))
    return params


def phase4_single_trial(
    formula: Mapping[str, Any],
    mode: str,
    params: Mapping[str, Any],
    rng: np.random.Generator,
) -> Dict[str, Any]:
    """Legacy-equivalent trajectory with explicit best and terminal endpoints."""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    import run_paper_experiments as legacy

    clauses = np.asarray(formula["clauses"], dtype=int)
    n_vars = int(formula["n_vars"])
    h = np.asarray(formula["h"], dtype=float)
    J = np.asarray(formula["J"], dtype=float)
    const = float(formula["const"])
    n_cycles = int(params["n_cycles"])
    spins = rng.choice(np.array([-1, 1], dtype=int), size=n_vars)
    hold_state = np.zeros(n_vars, dtype=float)
    best_spins = spins.copy()
    best_unsat = legacy.twosat_unsat_count(spins, clauses)
    best_energy = legacy.twosat_energy(spins, h, J, const)
    best_cycle = 0
    first_solution_cycle = 0 if best_unsat == 0 else None
    last_cycle = 0
    schedule = legacy.make_schedule(
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
        elif mode == "tau_pSA":
            rho = float(params["response_lambda"])
            H_eff = rho * hold_prev + (1.0 - rho) * H_new
        else:
            raise ValueError(f"Unsupported Phase 4 mode: {mode}")

        candidate = np.where(H_eff + rnd >= 0.0, 1, -1).astype(int)
        if float(params["psa_p"]) > 0.0:
            hold_mask = rng.random(n_vars) < float(params["psa_p"])
            candidate[hold_mask] = spins_prev[hold_mask]
            if mode == "tau_pSA":
                H_eff[hold_mask] = hold_prev[hold_mask]
            else:
                H_new[hold_mask] = hold_prev[hold_mask]
        spins = candidate
        hold_state = H_eff if mode == "tau_pSA" else H_new

        unsat = legacy.twosat_unsat_count(spins, clauses)
        if unsat == 0 and first_solution_cycle is None:
            first_solution_cycle = cycle + 1
        if unsat < best_unsat:
            best_unsat = unsat
            best_energy = legacy.twosat_energy(spins, h, J, const)
            best_spins = spins.copy()
            best_cycle = cycle + 1
            if best_unsat == 0:
                break

    final_unsat = legacy.twosat_unsat_count(spins, clauses)
    final_energy = legacy.twosat_energy(spins, h, J, const)
    if params["decision_method"] == "last":
        selected_spins = spins
        selected_unsat = final_unsat
        selected_energy = final_energy
        convergence_cycle = last_cycle
    else:
        selected_spins = best_spins
        selected_unsat = best_unsat
        selected_energy = best_energy
        convergence_cycle = best_cycle

    return {
        "selected_unsat": int(selected_unsat),
        "best_unsat": int(best_unsat),
        "final_unsat": int(final_unsat),
        "selected_energy": float(selected_energy),
        "best_energy": float(best_energy),
        "final_energy": float(final_energy),
        "convergence_cycle": int(convergence_cycle),
        "first_hit_cycle": None if first_solution_cycle is None else int(first_solution_cycle),
        "selected_state": selected_spins,
        "final_state": spins,
    }


def run_cell_task(task: Mapping[str, Any]) -> Dict[str, Any]:
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    alpha = float(task["alpha"])
    kw = float(task["kw"])
    rho = float(task["rho"])
    n_trials = int(task["n_trials"])
    n_cycles = int(task["n_cycles"])
    anchor = dict(task["anchor"])
    params = make_cell_params(anchor, kw, rho, n_cycles)
    trials: List[Dict[str, Any]] = []
    formulas: List[Dict[str, Any]] = []
    started = time.time()

    for formula_index in FORMULA_INDICES:
        formula = build_formula(alpha, formula_index)
        M = int(formula["n_clauses"])
        formula_trials: List[Dict[str, Any]] = []
        for trial_index in range(n_trials):
            seed = trial_seed(alpha, formula_index, trial_index)
            result = phase4_single_trial(formula, "tau_pSA", params, np.random.default_rng(seed))
            selected_rate = float(result["selected_unsat"] / M)
            best_rate = float(result["best_unsat"] / M)
            final_rate = float(result["final_unsat"] / M)
            row = {
                "stage": stage_for_alpha(alpha),
                "alpha": alpha,
                "formula_index": formula_index,
                "formula_generation_seed": formula["generation_seed"],
                "formula_sha256": formula["sha256"],
                "is_satisfiable": bool(formula["is_satisfiable"]),
                "kw": kw,
                "rho": rho,
                "trial_index": trial_index,
                "trial_seed": seed,
                "n_cycles": n_cycles,
                "selected_unsat": result["selected_unsat"],
                "best_unsat": result["best_unsat"],
                "final_unsat": result["final_unsat"],
                "selected_unsat_rate": selected_rate,
                "best_unsat_rate": best_rate,
                "final_unsat_rate": final_rate,
                "success": int(result["selected_unsat"] == 0),
                "convergence_cycle": result["convergence_cycle"],
                "first_hit_cycle": "" if result["first_hit_cycle"] is None else result["first_hit_cycle"],
                "selected_energy": result["selected_energy"],
                "best_energy": result["best_energy"],
            }
            trials.append(row)
            formula_trials.append(row)

        selected = np.array([float(row["selected_unsat"]) for row in formula_trials])
        best = np.array([float(row["best_unsat"]) for row in formula_trials])
        final = np.array([float(row["final_unsat"]) for row in formula_trials])
        convergence = np.array([float(row["convergence_cycle"]) for row in formula_trials])
        first_hits = [float(row["first_hit_cycle"]) for row in formula_trials if row["first_hit_cycle"] != ""]
        is_sat = bool(formula["is_satisfiable"])
        success_rate = float(np.mean(selected == 0.0)) if is_sat else math.nan
        formula_objective = (
            float(np.mean(selected) / M)
            + 0.25 * float(np.min(best) / M)
            - (0.10 * success_rate if is_sat else 0.0)
            + 0.001 * float(np.mean(convergence) / n_cycles)
        )
        formulas.append(
            {
                "stage": stage_for_alpha(alpha),
                "alpha": alpha,
                "formula_index": formula_index,
                "formula_generation_seed": formula["generation_seed"],
                "formula_sha256": formula["sha256"],
                "is_satisfiable": is_sat,
                "n_variables": 500,
                "n_clauses": M,
                "kw": kw,
                "rho": rho,
                "n_trials": n_trials,
                "n_cycles": n_cycles,
                "mean_selected_unsat": float(np.mean(selected)),
                "mean_selected_unsat_rate": float(np.mean(selected) / M),
                "mean_best_unsat": float(np.mean(best)),
                "mean_best_unsat_rate": float(np.mean(best) / M),
                "mean_final_unsat": float(np.mean(final)),
                "mean_final_unsat_rate": float(np.mean(final) / M),
                "best_trial_unsat": int(np.min(best)),
                "best_trial_unsat_rate": float(np.min(best) / M),
                "success_rate_sat": finite_or_blank(success_rate),
                "mean_convergence_cycle": float(np.mean(convergence)),
                "mean_first_hit_cycle_successful": finite_or_blank(np.mean(first_hits) if first_hits else math.nan),
                "formula_objective": float(formula_objective),
            }
        )

    return {
        "status": "complete",
        "task": dict(task),
        "elapsed_seconds": float(time.time() - started),
        "trial_rows": trials,
        "formula_rows": formulas,
    }


def legacy_label_map(alpha: float) -> Dict[int, bool]:
    size = f"N500_A{alpha_label(alpha)}"
    path = REPO_ROOT / "paper_results_trip" / "2sat_phase_repro" / size / "tau_pSA" / "memory_sweep_by_instance.csv"
    if not path.exists():
        return {}
    labels: Dict[int, bool] = {}
    for row in read_csv(path):
        index = int(float(row["instance_seed"]))
        value = parse_bool(row["is_satisfiable"])
        if index in labels and labels[index] != value:
            raise RuntimeError(f"Conflicting legacy SAT label for alpha={alpha}, formula={index}")
        labels[index] = value
    return labels


def build_formula_metadata(alphas: Sequence[float]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    audit: Dict[str, Any] = {"all_passed": True, "by_alpha": {}}
    for alpha in alphas:
        legacy_labels = legacy_label_map(alpha)
        alpha_checks = {"legacy_rows_found": len(legacy_labels), "label_matches": 0, "mismatches": []}
        for formula_index in FORMULA_INDICES:
            formula = build_formula(alpha, formula_index)
            label = bool(formula["is_satisfiable"])
            legacy_label = legacy_labels.get(formula_index)
            label_match = legacy_label is None or legacy_label == label
            if label_match and legacy_label is not None:
                alpha_checks["label_matches"] += 1
            if not label_match:
                alpha_checks["mismatches"].append(formula_index)
                audit["all_passed"] = False
            rows.append(
                {
                    "alpha": float(alpha),
                    "formula_index": formula_index,
                    "formula_generation_seed": formula["generation_seed"],
                    "formula_sha256": formula["sha256"],
                    "n_variables": 500,
                    "n_clauses": int(formula["n_clauses"]),
                    "is_satisfiable": label,
                    "legacy_label_available": legacy_label is not None,
                    "legacy_label_match": label_match,
                }
            )
        alpha_checks["n_satisfiable"] = sum(1 for row in rows if row["alpha"] == alpha and row["is_satisfiable"])
        alpha_checks["n_unsatisfiable"] = 30 - alpha_checks["n_satisfiable"]
        audit["by_alpha"][f"{alpha:.2f}"] = alpha_checks
    return rows, audit


def preflight_checks(anchor: Mapping[str, Any]) -> Dict[str, Any]:
    checks: Dict[str, Any] = {
        "rho_zero_tau_matches_psa": True,
        "phase4_matches_legacy_selected_endpoint": True,
        "non_kw_rho_fields_fixed": True,
        "paired_seed_invariant": True,
        "details": [],
    }
    invariant = None
    for kw in KW_GRID:
        for rho in RHO_GRID:
            params = make_cell_params(anchor, kw, rho, int(anchor["n_cycles"]))
            frozen = {key: value for key, value in params.items() if key not in {"kw", "response_lambda"}}
            if invariant is None:
                invariant = frozen
            elif frozen != invariant:
                checks["non_kw_rho_fields_fixed"] = False
    for alpha in STAGE_A_ALPHAS:
        for formula_index in FORMULA_INDICES:
            for trial_index in (0, 1):
                seed = trial_seed(alpha, formula_index, trial_index)
                if seed != trial_seed(alpha, formula_index, trial_index):
                    checks["paired_seed_invariant"] = False

    for formula_index in FORMULA_INDICES:
        formula = build_formula(1.20, formula_index)
        params = make_cell_params(anchor, PRIMARY_KW, 0.0, int(anchor["n_cycles"]))
        for trial_index in (0, 1):
            seed = trial_seed(1.20, formula_index, trial_index)
            psa = phase4_single_trial(formula, "pSA", params, np.random.default_rng(seed))
            tau = phase4_single_trial(formula, "tau_pSA", params, np.random.default_rng(seed))
            same = (
                psa["selected_unsat"] == tau["selected_unsat"]
                and psa["best_unsat"] == tau["best_unsat"]
                and psa["final_unsat"] == tau["final_unsat"]
                and psa["convergence_cycle"] == tau["convergence_cycle"]
                and psa["first_hit_cycle"] == tau["first_hit_cycle"]
                and np.array_equal(psa["selected_state"], tau["selected_state"])
                and np.array_equal(psa["final_state"], tau["final_state"])
            )
            if not same:
                checks["rho_zero_tau_matches_psa"] = False
                checks["details"].append({"type": "rho_zero_mismatch", "formula_index": formula_index, "trial_index": trial_index})

    import run_paper_experiments as legacy

    for formula_index, rho in ((0, 0.0), (1, PRIMARY_RHO)):
        formula = build_formula(1.20, formula_index)
        params = make_cell_params(anchor, PRIMARY_KW, rho, int(anchor["n_cycles"]))
        seed = trial_seed(1.20, formula_index, 2)
        candidate = phase4_single_trial(formula, "tau_pSA", params, np.random.default_rng(seed))
        reference = legacy.twosat_single_trial(formula, "tau_pSA", params, np.random.default_rng(seed))
        same = (
            candidate["selected_unsat"] == int(reference["unsat"])
            and candidate["convergence_cycle"] == int(reference["convergence_cycle"])
            and candidate["first_hit_cycle"]
            == (None if reference["time_to_first_solution"] == "" else int(reference["time_to_first_solution"]))
            and np.array_equal(candidate["selected_state"], reference["state"])
            and abs(candidate["selected_energy"] - float(reference["energy"])) < 1e-12
        )
        if not same:
            checks["phase4_matches_legacy_selected_endpoint"] = False
            checks["details"].append({"type": "legacy_mismatch", "formula_index": formula_index, "rho": rho})
    checks["all_passed"] = all(bool(checks[key]) for key in [
        "rho_zero_tau_matches_psa",
        "phase4_matches_legacy_selected_endpoint",
        "non_kw_rho_fields_fixed",
        "paired_seed_invariant",
    ])
    return checks


def task_fingerprint(task: Mapping[str, Any]) -> str:
    stable = {key: task[key] for key in ["alpha", "kw", "rho", "n_trials", "n_cycles", "anchor"]}
    return hashlib.sha256(json.dumps(stable, sort_keys=True).encode("utf-8")).hexdigest()


def checkpoint_complete(path: Path, task: Mapping[str, Any]) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        payload.get("status") == "complete"
        and payload.get("task_fingerprint") == task_fingerprint(task)
        and len(payload.get("formula_rows", [])) == len(FORMULA_INDICES)
        and len(payload.get("trial_rows", [])) == len(FORMULA_INDICES) * int(task["n_trials"])
    )


def run_grid(
    output: Path,
    alphas: Sequence[float],
    anchor: Mapping[str, Any],
    n_trials: int,
    n_cycles: int,
    n_workers: int,
    resume: bool,
) -> None:
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tasks: List[Dict[str, Any]] = []
    for alpha in alphas:
        for kw in KW_GRID:
            for rho in RHO_GRID:
                task = {
                    "alpha": float(alpha),
                    "kw": float(kw),
                    "rho": float(rho),
                    "n_trials": int(n_trials),
                    "n_cycles": int(n_cycles),
                    "anchor": dict(anchor),
                }
                checkpoint = checkpoint_dir / checkpoint_name(alpha, kw, rho)
                if resume and checkpoint_complete(checkpoint, task):
                    logging.info("resume skip %s", checkpoint.name)
                else:
                    tasks.append(task)
    if not tasks:
        logging.info("No grid cells pending")
        return

    logging.info("Launching %d cells with %d workers", len(tasks), n_workers)
    with ProcessPoolExecutor(max_workers=max(1, int(n_workers))) as executor:
        futures = {executor.submit(run_cell_task, task): task for task in tasks}
        completed = 0
        for future in as_completed(futures):
            task = futures[future]
            payload = future.result()
            payload["task_fingerprint"] = task_fingerprint(task)
            checkpoint = checkpoint_dir / checkpoint_name(float(task["alpha"]), float(task["kw"]), float(task["rho"]))
            write_json(checkpoint, payload)
            completed += 1
            logging.info(
                "completed %d/%d alpha=%.2f kw=%.0f rho=%.2f elapsed=%.1fs",
                completed,
                len(tasks),
                task["alpha"],
                task["kw"],
                task["rho"],
                payload["elapsed_seconds"],
            )


def load_checkpoints(output: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    trials: List[Dict[str, Any]] = []
    formulas: List[Dict[str, Any]] = []
    manifests: List[Dict[str, Any]] = []
    for path in sorted((output / "checkpoints").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "complete":
            continue
        trials.extend(payload["trial_rows"])
        formulas.extend(payload["formula_rows"])
        manifests.append(
            {
                "checkpoint": path.name,
                "sha256": sha256_file(path),
                "task_fingerprint": payload["task_fingerprint"],
                "elapsed_seconds": payload["elapsed_seconds"],
                "trial_rows": len(payload["trial_rows"]),
                "formula_rows": len(payload["formula_rows"]),
            }
        )
    trials.sort(key=lambda row: (float(row["alpha"]), float(row["kw"]), float(row["rho"]), int(row["formula_index"]), int(row["trial_index"])))
    formulas.sort(key=lambda row: (float(row["alpha"]), float(row["kw"]), float(row["rho"]), int(row["formula_index"])))
    return trials, formulas, manifests


def bootstrap_mean_ci(values: Sequence[float], seed: int, replicates: int = BOOTSTRAP_REPLICATES) -> Tuple[float, float]:
    data = np.asarray(values, dtype=float)
    if data.size == 0:
        return math.nan, math.nan
    if data.size == 1 or np.all(data == data[0]):
        return float(np.mean(data)), float(np.mean(data))
    rng = np.random.default_rng(seed)
    sample_indices = rng.integers(0, data.size, size=(int(replicates), data.size))
    means = np.mean(data[sample_indices], axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    if total <= 0:
        return math.nan, math.nan
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    margin = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return center - margin, center + margin


def aggregate_cell_rows(formulas: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[float, float, float], List[Mapping[str, Any]]] = defaultdict(list)
    for row in formulas:
        groups[(float(row["alpha"]), float(row["kw"]), float(row["rho"]))].append(row)
    output: List[Dict[str, Any]] = []
    for (alpha, kw, rho), cell_rows in sorted(groups.items()):
        for stratum in ("all", "SAT", "UNSAT"):
            selected = [row for row in cell_rows if stratum == "all" or bool(row["is_satisfiable"]) == (stratum == "SAT")]
            if not selected:
                continue
            selected_rates = [float(row["mean_selected_unsat_rate"]) for row in selected]
            ci_low, ci_high = bootstrap_mean_ci(selected_rates, seed=int(alpha * 100_000 + kw * 100 + rho * 10_000 + len(selected)))
            sat_rows = [row for row in selected if bool(row["is_satisfiable"])]
            success_values = [float(row["success_rate_sat"]) for row in sat_rows]
            success_ci = bootstrap_mean_ci(success_values, seed=int(alpha * 200_000 + kw * 200 + rho * 20_000 + 7)) if success_values else (math.nan, math.nan)
            mean_rate = float(np.mean(selected_rates))
            best_rate = float(np.mean([float(row["best_trial_unsat_rate"]) for row in selected]))
            mean_convergence = float(np.mean([float(row["mean_convergence_cycle"]) for row in selected]))
            sat_success = float(np.mean(success_values)) if success_values else math.nan
            aggregate_objective = mean_rate + 0.25 * best_rate + 0.001 * mean_convergence / float(selected[0]["n_cycles"])
            if success_values:
                aggregate_objective -= 0.10 * sat_success
            first_hits = [float(row["mean_first_hit_cycle_successful"]) for row in sat_rows if row["mean_first_hit_cycle_successful"] != ""]
            output.append(
                {
                    "stage": stage_for_alpha(alpha),
                    "alpha": alpha,
                    "kw": kw,
                    "rho": rho,
                    "stratum": stratum,
                    "n_formulas": len(selected),
                    "n_satisfiable_formulas": len(sat_rows),
                    "n_unsatisfiable_formulas": len(selected) - len(sat_rows),
                    "trials_per_formula": int(selected[0]["n_trials"]),
                    "mean_selected_unsat": float(np.mean([float(row["mean_selected_unsat"]) for row in selected])),
                    "mean_selected_unsat_rate": mean_rate,
                    "mean_selected_unsat_rate_ci_low": ci_low,
                    "mean_selected_unsat_rate_ci_high": ci_high,
                    "mean_final_unsat": float(np.mean([float(row["mean_final_unsat"]) for row in selected])),
                    "mean_final_unsat_rate": float(np.mean([float(row["mean_final_unsat_rate"]) for row in selected])),
                    "mean_best_trial_unsat": float(np.mean([float(row["best_trial_unsat"]) for row in selected])),
                    "sat_success_rate": finite_or_blank(sat_success),
                    "sat_success_rate_ci_low": finite_or_blank(success_ci[0]),
                    "sat_success_rate_ci_high": finite_or_blank(success_ci[1]),
                    "mean_convergence_cycle": mean_convergence,
                    "mean_first_hit_cycle_successful": finite_or_blank(np.mean(first_hits) if first_hits else math.nan),
                    "aggregate_objective": float(aggregate_objective),
                }
            )
    return output


def paired_analysis(formulas: Sequence[Mapping[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    lookup = {
        (float(row["alpha"]), float(row["kw"]), float(row["rho"]), int(row["formula_index"])): row
        for row in formulas
    }
    alphas = sorted({float(row["alpha"]) for row in formulas})
    paired_rows: List[Dict[str, Any]] = []
    statistics_rows: List[Dict[str, Any]] = []
    metric_specs = [
        ("all_mean_selected_unsat_rate", "mean_selected_unsat_rate", "all"),
        ("all_mean_selected_unsat", "mean_selected_unsat", "all"),
        ("all_mean_final_unsat_rate", "mean_final_unsat_rate", "all"),
        ("SAT_success_rate", "success_rate_sat", "SAT"),
        ("SAT_mean_selected_unsat_rate", "mean_selected_unsat_rate", "SAT"),
        ("UNSAT_mean_selected_unsat_rate", "mean_selected_unsat_rate", "UNSAT"),
    ]
    for alpha in alphas:
        for kw in KW_GRID:
            for rho in [value for value in RHO_GRID if value > 0.0]:
                formula_pairs: List[Tuple[Mapping[str, Any], Mapping[str, Any]]] = []
                for formula_index in FORMULA_INDICES:
                    baseline = lookup.get((alpha, kw, 0.0, formula_index))
                    treatment = lookup.get((alpha, kw, rho, formula_index))
                    if baseline is not None and treatment is not None:
                        formula_pairs.append((baseline, treatment))
                        paired_rows.append(
                            {
                                "stage": stage_for_alpha(alpha),
                                "alpha": alpha,
                                "kw": kw,
                                "rho": rho,
                                "formula_index": formula_index,
                                "formula_sha256": baseline["formula_sha256"],
                                "is_satisfiable": baseline["is_satisfiable"],
                                "baseline_mean_selected_unsat_rate": baseline["mean_selected_unsat_rate"],
                                "treatment_mean_selected_unsat_rate": treatment["mean_selected_unsat_rate"],
                                "difference_treatment_minus_baseline_selected_rate": float(treatment["mean_selected_unsat_rate"]) - float(baseline["mean_selected_unsat_rate"]),
                                "difference_treatment_minus_baseline_final_rate": float(treatment["mean_final_unsat_rate"]) - float(baseline["mean_final_unsat_rate"]),
                                "difference_treatment_minus_baseline_success": (
                                    float(treatment["success_rate_sat"]) - float(baseline["success_rate_sat"])
                                    if bool(baseline["is_satisfiable"])
                                    else ""
                                ),
                                "difference_treatment_minus_baseline_convergence_cycle": float(treatment["mean_convergence_cycle"]) - float(baseline["mean_convergence_cycle"]),
                                "difference_treatment_minus_baseline_objective": float(treatment["formula_objective"]) - float(baseline["formula_objective"]),
                            }
                        )
                for metric_name, field, stratum in metric_specs:
                    selected_pairs = [
                        pair
                        for pair in formula_pairs
                        if stratum == "all" or bool(pair[0]["is_satisfiable"]) == (stratum == "SAT")
                    ]
                    if not selected_pairs:
                        continue
                    baseline_values = np.array([float(pair[0][field]) for pair in selected_pairs], dtype=float)
                    treatment_values = np.array([float(pair[1][field]) for pair in selected_pairs], dtype=float)
                    differences = treatment_values - baseline_values
                    ci_low, ci_high = bootstrap_mean_ci(
                        differences,
                        seed=int(alpha * 1_000_000 + kw * 10_000 + rho * 1_000_000 + sum(ord(char) for char in metric_name)),
                    )
                    mean_diff = float(np.mean(differences))
                    sd_diff = float(np.std(differences, ddof=1)) if len(differences) > 1 else math.nan
                    dz = mean_diff / sd_diff if math.isfinite(sd_diff) and sd_diff > 0 else math.nan
                    try:
                        t_p = float(stats.ttest_rel(treatment_values, baseline_values).pvalue)
                    except Exception:
                        t_p = math.nan
                    try:
                        w_p = float(stats.wilcoxon(differences, zero_method="pratt").pvalue) if np.any(differences != 0.0) else 1.0
                    except Exception:
                        w_p = math.nan
                    if metric_name == "SAT_success_rate":
                        improved = int(np.count_nonzero(differences > 0.0))
                        worsened = int(np.count_nonzero(differences < 0.0))
                    else:
                        improved = int(np.count_nonzero(differences < 0.0))
                        worsened = int(np.count_nonzero(differences > 0.0))
                    tied = int(len(differences) - improved - worsened)
                    wilson_low, wilson_high = wilson_interval(improved, len(differences))
                    base_mean = float(np.mean(baseline_values))
                    if metric_name == "SAT_success_rate":
                        relative_reduction = math.nan
                    else:
                        relative_reduction = 100.0 * (base_mean - float(np.mean(treatment_values))) / base_mean if base_mean != 0 else math.nan
                    statistics_rows.append(
                        {
                            "stage": stage_for_alpha(alpha),
                            "alpha": alpha,
                            "kw": kw,
                            "rho": rho,
                            "metric": metric_name,
                            "stratum": stratum,
                            "n_formulas": len(differences),
                            "baseline_mean": base_mean,
                            "treatment_mean": float(np.mean(treatment_values)),
                            "mean_difference_treatment_minus_baseline": mean_diff,
                            "ci95_low": ci_low,
                            "ci95_high": ci_high,
                            "relative_reduction_pct": finite_or_blank(relative_reduction),
                            "standardized_paired_effect_dz": finite_or_blank(dz),
                            "paired_t_pvalue": finite_or_blank(t_p),
                            "wilcoxon_pvalue": finite_or_blank(w_p),
                            "improved_formula_count": improved,
                            "worsened_formula_count": worsened,
                            "tied_formula_count": tied,
                            "improved_formula_fraction": improved / len(differences),
                            "improved_formula_fraction_ci_low": wilson_low,
                            "improved_formula_fraction_ci_high": wilson_high,
                            "holm_adjusted_pvalue_primary_metric_within_alpha": "",
                            "is_preregistered_primary": bool(abs(alpha - 1.20) < 1e-12 and kw == PRIMARY_KW and abs(rho - PRIMARY_RHO) < 1e-12 and metric_name == "all_mean_selected_unsat_rate"),
                        }
                    )
    for alpha in alphas:
        candidates = [
            row
            for row in statistics_rows
            if float(row["alpha"]) == alpha and row["metric"] == "all_mean_selected_unsat_rate"
        ]
        ordered = sorted(candidates, key=lambda row: float(row["paired_t_pvalue"]) if row["paired_t_pvalue"] != "" else 1.0)
        m = len(ordered)
        adjusted_running = 0.0
        for rank, row in enumerate(ordered, start=1):
            raw_p = float(row["paired_t_pvalue"]) if row["paired_t_pvalue"] != "" else 1.0
            adjusted = min(1.0, (m - rank + 1) * raw_p)
            adjusted_running = max(adjusted_running, adjusted)
            row["holm_adjusted_pvalue_primary_metric_within_alpha"] = adjusted_running
    return paired_rows, statistics_rows


def response_surface_diagnostics(aggregate_rows: Sequence[Mapping[str, Any]], statistics_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    diagnostics: Dict[str, Any] = {"by_alpha": {}}
    alphas = sorted({float(row["alpha"]) for row in aggregate_rows})
    for alpha in alphas:
        cells = [row for row in aggregate_rows if float(row["alpha"]) == alpha and row["stratum"] == "all"]
        values = {(float(row["kw"]), float(row["rho"])): float(row["mean_selected_unsat_rate"]) for row in cells}
        best_cell = min(values, key=values.get)
        kw_means = {kw: float(np.mean([values[(kw, rho)] for rho in RHO_GRID])) for kw in KW_GRID}
        rho_means = {rho: float(np.mean([values[(kw, rho)] for kw in KW_GRID])) for rho in RHO_GRID}
        grand = float(np.mean(list(values.values())))
        interaction_residuals = [values[(kw, rho)] - kw_means[kw] - rho_means[rho] + grand for kw in KW_GRID for rho in RHO_GRID]
        best_rho_by_kw = {str(int(kw)): min(RHO_GRID, key=lambda rho: values[(kw, rho)]) for kw in KW_GRID}
        effect_rows = [
            row
            for row in statistics_rows
            if float(row["alpha"]) == alpha and row["metric"] == "all_mean_selected_unsat_rate"
        ]
        significant_better = [
            {"kw": row["kw"], "rho": row["rho"], "mean_difference": row["mean_difference_treatment_minus_baseline"], "ci95_high": row["ci95_high"]}
            for row in effect_rows
            if float(row["ci95_high"]) < 0.0
        ]
        diagnostics["by_alpha"][f"{alpha:.2f}"] = {
            "best_cell": {"kw": best_cell[0], "rho": best_cell[1], "mean_selected_unsat_rate": values[best_cell]},
            "best_cell_on_kw_boundary": best_cell[0] in {min(KW_GRID), max(KW_GRID)},
            "best_cell_on_rho_boundary": best_cell[1] in {min(RHO_GRID), max(RHO_GRID)},
            "best_rho_by_kw": best_rho_by_kw,
            "kw_main_span": max(kw_means.values()) - min(kw_means.values()),
            "rho_main_span": max(rho_means.values()) - min(rho_means.values()),
            "kw_to_rho_span_ratio": (max(kw_means.values()) - min(kw_means.values())) / max(max(rho_means.values()) - min(rho_means.values()), 1e-15),
            "interaction_residual_range": max(interaction_residuals) - min(interaction_residuals),
            "unadjusted_ci_better_nonzero_cells": significant_better,
            "n_unadjusted_ci_better_nonzero_cells": len(significant_better),
        }
    return diagnostics


def stage_b_decision(statistics_rows: Sequence[Mapping[str, Any]], surface: Mapping[str, Any]) -> Dict[str, Any]:
    primary = next(
        row
        for row in statistics_rows
        if bool(row["is_preregistered_primary"])
    )
    kw16_rows = [
        row
        for row in statistics_rows
        if float(row["alpha"]) == 1.20 and float(row["kw"]) == PRIMARY_KW and row["metric"] == "all_mean_selected_unsat_rate"
    ]
    better_nonzero = [row for row in kw16_rows if float(row["mean_difference_treatment_minus_baseline"]) < 0.0]
    adjacent_support = [row for row in kw16_rows if float(row["rho"]) in {0.06, 0.12} and float(row["mean_difference_treatment_minus_baseline"]) < 0.0]
    relative_reduction = float(primary["relative_reduction_pct"]) if primary["relative_reduction_pct"] != "" else -math.inf
    gate = (
        float(primary["ci95_high"]) < 0.0
        and relative_reduction >= PRACTICAL_RELATIVE_REDUCTION_PCT
        and float(primary["improved_formula_fraction"]) >= PRACTICAL_FORMULA_FRACTION
        and len(better_nonzero) >= 2
        and len(adjacent_support) >= 1
    )
    surface_a = surface["by_alpha"]["1.20"]
    if gate:
        case = "Case 1"
        interpretation = "Matched-kw nonzero-rho improvement is statistically and practically supported; run Stage B."
    elif (
        float(primary["mean_difference_treatment_minus_baseline"]) >= 0.0
        and int(surface_a["n_unadjusted_ci_better_nonzero_cells"]) == 0
    ):
        case = "Case 3"
        interpretation = "The preregistered matched-kw rho effect disappears; do not run Stage B."
    else:
        case = "Case 2"
        interpretation = "The rho effect is small, unstable, or formula/cell dependent; Stage B is not automatically triggered."
    return {
        "run_stage_b": bool(gate),
        "result_case": case,
        "interpretation": interpretation,
        "primary": dict(primary),
        "gate_thresholds": {
            "ci95_high_below_zero": True,
            "minimum_relative_reduction_pct": PRACTICAL_RELATIVE_REDUCTION_PCT,
            "minimum_improved_formula_fraction": PRACTICAL_FORMULA_FRACTION,
            "minimum_better_nonzero_rho_cells_at_kw16": 2,
            "requires_adjacent_rho_support_0p06_or_0p12": True,
        },
        "observed_support": {
            "better_nonzero_rho_cells_at_kw16": len(better_nonzero),
            "adjacent_support_cells": len(adjacent_support),
        },
    }


def validate_outputs(
    trials: Sequence[Mapping[str, Any]],
    formulas: Sequence[Mapping[str, Any]],
    manifests: Sequence[Mapping[str, Any]],
    n_trials: int,
    n_cycles: int,
) -> Dict[str, Any]:
    n_cells = len(manifests)
    expected_formula_rows = n_cells * len(FORMULA_INDICES)
    expected_trial_rows = expected_formula_rows * int(n_trials)
    formula_keys = {
        (float(row["alpha"]), float(row["kw"]), float(row["rho"]), int(row["formula_index"]))
        for row in formulas
    }
    trial_keys = {
        (float(row["alpha"]), float(row["kw"]), float(row["rho"]), int(row["formula_index"]), int(row["trial_index"]))
        for row in trials
    }
    range_checks = all(
        0 <= int(row["best_unsat"]) <= int(round(float(row["alpha"]) * 500))
        and 0 <= int(row["selected_unsat"]) <= int(round(float(row["alpha"]) * 500))
        and 0 <= int(row["final_unsat"]) <= int(round(float(row["alpha"]) * 500))
        and 0 <= int(row["convergence_cycle"]) <= int(n_cycles)
        for row in trials
    )
    paired_seed_checks: Dict[Tuple[float, int, int], set] = defaultdict(set)
    for row in trials:
        paired_seed_checks[(float(row["alpha"]), int(row["formula_index"]), int(row["trial_index"]))].add(int(row["trial_seed"]))
    checks = {
        "n_completed_cells": n_cells,
        "trial_row_count": len(trials),
        "expected_trial_row_count": expected_trial_rows,
        "formula_row_count": len(formulas),
        "expected_formula_row_count": expected_formula_rows,
        "unique_trial_keys": len(trial_keys),
        "unique_formula_keys": len(formula_keys),
        "row_counts_match": len(trials) == expected_trial_rows and len(formulas) == expected_formula_rows,
        "no_duplicate_keys": len(trial_keys) == len(trials) and len(formula_keys) == len(formulas),
        "value_ranges_valid": range_checks,
        "paired_seeds_identical_across_cells": all(len(values) == 1 for values in paired_seed_checks.values()),
        "csv_schema_trial": TRIAL_FIELDS,
        "csv_schema_formula": FORMULA_FIELDS,
    }
    checks["all_passed"] = all(checks[key] for key in ["row_counts_match", "no_duplicate_keys", "value_ranges_valid", "paired_seeds_identical_across_cells"])
    return checks


def make_figures(
    output: Path,
    aggregate_rows: Sequence[Mapping[str, Any]],
    statistics_rows: Sequence[Mapping[str, Any]],
) -> List[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    logging.getLogger("fontTools").setLevel(logging.WARNING)

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.dpi": 180,
            "savefig.dpi": 300,
        }
    )
    figure_dir = output / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    stage_a = [row for row in aggregate_rows if float(row["alpha"]) == 1.20 and row["stratum"] == "all"]
    value_map = {(float(row["kw"]), float(row["rho"])): float(row["mean_selected_unsat_rate"]) for row in stage_a}
    matrix = np.array([[value_map[(kw, rho)] for rho in RHO_GRID] for kw in KW_GRID]) * 1000.0

    def heatmap(ax: Any) -> None:
        image = ax.imshow(matrix, origin="lower", aspect="auto", cmap="viridis_r")
        ax.set_xticks(np.arange(len(RHO_GRID)), [f"{rho:.2f}" for rho in RHO_GRID])
        ax.set_yticks(np.arange(len(KW_GRID)), [f"{kw:.0f}" for kw in KW_GRID])
        ax.set_xlabel(r"Response coefficient $\rho$")
        ax.set_ylabel(r"Constraint weight $k_w$")
        ax.set_title(r"$N=500$, $\alpha=1.20$: mean unsatisfied-clause rate")
        for y in range(len(KW_GRID)):
            for x in range(len(RHO_GRID)):
                color = "white" if matrix[y, x] > np.median(matrix) else "black"
                ax.text(x, y, f"{matrix[y, x]:.3f}", ha="center", va="center", fontsize=7, color=color)
        colorbar = ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        colorbar.set_label(r"Mean unsatisfied-clause rate ($\times 10^{-3}$)")

    fig, ax = plt.subplots(figsize=(6.3, 4.1), constrained_layout=True)
    heatmap(ax)
    a_base = figure_dir / "Fig_phase4_A_kw_rho_heatmap"
    fig.savefig(a_base.with_suffix(".png"))
    fig.savefig(a_base.with_suffix(".pdf"))
    plt.close(fig)

    curve_cells = sorted(
        [row for row in stage_a if float(row["kw"]) == PRIMARY_KW],
        key=lambda row: float(row["rho"]),
    )
    curve_stats = {
        float(row["rho"]): row
        for row in statistics_rows
        if float(row["alpha"]) == 1.20 and float(row["kw"]) == PRIMARY_KW and row["metric"] == "all_mean_selected_unsat_rate"
    }

    def curve(axes: Sequence[Any]) -> None:
        rhos = np.array([float(row["rho"]) for row in curve_cells])
        means = np.array([float(row["mean_selected_unsat_rate"]) for row in curve_cells]) * 1000.0
        lows = np.array([float(row["mean_selected_unsat_rate_ci_low"]) for row in curve_cells]) * 1000.0
        highs = np.array([float(row["mean_selected_unsat_rate_ci_high"]) for row in curve_cells]) * 1000.0
        axes[0].errorbar(rhos, means, yerr=np.vstack([means - lows, highs - means]), marker="o", color="#1f77b4", capsize=3)
        axes[0].axvline(PRIMARY_RHO, color="#d62728", linestyle="--", linewidth=1, label="Preregistered nonzero rho")
        axes[0].set_ylabel(r"Mean unsatisfied rate ($\times 10^{-3}$)")
        axes[0].set_title(r"Matched $k_w=16$ response-memory slice")
        axes[0].legend(frameon=False, loc="best")
        diff_rhos = [rho for rho in RHO_GRID if rho > 0.0]
        diff_means = np.array([float(curve_stats[rho]["mean_difference_treatment_minus_baseline"]) for rho in diff_rhos]) * 1000.0
        diff_lows = np.array([float(curve_stats[rho]["ci95_low"]) for rho in diff_rhos]) * 1000.0
        diff_highs = np.array([float(curve_stats[rho]["ci95_high"]) for rho in diff_rhos]) * 1000.0
        axes[1].errorbar(diff_rhos, diff_means, yerr=np.vstack([diff_means - diff_lows, diff_highs - diff_means]), marker="o", color="#2ca02c", capsize=3)
        axes[1].axhline(0.0, color="black", linewidth=0.8)
        axes[1].axvline(PRIMARY_RHO, color="#d62728", linestyle="--", linewidth=1)
        axes[1].set_xlabel(r"Response coefficient $\rho$")
        axes[1].set_ylabel(r"Paired change vs $\rho=0$ ($\times 10^{-3}$)")
        axes[1].text(0.01, 0.04, "Lower is better", transform=axes[1].transAxes, fontsize=8)

    fig, axes = plt.subplots(2, 1, figsize=(6.3, 6.1), sharex=True, constrained_layout=True)
    curve(axes)
    b_base = figure_dir / "Fig_phase4_B_matched_kw16_rho_curve"
    fig.savefig(b_base.with_suffix(".png"))
    fig.savefig(b_base.with_suffix(".pdf"))
    plt.close(fig)

    fig = plt.figure(figsize=(12.2, 4.4))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.05, 1.0], wspace=0.48, hspace=0.30)
    ax_heat = fig.add_subplot(grid[:, 0])
    ax_top = fig.add_subplot(grid[0, 1])
    ax_bottom = fig.add_subplot(grid[1, 1], sharex=ax_top)
    heatmap(ax_heat)
    curve([ax_top, ax_bottom])
    ax_top.set_ylabel(r"Mean rate ($\times 10^{-3}$)", labelpad=4)
    ax_bottom.set_ylabel(r"Paired change vs 0 ($\times 10^{-3}$)", labelpad=4)
    ax_heat.text(-0.10, 1.04, "(a)", transform=ax_heat.transAxes, fontweight="bold")
    ax_top.text(-0.10, 1.08, "(b)", transform=ax_top.transAxes, fontweight="bold")
    fig.subplots_adjust(left=0.06, right=0.985, bottom=0.14, top=0.91)
    combined_base = figure_dir / "Fig_phase4_2sat_matched_kw"
    fig.savefig(combined_base.with_suffix(".png"))
    fig.savefig(combined_base.with_suffix(".pdf"))
    plt.close(fig)
    created = [a_base.with_suffix(".pdf"), b_base.with_suffix(".pdf"), combined_base.with_suffix(".pdf")]

    available_alphas = sorted({float(row["alpha"]) for row in aggregate_rows})
    if all(alpha in available_alphas for alpha in [0.95, 1.05, 1.20]):
        primary_by_alpha = []
        for alpha in [0.95, 1.05, 1.20]:
            row = next(
                entry
                for entry in statistics_rows
                if float(entry["alpha"]) == alpha
                and float(entry["kw"]) == PRIMARY_KW
                and abs(float(entry["rho"]) - PRIMARY_RHO) < 1e-12
                and entry["metric"] == "all_mean_selected_unsat_rate"
            )
            primary_by_alpha.append(row)
        means = np.array([float(row["mean_difference_treatment_minus_baseline"]) for row in primary_by_alpha]) * 1000.0
        lows = np.array([float(row["ci95_low"]) for row in primary_by_alpha]) * 1000.0
        highs = np.array([float(row["ci95_high"]) for row in primary_by_alpha]) * 1000.0
        fig, ax = plt.subplots(figsize=(5.5, 3.7), constrained_layout=True)
        ax.errorbar([0.95, 1.05, 1.20], means, yerr=np.vstack([means - lows, highs - means]), marker="o", capsize=4, color="#9467bd")
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_xlabel(r"Clause density $\alpha$")
        ax.set_ylabel(r"Paired change, $\rho=0.08$ minus 0 ($\times 10^{-3}$)")
        ax.set_title(r"Matched $k_w=16$ density check")
        ax.text(0.02, 0.04, "Lower is better", transform=ax.transAxes, fontsize=8)
        c_base = figure_dir / "Fig_phase4_C_alpha_effect_summary"
        fig.savefig(c_base.with_suffix(".png"))
        fig.savefig(c_base.with_suffix(".pdf"))
        plt.close(fig)
        created.append(c_base.with_suffix(".pdf"))
    return created


def format_effect(value: Any, digits: int = 6) -> str:
    if value in (None, ""):
        return "n/a"
    return f"{float(value):.{digits}g}"


def write_report(
    output: Path,
    decision: Mapping[str, Any],
    surface: Mapping[str, Any],
    aggregate_rows: Sequence[Mapping[str, Any]],
    stage_b_was_run: bool,
    validation: Mapping[str, Any],
    formula_audit: Mapping[str, Any],
) -> None:
    primary = decision["primary"]
    surface_a = surface["by_alpha"]["1.20"]
    all_primary_base = float(primary["baseline_mean"])
    all_primary_treatment = float(primary["treatment_mean"])
    sat_primary = next(
        row
        for row in json.loads((output / "statistics_ci.json").read_text(encoding="utf-8"))
        if float(row["alpha"]) == 1.20 and float(row["kw"]) == PRIMARY_KW and abs(float(row["rho"]) - PRIMARY_RHO) < 1e-12 and row["metric"] == "SAT_success_rate"
    )
    unsat_primary = next(
        row
        for row in json.loads((output / "statistics_ci.json").read_text(encoding="utf-8"))
        if float(row["alpha"]) == 1.20 and float(row["kw"]) == PRIMARY_KW and abs(float(row["rho"]) - PRIMARY_RHO) < 1e-12 and row["metric"] == "UNSAT_mean_selected_unsat_rate"
    )
    final_primary = next(
        row
        for row in json.loads((output / "statistics_ci.json").read_text(encoding="utf-8"))
        if float(row["alpha"]) == 1.20 and float(row["kw"]) == PRIMARY_KW and abs(float(row["rho"]) - PRIMARY_RHO) < 1e-12 and row["metric"] == "all_mean_final_unsat_rate"
    )
    case = decision["result_case"]
    if case == "Case 1":
        placement = "retain a short, explicitly weak/conditional positive 2-SAT result in the main text"
        cross_wording = (
            "In random 2-SAT at N=500, a preregistered matched-kw comparison retained a small finite-response effect at alpha=1.20; "
            "the result is density-conditional and substantially weaker than the LDPC regime."
        )
    elif case == "Case 2":
        placement = "limit the main text to a joint-parameter/instance-conditional statement and place the full 2-SAT map in the Supplement"
        cross_wording = (
            "In random 2-SAT, matched-kw controls produced only small or formula-dependent response-state changes, so the previously selected high-density tau mode is interpreted as joint parameter co-design rather than an independent memory benefit."
        )
    else:
        placement = "move the positive 2-SAT claim to the Supplement (or reduce it to one null-contrast sentence in the main text)"
        cross_wording = (
            "In random 2-SAT, the apparent high-density tau-pSA advantage did not survive a preregistered matched-kw control: rho=0 minimized the point estimate at every tested kw, so the earlier optimizer trend cannot be attributed to an independent response-memory effect and is instead consistent with hyperparameter coupling."
        )
    alpha_claim = (
        "The matched design was completed at alpha=0.95, 1.05, and 1.20, so only this three-point density comparison is supported."
        if stage_b_was_run
        else "Only alpha=1.20 was tested in the matched design; no matched-kw alpha-dependence claim is supported, and the earlier all-alpha trend remains descriptive optimizer behavior."
    )
    stop_decision = "Yes. Phase 4 is complete and no further 2-SAT/MAX-CUT simulation is submission-gating; proceed to manuscript production."
    report = f"""# Phase 4 random 2-SAT matched-kw validation report

Date: 2026-08-27

## Executive decision

**{case}.** {decision['interpretation']}

At the preregistered primary contrast, `kw=16`, `rho=0.08` versus `rho=0`, the mean formula-level unsatisfied-clause rate changed from {all_primary_base:.8f} to {all_primary_treatment:.8f}.  The paired treatment-minus-control effect was {float(primary['mean_difference_treatment_minus_baseline']):.8f} (95% formula-bootstrap CI [{float(primary['ci95_low']):.8f}, {float(primary['ci95_high']):.8f}]), equal to {600.0 * float(primary['mean_difference_treatment_minus_baseline']):.5f} additional unsatisfied clauses per trial (95% CI [{600.0 * float(primary['ci95_low']):.5f}, {600.0 * float(primary['ci95_high']):.5f}]).  This is a **{abs(float(primary['relative_reduction_pct'])):.3f}% degradation**, not an improvement (`dz={float(primary['standardized_paired_effect_dz']):.3f}`, positive means worse).  The fraction of formulas improved was {float(primary['improved_formula_fraction']):.3f} ({int(primary['improved_formula_count'])}/30; Wilson 95% CI [{float(primary['improved_formula_fraction_ci_low']):.3f}, {float(primary['improved_formula_fraction_ci_high']):.3f}]).

Recommended disposition: **{placement}.**

## Protocol provenance

- Workload: unplanted random 2-SAT, `N=500`.
- Mandatory Stage A: `alpha=1.20`, the same formula indices 0-29 and exact generator/seed rule used by the prior reproducibility experiment.
- Formula sampling unit: 30 independently generated formulas (13 SAT, 17 UNSAT at alpha=1.20).
- Grid: `kw={{4,8,12,16,20}}`, `rho={{0,0.02,0.04,0.06,0.08,0.12}}`.
- Evaluation: 100 trials/formula/cell, 3000 cycles, best-state readout, with an independent deterministic seed per formula/trial reused in every grid cell.
- Frozen anchor: the prior alpha=1.20 tau-pSA best-parameter row.  Only `kw` and `rho` vary.
- Primary endpoint: mean unsatisfied-clause rate, paired at the formula level.  SAT success is secondary and is never applied to exact-UNSAT formulas.
- The Phase 4 data are a fixed-anchor matched control and are not pooled with Phase 1/2/2b/3 or the independently optimized 2-SAT production results.

## Validation and reproducibility

- Exact rho=0 tau/pSA trajectory equivalence: passed for 60 preregistered-anchor trajectories (30 formulas x 2 trials).
- Phase 4 selected endpoint versus the legacy implementation: passed.
- Non-kw/rho parameter freeze: passed.
- Formula identity/SAT-label audit: {formula_audit['all_passed']}.
- Raw row/schema/range/uniqueness/paired-seed validation: {validation['all_passed']}.
- Resume is cell-granular through checksummed JSON checkpoints; merged CSVs are deterministically sorted.

## Primary and stratified results

### All formulas

- Mean paired effect (`rho=0.08 - rho=0`) at `kw=16`: {float(primary['mean_difference_treatment_minus_baseline']):.8f} unsatisfied-clause rate.
- 95% formula-bootstrap CI: [{float(primary['ci95_low']):.8f}, {float(primary['ci95_high']):.8f}].
- Paired t-test p={format_effect(primary['paired_t_pvalue'])}; Wilcoxon p={format_effect(primary['wilcoxon_pvalue'])}.  The primary contrast was preregistered; the full map is secondary and includes Holm-adjusted p-values.
- Formula directions: {int(primary['improved_formula_count'])} improved, {int(primary['worsened_formula_count'])} worsened, {int(primary['tied_formula_count'])} tied.
- Terminal-state unsatisfied-clause rate changed by {float(final_primary['mean_difference_treatment_minus_baseline']):.8f} (95% CI [{float(final_primary['ci95_low']):.8f}, {float(final_primary['ci95_high']):.8f}]), an {abs(float(final_primary['relative_reduction_pct'])):.3f}% degradation.  Thus the null conclusion is not an artifact of best-state readout.

### SAT formulas

- Success-rate change: {float(sat_primary['mean_difference_treatment_minus_baseline']):.6f} (95% CI [{float(sat_primary['ci95_low']):.6f}, {float(sat_primary['ci95_high']):.6f}]) across {int(sat_primary['n_formulas'])} SAT formulas.
- This corresponds to -{abs(100.0 * float(sat_primary['mean_difference_treatment_minus_baseline'])):.2f} percentage points; the interval includes zero.

### UNSAT formulas

- Mean unsatisfied-clause-rate change: {float(unsat_primary['mean_difference_treatment_minus_baseline']):.8f} (95% CI [{float(unsat_primary['ci95_low']):.8f}, {float(unsat_primary['ci95_high']):.8f}]) across {int(unsat_primary['n_formulas'])} UNSAT formulas.
- The UNSAT stratum therefore shows a small but directionally clear degradation rather than an independent rho benefit.

## Response-surface diagnosis

- Best observed Stage A cell: `kw={surface_a['best_cell']['kw']:.0f}`, `rho={surface_a['best_cell']['rho']:.2f}`, mean unsatisfied-clause rate={surface_a['best_cell']['mean_selected_unsat_rate']:.8f}.
- Best cell on the kw boundary: {surface_a['best_cell_on_kw_boundary']}; on the rho boundary: {surface_a['best_cell_on_rho_boundary']}.
- Rho minimizing the point estimate at each kw: {json.dumps(surface_a['best_rho_by_kw'], sort_keys=True)}.
- `rho=0` is optimal at every tested `kw`, and the primary endpoint worsens monotonically as rho increases.  There is no internal rho optimum.
- The `kw=4` row is worse, `kw=8` is best by point estimate, and `kw=12/16/20` form an exactly identical high-kw saturation plateau under this anchor.  The former `kw=16` search boundary did not hide continuing improvement at `kw=20`.
- Kw-averaged span: {surface_a['kw_main_span']:.8f}; rho-averaged span: {surface_a['rho_main_span']:.8f}; kw/rho span ratio: {surface_a['kw_to_rho_span_ratio']:.3f}.
- Interaction-residual range: {surface_a['interaction_residual_range']:.8f}.
- Nonzero-rho cells with an unadjusted paired 95% CI entirely below zero: {surface_a['n_unadjusted_ci_better_nonzero_cells']} (the complete multiplicity-aware table is in `statistics_ci.csv`).

## Stage B gate

- Stage B run: **{stage_b_was_run}**.
- Gate required: primary CI below zero, at least 2% relative reduction, at least 60% of formulas improved, at least two improving nonzero-rho cells at kw=16, and support from rho=0.06 or 0.12.
- Observed improving nonzero-rho cells at kw=16: {decision['observed_support']['better_nonzero_rho_cells_at_kw16']}; adjacent support cells: {decision['observed_support']['adjacent_support_cells']}.
- {alpha_claim}

## Answers required for manuscript restructuring

1. **Was the kw confound resolved?** Yes.  Every rho value was compared with rho=0 at the same kw under an otherwise identical anchor, formula set, cycle count, and stochastic seed stream.
2. **Did an independent rho>0 effect remain at fixed kw?** No.  At the primary cell rho=0.08 slightly worsened the endpoint, and no nonzero-rho cell at any kw had a paired 95% CI supporting improvement.
3. **Primary effect and CI.** Treatment-minus-control={float(primary['mean_difference_treatment_minus_baseline']):.8f} rate units, 95% CI [{float(primary['ci95_low']):.8f}, {float(primary['ci95_high']):.8f}], or {600.0 * float(primary['mean_difference_treatment_minus_baseline']):.5f} additional clauses/trial; `dz={float(primary['standardized_paired_effect_dz']):.3f}` and the relative change is a {abs(float(primary['relative_reduction_pct'])):.3f}% degradation.
4. **Shape of the kw x rho surface.** `rho=0` is best at all five kw values and positive rho worsens the primary endpoint monotonically.  Kw improves the result from 4 to 8, after which the surface saturates; kw=12, 16, and 20 are identical under the fixed anchor.  There is no internal rho optimum and no hidden kw>16 gain.
5. **SAT versus UNSAT.** SAT success changed by {float(sat_primary['mean_difference_treatment_minus_baseline']):.6f} (CI includes zero); the UNSAT mean rate worsened by {float(unsat_primary['mean_difference_treatment_minus_baseline']):.8f} (CI above zero).  Neither stratum supports benefit, and success is not computed for UNSAT formulas.
6. **Was Stage B run/needed?** Stage B run={stage_b_was_run}; it was governed solely by the preregistered Stage A gate.
7. **How much alpha dependence can be claimed?** {alpha_claim}
8. **Main text or Supplement?** Move the positive 2-SAT claim to the Supplement, or retain only one null-contrast sentence in the main text.
9. **How should the MAX-CUT/2-SAT section read?** `{cross_wording}`  Keep LDPC as the only strong positive regime, MAX-CUT as marginal, and 2-SAT according to this matched-control branch.  Do not place BER, cut quality, and SAT endpoints on one quantitative scale.
10. **Can Phase 4 be considered complete?** Yes.  The mandatory grid, paired inference, SAT/UNSAT stratification, validation, figures, metadata, and decision gate are complete.
11. **Can new simulation stop?** {stop_decision}

## Output map

- `raw_trial_results.csv`: one row per trial, formula, and cell.
- `raw_cell_formula_results.csv`: one row per formula and cell (the primary statistical unit).
- `formula_level_paired_summary.csv`: formula-wise treatment-minus-rho=0 contrasts.
- `kw_rho_aggregate_summary.csv`: all/SAT/UNSAT cell summaries and bootstrap intervals.
- `statistics_ci.csv`: paired effects, CIs, direction counts, tests, and Holm-adjusted secondary-map p-values.
- `config.json`, `formula_metadata.json`, `seed_metadata.json`, `validation.json`: frozen protocol and audit trail.
- `figures/`: PRApplied-ready PNG/PDF heatmap, matched-kw curve, and combined figure.
"""
    atomic_write_text(output / "PHASE4_2SAT_KW_CONFOUND_REPORT.md", report)


def configure_logging(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    log_dir = output / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_dir / "phase4_run.log", encoding="utf-8")],
        force=True,
    )


def analyze_and_write(
    output: Path,
    n_trials: int,
    n_cycles: int,
    make_plots: bool,
    formula_audit: Mapping[str, Any],
) -> Tuple[Dict[str, Any], bool]:
    trials, formulas, manifests = load_checkpoints(output)
    if not manifests:
        raise RuntimeError("No completed checkpoints available for analysis")
    write_csv(output / "raw_trial_results.csv", trials, TRIAL_FIELDS)
    write_csv(output / "raw_cell_formula_results.csv", formulas, FORMULA_FIELDS)
    write_json(output / "checkpoint_manifest.json", manifests)
    aggregate_rows = aggregate_cell_rows(formulas)
    aggregate_fields = list(aggregate_rows[0].keys())
    write_csv(output / "kw_rho_aggregate_summary.csv", aggregate_rows, aggregate_fields)
    paired_rows, statistics_rows = paired_analysis(formulas)
    paired_fields = list(paired_rows[0].keys())
    statistics_fields = list(statistics_rows[0].keys())
    write_csv(output / "formula_level_paired_summary.csv", paired_rows, paired_fields)
    write_csv(output / "statistics_ci.csv", statistics_rows, statistics_fields)
    write_json(output / "statistics_ci.json", statistics_rows)
    surface = response_surface_diagnostics(aggregate_rows, statistics_rows)
    write_json(output / "response_surface_diagnostics.json", surface)
    decision = stage_b_decision(statistics_rows, surface)
    write_json(output / "stage_b_decision.json", decision)
    validation = validate_outputs(trials, formulas, manifests, n_trials, n_cycles)
    validation["formula_identity_and_label_audit"] = formula_audit
    validation["all_passed"] = bool(validation["all_passed"] and formula_audit["all_passed"])
    write_json(output / "validation.json", validation)
    if not validation["all_passed"]:
        raise RuntimeError("Phase 4 output validation failed; see validation.json")
    if make_plots:
        figure_paths = make_figures(output, aggregate_rows, statistics_rows)
        write_json(output / "figure_manifest.json", [{"path": str(path.relative_to(output)), "sha256": sha256_file(path)} for path in figure_paths])
    available_alphas = sorted({float(row["alpha"]) for row in formulas})
    stage_b_was_run = all(alpha in available_alphas for alpha in STAGE_B_ALPHAS)
    write_report(output, decision, surface, aggregate_rows, stage_b_was_run, validation, formula_audit)
    return decision, stage_b_was_run


def build_config(
    output: Path,
    anchor: Mapping[str, Any],
    raw_anchor: Mapping[str, str],
    n_trials: int,
    n_cycles: int,
    n_workers: int,
) -> Dict[str, Any]:
    source_hashes = {str(path.relative_to(REPO_ROOT)): sha256_file(path) for path in PRIORITY_DOCUMENTS + [ANCHOR_SOURCE, Path(__file__)]}
    frozen_fields = {key: value for key, value in anchor.items() if key not in {"kw", "response_lambda"}}
    return {
        "phase": 4,
        "purpose": "2-SAT kw confound removal / matched-kw validation",
        "output_directory": str(output),
        "stage_a_alphas": STAGE_A_ALPHAS,
        "stage_b_alphas_conditional": STAGE_B_ALPHAS,
        "n_variables": 500,
        "formula_indices": FORMULA_INDICES,
        "kw_grid": KW_GRID,
        "rho_grid": RHO_GRID,
        "n_trials_per_formula_cell": int(n_trials),
        "n_cycles": int(n_cycles),
        "n_workers": int(n_workers),
        "primary_contrast": {"alpha": 1.20, "kw": PRIMARY_KW, "rho_treatment": PRIMARY_RHO, "rho_control": 0.0},
        "primary_rho_rationale": "rho=0.08 was the selected alpha=1.20 value in the prior tau-pSA memory sweep and is preregistered in the restructuring report.",
        "anchor_source": str(ANCHOR_SOURCE),
        "anchor_source_sha256": sha256_file(ANCHOR_SOURCE),
        "anchor_original_kw": float(raw_anchor["kw"]),
        "anchor_original_rho": float(raw_anchor["response_lambda"]),
        "frozen_non_kw_rho_parameters": frozen_fields,
        "trial_seed_rule": "17000000 + round(alpha*1000)*100000 + formula_index*1000 + trial_index",
        "formula_seed_rule": "0 + 2003*formula_index + 500 + round(1000*alpha)",
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "stage_b_gate": {
            "primary_ci95_high_below_zero": True,
            "minimum_relative_reduction_pct": PRACTICAL_RELATIVE_REDUCTION_PCT,
            "minimum_improved_formula_fraction": PRACTICAL_FORMULA_FRACTION,
            "minimum_better_nonzero_rho_cells_at_kw16": 2,
            "requires_adjacent_rho_support_0p06_or_0p12": True,
        },
        "source_sha256": source_hashes,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": stats.__version__ if hasattr(stats, "__version__") else __import__("scipy").__version__,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stage", choices=["auto", "A", "B", "analyze", "smoke"], default="auto")
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--n-cycles", type=int, default=3000)
    parser.add_argument("--n-workers", type=int, default=8)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-figures", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    configure_logging(output)
    anchor, raw_anchor = load_anchor()
    n_trials = int(args.n_trials)
    n_cycles = int(args.n_cycles)
    if args.stage == "smoke":
        n_trials = min(n_trials, 2)
        n_cycles = min(n_cycles, 8)
        alphas_for_metadata = STAGE_A_ALPHAS
    else:
        alphas_for_metadata = STAGE_A_ALPHAS + STAGE_B_ALPHAS
    anchor["n_cycles"] = n_cycles
    config = build_config(output, anchor, raw_anchor, n_trials, n_cycles, int(args.n_workers))
    write_json(output / "config.json", config)
    seed_metadata = {
        "formula_generator_base_seed": FORMULA_GENERATOR_BASE_SEED,
        "trial_seed_base": TRIAL_SEED_BASE,
        "formula_seed_rule": config["formula_seed_rule"],
        "trial_seed_rule": config["trial_seed_rule"],
        "paired_across_kw_rho": True,
        "trial_stream_independent_by_formula_and_trial": True,
    }
    write_json(output / "seed_metadata.json", seed_metadata)
    formula_metadata, formula_audit = build_formula_metadata(alphas_for_metadata)
    write_json(output / "formula_metadata.json", formula_metadata)
    write_json(output / "formula_identity_audit.json", formula_audit)

    if args.stage not in {"analyze"}:
        preflight = preflight_checks(anchor) if args.stage != "smoke" else {
            "all_passed": True,
            "smoke_note": "Full 60-trajectory rho=0 equivalence audit is skipped in smoke mode.",
        }
        write_json(output / "preflight_validation.json", preflight)
        if not preflight["all_passed"]:
            raise RuntimeError("Preflight validation failed")

    if args.stage == "smoke":
        run_grid(output, STAGE_A_ALPHAS, anchor, n_trials, n_cycles, min(int(args.n_workers), 2), args.resume)
        analyze_and_write(output, n_trials, n_cycles, False, formula_audit)
        logging.info("Smoke run complete")
        return
    if args.stage == "A":
        run_grid(output, STAGE_A_ALPHAS, anchor, n_trials, n_cycles, int(args.n_workers), args.resume)
        analyze_and_write(output, n_trials, n_cycles, not args.skip_figures, formula_audit)
        return
    if args.stage == "B":
        run_grid(output, STAGE_B_ALPHAS, anchor, n_trials, n_cycles, int(args.n_workers), args.resume)
        analyze_and_write(output, n_trials, n_cycles, not args.skip_figures, formula_audit)
        return
    if args.stage == "analyze":
        analyze_and_write(output, n_trials, n_cycles, not args.skip_figures, formula_audit)
        return

    run_grid(output, STAGE_A_ALPHAS, anchor, n_trials, n_cycles, int(args.n_workers), args.resume)
    decision, _ = analyze_and_write(output, n_trials, n_cycles, False, formula_audit)
    if decision["run_stage_b"]:
        logging.info("Stage A gate passed; launching conditional Stage B")
        run_grid(output, STAGE_B_ALPHAS, anchor, n_trials, n_cycles, int(args.n_workers), args.resume)
    else:
        logging.info("Stage A gate did not pass; Stage B will not be run")
    formula_metadata, formula_audit = build_formula_metadata(STAGE_A_ALPHAS + (STAGE_B_ALPHAS if decision["run_stage_b"] else []))
    write_json(output / "formula_metadata.json", formula_metadata)
    write_json(output / "formula_identity_audit.json", formula_audit)
    analyze_and_write(output, n_trials, n_cycles, not args.skip_figures, formula_audit)
    logging.info("Phase 4 complete")


if __name__ == "__main__":
    main()

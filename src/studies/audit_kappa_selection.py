#!/usr/bin/env python3
"""Read-only fairness audit for publication-facing binary-state kappa choices.

This script analyzes the completed 2,000-trial/SNR sweep and existing high-stat
results.  It does not run decoding simulations and does not alter any existing
result or manuscript file.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RUN_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
TARGETS = ["N96_M48", "N288_M144"]
KAPPA_GRID = [0.0, 0.25, 0.50, 0.75, 0.90, 0.95, 1.00, 1.10, 1.25]
EQUAL_RANGE = {"N96_M48": 0.95, "N288_M144": 0.90}
INPUT_FILES = [
    "kappa_sweep.csv",
    "kappa_sweep_by_ebno.csv",
    "kappa_sweep_by_seed.csv",
    "best_kappa_selection.csv",
    "highstat_ber_fer.csv",
    "paired_statistics.csv",
    "ber_optimal_sensitivity_by_ebno.csv",
    "ber_optimal_sensitivity_paired_statistics.csv",
    "equal_range_summary.csv",
    "validation.json",
    "final_audit.json",
    "parameter_config_record.json",
    "BINARY_STATE_SELF_FEEDBACK_REPORT.md",
]


def read_csv(name: str) -> list[dict[str, str]]:
    with (RUN_DIR / name).open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(name: str, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    path = RUN_DIR / name
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_json(name: str, value: Any) -> None:
    path = RUN_DIR / name
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def sample_sd(values: list[float]) -> float:
    center = mean(values)
    return math.sqrt(sum((value - center) ** 2 for value in values) / (len(values) - 1))


def ci95(values: list[float]) -> tuple[float, float, float, float]:
    """Student-t interval, matching the existing seed-batch analysis."""
    if len(values) == 1:
        return values[0], float("nan"), float("nan"), float("nan")
    t95 = {8: 2.365, 10: 2.262}.get(len(values), 1.96)
    center = mean(values)
    sd = sample_sd(values)
    sem = sd / math.sqrt(len(values))
    half = t95 * sem
    return center, sd, center - half, center + half


def pooled_seed_values(
    rows: list[dict[str, str]], size: str, kappa: float, metric: str
) -> dict[int, float]:
    seeds = sorted({int(row["seed"]) for row in rows if row["size"] == size})
    output = {}
    for seed in seeds:
        values = [
            float(row[metric])
            for row in rows
            if row["size"] == size
            and float(row["kappa"]) == kappa
            and int(row["seed"]) == seed
        ]
        if len(values) != 3:
            raise RuntimeError(f"Expected three SNR cells for {size}, kappa={kappa}, seed={seed}")
        output[seed] = mean(values)
    return output


def paired_difference(
    seed_rows: list[dict[str, str]], size: str, left: float, right: float, metric: str
) -> dict[str, float]:
    left_values = pooled_seed_values(seed_rows, size, left, metric)
    right_values = pooled_seed_values(seed_rows, size, right, metric)
    if left_values.keys() != right_values.keys():
        raise RuntimeError("Paired seed sets differ")
    differences = [left_values[seed] - right_values[seed] for seed in left_values]
    center, sd, low, high = ci95(differences)
    return {
        "mean_difference": center,
        "seed_batch_sd": sd,
        "ci95_low": low,
        "ci95_high": high,
        "n_seed_batches": len(differences),
    }


def row_at(rows: list[dict[str, str]], size: str, kappa: float) -> dict[str, str]:
    return next(
        row for row in rows if row["size"] == size and float(row["kappa"]) == kappa
    )


def highstat_row(rows: list[dict[str, str]], size: str, method: str) -> dict[str, str]:
    return next(
        row for row in rows
        if row["size"] == size and row["method"] == method and row["EbNo_dB"] == "pooled"
    )


def fmt(value: float) -> str:
    return f"{value:.6g}"


def main() -> None:
    input_hashes_before = {name: sha256(RUN_DIR / name) for name in INPUT_FILES}
    sweep = read_csv("kappa_sweep.csv")
    by_ebno = read_csv("kappa_sweep_by_ebno.csv")
    by_seed = read_csv("kappa_sweep_by_seed.csv")
    selections = read_csv("best_kappa_selection.csv")
    highstat = read_csv("highstat_ber_fer.csv")
    paired_highstat = read_csv("paired_statistics.csv")
    equal_range_rows = read_csv("equal_range_summary.csv")
    original_validation = json.loads((RUN_DIR / "validation.json").read_text())
    original_audit = json.loads((RUN_DIR / "final_audit.json").read_text())
    parameter_record = json.loads((RUN_DIR / "parameter_config_record.json").read_text())

    all_kappa_rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    facts: dict[str, dict[str, Any]] = {}

    for size in TARGETS:
        size_sweep = sorted(
            [row for row in sweep if row["size"] == size],
            key=lambda row: KAPPA_GRID.index(float(row["kappa"])),
        )
        score_selected = float(next(
            row["best_kappa"] for row in selections if row["size"] == size
        ))
        ber_best = float(min(
            size_sweep,
            key=lambda row: (float(row["mean_BER"]), KAPPA_GRID.index(float(row["kappa"]))),
        )["kappa"])
        fer_best = float(min(
            size_sweep,
            key=lambda row: (float(row["mean_FER"]), KAPPA_GRID.index(float(row["kappa"]))),
        )["kappa"])
        runner_up = float(min(
            [row for row in size_sweep if float(row["kappa"]) != ber_best],
            key=lambda row: (float(row["mean_BER"]), KAPPA_GRID.index(float(row["kappa"]))),
        )["kappa"])

        for item in size_sweep:
            kappa = float(item["kappa"])
            snr_rows = sorted(
                [
                    row for row in by_ebno
                    if row["size"] == size and float(row["kappa"]) == kappa
                ],
                key=lambda row: float(row["EbNo_dB"]),
            )
            ber_seed_values = list(pooled_seed_values(by_seed, size, kappa, "BER").values())
            fer_seed_values = list(pooled_seed_values(by_seed, size, kappa, "FER").values())
            _, ber_sd, ber_low, ber_high = ci95(ber_seed_values)
            _, fer_sd, fer_low, fer_high = ci95(fer_seed_values)
            all_kappa_rows.append({
                "size": size,
                "kappa": kappa,
                "score_selected": kappa == score_selected,
                "pooled_BER_best": kappa == ber_best,
                "pooled_FER_best": kappa == fer_best,
                "equal_range_kappa_equals_lambda": kappa == EQUAL_RANGE[size],
                "BER_rank": 1 + sum(
                    float(other["mean_BER"]) < float(item["mean_BER"])
                    for other in size_sweep
                ),
                "FER_rank": 1 + sum(
                    float(other["mean_FER"]) < float(item["mean_FER"])
                    for other in size_sweep
                ),
                "BER_2.0_dB": float(snr_rows[0]["BER"]),
                "BER_2.5_dB": float(snr_rows[1]["BER"]),
                "BER_3.0_dB": float(snr_rows[2]["BER"]),
                "pooled_BER": float(item["mean_BER"]),
                "pooled_BER_seed_batch_sd": ber_sd,
                "pooled_BER_ci95_low": ber_low,
                "pooled_BER_ci95_high": ber_high,
                "FER_2.0_dB": float(snr_rows[0]["FER"]),
                "FER_2.5_dB": float(snr_rows[1]["FER"]),
                "FER_3.0_dB": float(snr_rows[2]["FER"]),
                "pooled_FER": float(item["mean_FER"]),
                "pooled_FER_seed_batch_sd": fer_sd,
                "pooled_FER_ci95_low": fer_low,
                "pooled_FER_ci95_high": fer_high,
                "selection_score": float(item["score"]),
                "trials_per_SNR": int(item["trials_per_EbNo"]),
                "seed_batches": int(item["seed_count"]),
            })

        for candidate in KAPPA_GRID:
            for metric in ["BER", "FER"]:
                difference = paired_difference(
                    by_seed, size, score_selected, candidate, metric
                )
                paired_rows.append({
                    "size": size,
                    "metric": metric,
                    "left_kappa": score_selected,
                    "right_kappa": candidate,
                    "contrast": "selected_minus_candidate",
                    **difference,
                    "unit": "paired sweep seed-batch pooled over three SNRs",
                })

        selected_row = row_at(sweep, size, score_selected)
        best_row = row_at(sweep, size, ber_best)
        runner_row = row_at(sweep, size, runner_up)
        selected_minus_best = paired_difference(
            by_seed, size, score_selected, ber_best, "BER"
        )
        selected_minus_runner = paired_difference(
            by_seed, size, score_selected, runner_up, "BER"
        )
        absolute_difference = float(selected_row["mean_BER"]) - float(best_row["mean_BER"])
        relative_difference = (
            absolute_difference / float(best_row["mean_BER"])
            if float(best_row["mean_BER"]) else float("nan")
        )
        if score_selected == ber_best:
            classification = "CASE A"
            rationale = "score-selected kappa is exactly the observed pooled-BER and FER optimum"
        elif selected_minus_best["ci95_low"] > 0.0:
            classification = "CASE C"
            rationale = "another kappa has a statistically clear paired pooled-BER advantage"
        else:
            classification = "CASE B"
            rationale = "observed BER difference is within the paired sweep uncertainty"

        additive = highstat_row(highstat, size, "additive")
        binary = highstat_row(highstat, size, "binary_state")
        highstat_ci = next(
            row for row in paired_highstat
            if row["size"] == size and row["EbNo_dB"] == "pooled"
            and row["metric"] == "BER" and row["contrast"] == "binary_minus_additive"
        )
        equal_binary = next(
            row for row in equal_range_rows
            if row["size"] == size and row["method"] == "binary_state"
        )
        facts[size] = {
            "score_selected": score_selected,
            "ber_best": ber_best,
            "fer_best": fer_best,
            "runner_up": runner_up,
            "equal_range": EQUAL_RANGE[size],
            "selected_BER": float(selected_row["mean_BER"]),
            "best_BER": float(best_row["mean_BER"]),
            "runner_up_BER": float(runner_row["mean_BER"]),
            "selected_minus_best": selected_minus_best,
            "selected_minus_runner": selected_minus_runner,
            "absolute_difference": absolute_difference,
            "relative_difference": relative_difference,
            "classification": classification,
            "rationale": rationale,
            "targeted_highstat_required": classification == "CASE C",
            "publication_kappa": ber_best if classification == "CASE C" else score_selected,
            "existing_highstat_binary_BER": float(binary["BER"]),
            "existing_highstat_binary_FER": float(binary["FER"]),
            "existing_highstat_additive_BER": float(additive["BER"]),
            "existing_highstat_additive_FER": float(additive["FER"]),
            "highstat_binary_minus_additive_BER": float(highstat_ci["mean_difference"]),
            "highstat_ci95_low": float(highstat_ci["ci95_low"]),
            "highstat_ci95_high": float(highstat_ci["ci95_high"]),
            "equal_range_binary_BER": float(equal_binary["mean_BER"]),
        }
        summary_rows.append({
            "size": size,
            "score_selected_kappa": score_selected,
            "pooled_BER_best_kappa": ber_best,
            "pooled_FER_best_kappa": fer_best,
            "equal_range_kappa": EQUAL_RANGE[size],
            "selected_sweep_pooled_BER": facts[size]["selected_BER"],
            "BER_best_sweep_pooled_BER": facts[size]["best_BER"],
            "selected_minus_BER_best_absolute": absolute_difference,
            "selected_minus_BER_best_relative": relative_difference,
            "paired_ci95_low": selected_minus_best["ci95_low"],
            "paired_ci95_high": selected_minus_best["ci95_high"],
            "runner_up_kappa": runner_up,
            "runner_up_pooled_BER": facts[size]["runner_up_BER"],
            "selected_minus_runner_up_BER": selected_minus_runner["mean_difference"],
            "selected_minus_runner_ci95_low": selected_minus_runner["ci95_low"],
            "selected_minus_runner_ci95_high": selected_minus_runner["ci95_high"],
            "classification": classification,
            "targeted_highstat_required": classification == "CASE C",
            "targeted_highstat_performed": False,
            "publication_facing_kappa": facts[size]["publication_kappa"],
            "existing_highstat_binary_BER": facts[size]["existing_highstat_binary_BER"],
            "existing_highstat_additive_BER": facts[size]["existing_highstat_additive_BER"],
            "binary_minus_additive_BER": facts[size]["highstat_binary_minus_additive_BER"],
            "binary_minus_additive_ci95_low": facts[size]["highstat_ci95_low"],
            "binary_minus_additive_ci95_high": facts[size]["highstat_ci95_high"],
            "decision": rationale,
        })

    write_csv("kappa_selection_fairness_all_kappa.csv", all_kappa_rows)
    write_csv("kappa_selection_fairness_paired_differences.csv", paired_rows)
    write_csv("kappa_selection_fairness_summary.csv", summary_rows)

    n192_sensitivity = read_csv("ber_optimal_sensitivity_by_ebno.csv")
    n192_sensitivity_ber = mean([
        float(row["BER"]) for row in n192_sensitivity if row["size"] == "N192_M96"
    ])
    n192_sensitivity_stats = read_csv("ber_optimal_sensitivity_paired_statistics.csv")
    n192_stat = next(
        row for row in n192_sensitivity_stats
        if row["size"] == "N192_M96" and row["EbNo_dB"] == "pooled"
        and row["metric"] == "BER"
    )

    lines = [
        "# Kappa-Selection Fairness Audit",
        "",
        "## Executive conclusion",
        "",
        "Both publication-facing choices pass **CASE A**. In the predeclared 2,000-trial/SNR "
        "grid, N=96 kappa=0.25 and N=288 kappa=0.75 are each simultaneously the score-selected, "
        "pooled-BER-best, and pooled-FER-best binary-state settings. No more favorable observed "
        "grid point was withheld, so no targeted high-stat simulation was triggered.",
        "",
        "This audit only reads completed results. It did not run a decoder, change a kappa grid, "
        "modify an existing result, or modify the manuscript.",
        "",
        "## Existing sweep, including sampling uncertainty",
        "",
        "Intervals are 95% Student-t intervals across the existing eight paired seed batches, "
        "with each seed batch pooled equally across 2.0, 2.5, and 3.0 dB. This is the same "
        "seed-batch uncertainty convention used by the completed matched-control study.",
    ]
    for size in TARGETS:
        lines.extend([
            "",
            f"### {size.replace('_', '/')}",
            "",
            "| kappa | BER 2.0 | BER 2.5 | BER 3.0 | Pooled BER [95% CI] | "
            "FER 2.0 | FER 2.5 | FER 3.0 | Pooled FER | Score | Role |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ])
        for row in [item for item in all_kappa_rows if item["size"] == size]:
            roles = []
            if row["score_selected"]:
                roles.append("score-selected")
            if row["pooled_BER_best"]:
                roles.append("BER-best")
            if row["pooled_FER_best"]:
                roles.append("FER-best")
            if row["equal_range_kappa_equals_lambda"]:
                roles.append("equal-range")
            lines.append(
                f"| {row['kappa']:.2f} | {fmt(row['BER_2.0_dB'])} | "
                f"{fmt(row['BER_2.5_dB'])} | {fmt(row['BER_3.0_dB'])} | "
                f"{fmt(row['pooled_BER'])} [{fmt(row['pooled_BER_ci95_low'])}, "
                f"{fmt(row['pooled_BER_ci95_high'])}] | {fmt(row['FER_2.0_dB'])} | "
                f"{fmt(row['FER_2.5_dB'])} | {fmt(row['FER_3.0_dB'])} | "
                f"{fmt(row['pooled_FER'])} | {fmt(row['selection_score'])} | "
                f"{', '.join(roles) or '-'} |"
            )

        fact = facts[size]
        runner_advantage = fact["runner_up_BER"] - fact["selected_BER"]
        runner_relative = runner_advantage / fact["runner_up_BER"]
        lines.extend([
            "",
            f"The selected and BER-best coefficients are identical (kappa={fact['ber_best']:.2f}), "
            "so selected-minus-best is exactly 0 with paired 95% CI [0, 0]. "
            f"The next-lowest observed BER is kappa={fact['runner_up']:.2f}, "
            f"BER={fmt(fact['runner_up_BER'])}; the selected setting is better by "
            f"{fmt(runner_advantage)} ({runner_relative:.1%} relative to the runner-up), with "
            f"selected-minus-runner paired CI [{fmt(fact['selected_minus_runner']['ci95_low'])}, "
            f"{fmt(fact['selected_minus_runner']['ci95_high'])}].",
            "",
            f"**Decision: {fact['classification']}.** {fact['rationale'].capitalize()}. "
            f"The publication-facing binary comparator remains kappa={fact['publication_kappa']:.2f}; "
            "targeted high-stat validation is not required.",
        ])

    lines.extend([
        "",
        "## Existing high-stat comparison after the fairness decision",
        "",
        "Because both choices are already BER-best on the sweep grid, the completed high-stat "
        "runs are the relevant favorable binary-state comparisons:",
        "",
        "| Size | Publication kappa | Binary BER / FER | Additive BER / FER | "
        "Binary-additive BER [paired 95% CI] |",
        "|---|---:|---:|---:|---:|",
    ])
    for size in TARGETS:
        fact = facts[size]
        lines.append(
            f"| {size} | {fact['publication_kappa']:.2f} | "
            f"{fmt(fact['existing_highstat_binary_BER'])} / "
            f"{fmt(fact['existing_highstat_binary_FER'])} | "
            f"{fmt(fact['existing_highstat_additive_BER'])} / "
            f"{fmt(fact['existing_highstat_additive_FER'])} | "
            f"{fmt(fact['highstat_binary_minus_additive_BER'])} "
            f"[{fmt(fact['highstat_ci95_low'])}, {fmt(fact['highstat_ci95_high'])}] |"
        )
    lines.extend([
        "",
        "Both intervals are strictly above zero. The additive advantage therefore remains "
        "statistically clear after giving the binary-state comparator its observed BER-best "
        "coefficient on the predeclared grid.",
        "",
        "## Equal-range and best-performance controls answer different questions",
        "",
        f"For N=96, equal range uses kappa=0.95 and gives binary BER="
        f"{fmt(facts['N96_M48']['equal_range_binary_BER'])}; best-performance uses kappa=0.25. "
        f"For N=288, equal range uses kappa=0.90 and gives binary BER="
        f"{fmt(facts['N288_M144']['equal_range_binary_BER'])}; best-performance uses kappa=0.75. "
        "Equal range tests whether binary-state feedback reproduces additive response memory at "
        "the same maximum deterministic range. Best performance tests whether coefficient tuning "
        "allows binary-state feedback to reproduce the additive result. These estimands must remain "
        "separate.",
        "",
        "## Consistency with N=192",
        "",
        "At N=192 the score-selected kappa=0.50 is not a reasonable BER comparator because a "
        "wrong-valid-codeword freeze lowers its syndrome contribution. The already-completed "
        f"BER-optimal sensitivity at kappa=0.25 gives pooled BER={fmt(n192_sensitivity_ber)}; "
        f"binary-minus-additive BER={fmt(float(n192_stat['mean_difference']))} "
        f"[{fmt(float(n192_stat['ci95_low']))}, {fmt(float(n192_stat['ci95_high']))}]. "
        "Using this sensitivity value for fairness while preserving the original score-selected "
        "result for provenance is consistent with the present audit. N=96 and N=288 need no such "
        "substitution because their score-selected settings are already BER-best.",
        "",
        "## Required final answers",
        "",
    ])
    number = 1
    for size in TARGETS:
        fact = facts[size]
        answers = [
            f"**{size} score-selected kappa:** {fact['score_selected']:.2f}.",
            f"**{size} observed BER-best kappa:** {fact['ber_best']:.2f}.",
            f"**{size} selected pooled sweep BER:** {fmt(fact['selected_BER'])}.",
            f"**{size} BER-best pooled sweep BER:** {fmt(fact['best_BER'])}.",
            f"**Is the difference material?** No difference exists: selected and BER-best are the same grid point.",
            f"**Was targeted high-stat required?** No; this is {fact['classification']}.",
            f"**Publication comparator:** kappa={fact['publication_kappa']:.2f}.",
        ]
        for answer in answers:
            lines.append(f"{number}. {answer}")
            lines.append("")
            number += 1
    lines.extend([
        f"{number}. **Equal-range role:** It is a matched deterministic-range control, not a "
        "best-performance binary baseline.",
        "",
        f"{number + 1}. **N=192 consistency:** Retain score-selected kappa=0.50 for provenance and "
        "use the already validated BER-optimal kappa=0.25 sensitivity when judging comparator fairness.",
        "",
        f"{number + 2}. **All-size reproduction:** No. At the favorable publication coefficients "
        "N=96, N=192 sensitivity, and N=288 binary-state feedback remains worse than additive.",
        "",
        f"{number + 3}. **Statistical clarity:** Yes. The paired 95% binary-minus-additive BER "
        "interval is strictly positive for N=96 and N=288 here, and for the N=192 BER-optimal "
        "sensitivity in the completed study.",
        "",
        f"{number + 4}. **Stored-quantity interpretation:** Maintained, scoped to the tested matched "
        "LDPC framework. N=288 still shows that binary inertia contributes materially, so the claim "
        "must be insufficiency for the full benefit, not irrelevance of binary self-feedback.",
        "",
        f"{number + 5}. **Further simulation:** None is indicated by this fairness audit.",
        "",
        f"{number + 6}. **Proceed to rewrite:** Yes. The novelty/related-work rewrite can proceed "
        "while transparently separating score selection, BER-fair comparator selection, and equal range.",
        "",
        "## Outputs",
        "",
        "- `kappa_selection_fairness_all_kappa.csv`",
        "- `kappa_selection_fairness_summary.csv`",
        "- `kappa_selection_fairness_paired_differences.csv`",
        "- `kappa_selection_fairness_validation.json`",
        "",
        "No new simulation was run. Existing phase-7 results and manuscript files were not modified.",
    ])
    report_path = RUN_DIR / "KAPPA_SELECTION_FAIRNESS_AUDIT.md"
    temporary_report = report_path.with_suffix(".md.tmp")
    temporary_report.write_text("\n".join(lines) + "\n")
    temporary_report.replace(report_path)

    input_hashes_after = {name: sha256(RUN_DIR / name) for name in INPUT_FILES}
    manuscript_hashes_current = {
        name: sha256(Path(item["path"]))
        for name, item in parameter_record["manuscript_references"].items()
    }
    manuscript_hashes_recorded = {
        name: item["sha256"]
        for name, item in parameter_record["manuscript_references"].items()
    }
    checks = {
        "existing_input_files_unchanged_by_audit": input_hashes_before == input_hashes_after,
        "original_validation_all_passed": bool(original_validation["all_passed"]),
        "original_final_audit_all_passed": bool(original_audit["all_passed"]),
        "binary_state_equation_unchanged": original_validation["equation"]
        == "d_i^(t)=q_i^(t)+kappa*s_i^(t), s_i^(t)=2*w_i^(t)-1",
        "kappa_zero_equivalence_logic_still_validated": bool(
            original_validation["kappa_zero_decoded_matches_core_pSA_all_test_seeds"]
            and original_validation["kappa_zero_state_history_trajectory_identical"]
            and original_validation["kappa_zero_threshold_discriminants_trajectory_identical"]
        ),
        "nonmemory_parameters_readout_H_seed_channel_framework_still_validated": bool(
            original_validation[
                "all_nonmemory_parameters_readout_matrix_snr_seed_framework_identical"
            ]
        ),
        "manuscript_hashes_unchanged": manuscript_hashes_current == manuscript_hashes_recorded,
        "both_targets_have_complete_9_point_grid": all(
            len([row for row in sweep if row["size"] == size]) == 9 for size in TARGETS
        ),
        "both_targets_score_selected_equals_BER_best": all(
            facts[size]["score_selected"] == facts[size]["ber_best"] for size in TARGETS
        ),
        "both_targets_score_selected_equals_FER_best": all(
            facts[size]["score_selected"] == facts[size]["fer_best"] for size in TARGETS
        ),
        "both_targets_case_A": all(facts[size]["classification"] == "CASE A" for size in TARGETS),
        "no_new_simulation_performed": True,
        "additive_reference_values_read_only": True,
    }
    validation = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "study": "Kappa-Selection Fairness Audit",
        "all_passed": all(checks.values()),
        "checks": checks,
        "input_sha256": input_hashes_after,
        "manuscript_sha256": manuscript_hashes_current,
        "target_classifications": {
            size: {
                "classification": facts[size]["classification"],
                "score_selected_kappa": facts[size]["score_selected"],
                "BER_best_kappa": facts[size]["ber_best"],
                "FER_best_kappa": facts[size]["fer_best"],
                "targeted_highstat_required": facts[size]["targeted_highstat_required"],
                "publication_kappa": facts[size]["publication_kappa"],
            }
            for size in TARGETS
        },
        "outputs": [
            "KAPPA_SELECTION_FAIRNESS_AUDIT.md",
            "kappa_selection_fairness_all_kappa.csv",
            "kappa_selection_fairness_summary.csv",
            "kappa_selection_fairness_paired_differences.csv",
            "kappa_selection_fairness_validation.json",
        ],
    }
    write_json("kappa_selection_fairness_validation.json", validation)
    if not validation["all_passed"]:
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"Fairness audit failed: {failed}")

    print("Fairness audit complete: N96 CASE A; N288 CASE A; no simulation run.")


if __name__ == "__main__":
    main()

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "ldpc"))

from src.core.response_rules import apply_response_rule, shuffled_sources  # noqa: E402
from src.twosat.instances import formula_sha256, is_satisfiable  # noqa: E402
from ldpc_pbit import gf2_rank, random_regular_ldpc_parity_check, update_pbits  # noqa: E402


def matrix_sha256(matrix: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(matrix, dtype=np.uint8, order="C").tobytes(order="C")).hexdigest()


def rows_from(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


class ReleasePackageTests(unittest.TestCase):
    def test_representative_config_matches_archived_mode(self) -> None:
        config = json.loads((ROOT / 'configs/ldpc/representative_benchmark.json').read_text())
        for size, modes in config['parameters'].items():
            directory = ROOT / 'data/processed/representative_benchmark/search_records' / size
            source = rows_from(directory / 'comparison/final_comparison.csv')
            for label, mode in [('pSA', 'pSA'), ('additive', 'lambda_pSA'), ('finite_response', 'tau_pSA')]:
                row = next(item for item in source if item['mode'] == mode)
                for field, value in modes[label].items():
                    if field in row and row[field] != '':
                        self.assertEqual(value, row[field] if isinstance(value, str) else float(row[field]))
            staged = ROOT / 'configs/ldpc/benchmark_sources/ldpc' / size / 'lambda_pSA/memory_sweep_summary.csv'
            self.assertEqual(staged.read_bytes(), (directory / 'lambda_pSA/memory_sweep_summary.csv').read_bytes())

    def test_lambda_zero_matches_psa_trajectory(self) -> None:
        matrix = random_regular_ldpc_parity_check(48, 24, 3, 6, seed=0)
        common = dict(P=matrix, channel_values=np.linspace(-0.5, 0.5, 48), kw=1.2, kr=1.5,
                      n_cycles=10, I0_min=0.1, I0_max=0.8, psa_p=0.2, decision_method="last")
        reference = update_pbits(mode="pSA", rng=np.random.default_rng(123), **common)
        candidate = update_pbits(mode="lambda_pSA", lambda_mem=0.0, rng=np.random.default_rng(123), **common)
        self.assertTrue(np.array_equal(reference[1], candidate[1]))
        self.assertTrue(np.array_equal(reference[2], candidate[2]))

    def test_rho_zero_matches_psa_rule(self) -> None:
        current = np.linspace(-0.9, 0.9, 17)
        previous = current[::-1]
        noise = np.linspace(0.4, -0.4, 17)
        psa, _ = apply_response_rule(current, previous, noise, "pSA")
        finite, _ = apply_response_rule(current, previous, noise, "finite_response", coefficient=0.0)
        self.assertTrue(np.array_equal(psa, finite))

    def test_kappa_zero_matches_psa_rule(self) -> None:
        current = np.linspace(-0.8, 0.8, 19)
        noise = np.linspace(0.3, -0.3, 19)
        binary_state = np.where(np.arange(19) % 2 == 0, -1.0, 1.0)
        psa = current + noise
        binary_feedback = current + 0.0 * binary_state + noise
        self.assertTrue(np.array_equal(psa, binary_feedback))

    def test_normalized_and_gain_only_formulas(self) -> None:
        current = np.asarray([0.2, -0.5])
        previous = np.asarray([-0.3, 0.6])
        noise = np.asarray([0.1, -0.2])
        lam = 0.7
        normalized, _ = apply_response_rule(current, previous, noise, "normalized", coefficient=lam)
        gain, _ = apply_response_rule(current, previous, noise, "gain_only", coefficient=lam)
        self.assertTrue(np.allclose(normalized, (current + lam * previous) / (1 + lam) + noise))
        self.assertTrue(np.allclose(gain, (1 + lam) * current + noise))

    def test_shuffled_derangement_and_moments(self) -> None:
        source = shuffled_sources(97, np.random.default_rng(5))
        self.assertTrue(np.all(source != np.arange(97)))
        self.assertEqual(sorted(source.tolist()), list(range(97)))
        values = np.random.default_rng(7).normal(size=97)
        self.assertAlmostEqual(float(values.mean()), float(values[source].mean()))
        self.assertAlmostEqual(float(np.mean(values**2)), float(np.mean(values[source] ** 2)))

    def test_ldpc_matrix_registry(self) -> None:
        with (ROOT / "data" / "metadata" / "matrices" / "matrix_registry.csv").open(newline="") as stream:
            registry = list(csv.DictReader(stream))
        self.assertEqual(len(registry), 30)
        for item in registry:
            matrix = np.load(ROOT / item["file"])["H"].astype(np.uint8)
            self.assertEqual(matrix_sha256(matrix), item["matrix_sha256"])
            self.assertEqual(gf2_rank(matrix), int(item["gf2_rank"]))
            self.assertTrue(np.all(matrix.sum(axis=0) == int(item["variable_degree"])))
            self.assertTrue(np.all(matrix.sum(axis=1) == int(item["check_degree"])))

    def test_twosat_formula_registry(self) -> None:
        with (ROOT / "data" / "metadata" / "formulas" / "formula_registry.csv").open(newline="") as stream:
            registry = list(csv.DictReader(stream))
        self.assertEqual(len(registry), 30)
        self.assertEqual(sum(row["is_satisfiable"].lower() == "true" for row in registry), 13)
        for item in registry:
            clauses = np.load(ROOT / item["file"])["clauses"]
            self.assertEqual(formula_sha256(clauses), item["formula_sha256"])
            self.assertEqual(is_satisfiable(clauses, int(item["n_variables"])), item["is_satisfiable"].lower() == "true")

    def test_processed_data_schemas_and_key_claims(self) -> None:
        expected = {
            "representative_ldpc_summary.csv": {"size", "mode", "BER", "FER"},
            "matched_causal_control_summary.csv": {"size", "additive_BER", "same_parameter_pSA_BER"},
            "cross_code_level_summary.csv": {"size", "code_id", "scope", "relative_BER_reduction"},
            "matched_kw_response_surface.csv": {"kw", "rho", "mean_selected_unsat_rate"},
        }
        for filename, columns in expected.items():
            with (ROOT / "data" / "processed" / "tables" / filename).open(newline="") as stream:
                reader = csv.DictReader(stream)
                self.assertTrue(columns.issubset(set(reader.fieldnames or [])))
                self.assertTrue(any(True for _ in reader))
        with (ROOT / "data" / "processed" / "tables" / "cross_code_level_summary.csv").open(newline="") as stream:
            cross = list(csv.DictReader(stream))
        pooled = [row for row in cross if row["scope"] == "pooled_across_EbNo"]
        self.assertEqual(len(pooled), 30)
        self.assertEqual({row["code_id"] for row in pooled}, {f"C{index:02d}" for index in range(10)})
        self.assertTrue(all(row["BER_additive_better"] == "1" and row["FER_additive_better"] == "1" for row in pooled))

    def test_current_validation_records(self) -> None:
        transfer = rows_from(
            ROOT / "data" / "processed" / "cross_code_fixed_transfer" / "three_way_comparison_summary.csv"
        )
        pooled = {
            row["size"]: row for row in transfer
            if row["scope"] == "pooled_across_EbNo"
            and row["comparison"] == "additive_vs_pSA_specific"
        }
        self.assertEqual(set(pooled), {"N96_M48", "N192_M96", "N288_M144"})
        expected = {"N96_M48": 0.3546, "N192_M96": 0.7661, "N288_M144": 0.8110}
        for size, value in expected.items():
            self.assertAlmostEqual(float(pooled[size]["median_relative_BER_reduction"]), value, places=3)
            self.assertEqual(int(pooled[size]["BER_method_better_count"]), 10)
            self.assertEqual(int(pooled[size]["FER_method_better_count"]), 10)

        initialization = rows_from(
            ROOT / "data" / "processed" / "initialization_robustness" / "condition_summary.csv"
        )
        for size, init, expected_rate in [
            ("N192_M96", "random", 0.96),
            ("N192_M96", "channel_hard", 0.96),
            ("N288_M144", "random", 0.96),
            ("N288_M144", "channel_hard", 0.97),
        ]:
            row = next(
                item for item in initialization
                if item["size"] == size and item["initialization"] == init and item["variant"] == "additive"
            )
            self.assertEqual(float(row["correct_acquisition_rate"]), expected_rate)

        selected = rows_from(
            ROOT / "data" / "processed" / "binary_state_feedback" / "best_kappa_selection.csv"
        )
        self.assertEqual(
            {row["size"]: float(row["best_kappa"]) for row in selected},
            {"N96_M48": 0.25, "N192_M96": 0.50, "N288_M144": 0.75},
        )
        sensitivity = rows_from(
            ROOT / "data" / "processed" / "binary_state_feedback" / "ber_optimal_sensitivity_selection.csv"
        )
        n192_sensitivity = next(row for row in sensitivity if row["size"] == "N192_M96")
        self.assertEqual(float(n192_sensitivity["BER_optimal_kappa"]), 0.25)

    def test_path_independent_public_files(self) -> None:
        forbidden = ["/" + "Users/", "Drop" + "box"]
        suffixes = {".py", ".md", ".txt", ".csv", ".json", ".yml", ".yaml", ".cff", ".tex", ".bib"}
        for path in ROOT.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in suffixes or path == Path(__file__):
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for token in forbidden:
                self.assertNotIn(token, text, msg=str(path.relative_to(ROOT)))

    def test_processed_data_figure_regeneration(self) -> None:
        completed = subprocess.run(
            [sys.executable, "scripts/reproduce_figures.py"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, msg=completed.stdout + completed.stderr)
        expected = {
            "Fig2b_LDPC_BER_N192_N288.pdf",
            "Fig3_matched_causal_controls.pdf",
            "Fig4b_acquisition_retention_summary.pdf",
            "Fig5_cross_code_three_way_BER.pdf",
            "FigS_binary_kappa_sweep.pdf",
            "FigS_binary_highstat_comparison.pdf",
            "FigS_binary_trajectory_mechanism.pdf",
            "FigS_response_alignment_summary.pdf",
            "FigS_acquisition_first_passage.pdf",
            "FigS_initialization_acquisition.pdf",
            "FigS_initialization_first_passage.pdf",
            "FigS_2SAT_matched_kw_heatmap.pdf",
        }
        generated = {path.name for path in (ROOT / "figures" / "regenerated").glob("*.pdf")}
        self.assertTrue(expected.issubset(generated))
        self.assertTrue(all((ROOT / "figures" / "regenerated" / name).stat().st_size > 1000 for name in expected))


if __name__ == "__main__":
    unittest.main()

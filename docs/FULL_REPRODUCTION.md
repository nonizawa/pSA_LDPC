# Full-study reproduction entry points

Full runs are expensive and were not executed during release preparation.
Plot-only reproduction uses `python scripts/reproduce_figures.py` and does not
need these commands. The default public smoke launchers were tested separately.

## Representative benchmark and early matched studies

Use `--help` on each public launcher to inspect the preserved driver interface:

```bash
python scripts/reproduce_ldpc_benchmark.py --help
python scripts/reproduce_matched_controls.py --help
python scripts/reproduce_trajectory_analysis.py --help
python scripts/reproduce_acquisition_stability.py --help
python scripts/reproduce_cross_code_transfer.py --help
python scripts/reproduce_2sat_matched_kw.py --help
```

The representative search records under
`data/processed/representative_benchmark/search_records/` contain each size and
mode's actual candidates, refinement and response-coefficient sweep. The exact
search design and evaluation-stream limitation are retained in Supplemental
S2 under `docs/source_records/supplement.tex`. The source implementation is
`src/reference/run_paper_experiments.py` with `src/ldpc/auto_match_bp.py`.

## Additional completed validations

The following commands invoke the preserved full-study drivers. They write
new outputs under `outputs/`, leaving archived study records unchanged:

```bash
python src/studies/run_psa_specific_fixed_transfer.py --n-workers 10
python src/studies/run_initialization_robustness.py --n-workers 8
python src/studies/run_binary_state_self_feedback.py --sweep-workers 8 --highstat-workers 10 --trajectory-workers 8
```

These full paths were inspected and their plotting functions imported, but a
complete end-to-end rerun was intentionally not performed. Their archived
validation records document the original completed experiments. The
binary-state BER-favorable sensitivity and selection-fairness CSVs are included
as completed records; do not replace them with the score-selected comparator.

The full drivers preserve original analysis/report templates and some internal
source names. The public manuscript map defines which outputs are current.
Any future rerun should use a separate output directory and should compare its
instance/config identities against the archived validation before interpreting
numerical differences.

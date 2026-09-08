# Data dictionary

All paths are repository-relative. CSV files use a header row and UTF-8 text;
JSON files preserve exact configuration or validation records; NPZ files store
NumPy arrays without executable objects.

## Instances and metadata

| Location | Unit / key | Main contents |
|---|---|---|
| `data/metadata/matrices/matrix_registry.csv` | One row per size/code identifier | $N$, $M$, degrees, generation seed, GF(2) rank, rate, file path, SHA-256 |
| `data/metadata/matrices/<size>/C00.npz` ... `C09.npz` | One matrix | Binary parity-check array `H` |
| `data/metadata/formulas/formula_registry.csv` | One row per formula | Formula ID, variable/clause count, $\alpha$, seed, SAT label, file path, SHA-256 |
| `data/metadata/formulas/F00.npz` ... `F29.npz` | One formula | Signed two-literal clause array `clauses` |
| `data/metadata/seeds/` | Study seed policy | Base/stride rules, formula seeds, bootstrap identifiers where retained |

## Representative benchmark and matched controls

| Location | Statistical unit | Main fields |
|---|---|---|
| `data/processed/figure_source_data/ldpc/*/final_comparison.csv` | Mode and SNR | BER, FER, syndrome/failure outcomes, BP reference |
| `data/processed/manuscript/representative_ldpc_summary.csv` | Size and mode | Pooled BER/FER and relative reduction |
| `data/processed/representative_benchmark/search_records/` | Candidate/refinement/sweep record | Selected parameters, per-SNR summaries, score components |
| `data/raw/matched_controls/*_by_seed_ebno.csv` | Seed batch and SNR | Matched arm count/BER/FER records |
| `data/processed/manuscript/matched_causal_control_summary.csv` | Size and rule | Pooled matched-control outcomes |
| `data/processed/manuscript/matched_causal_control_paired_statistics.csv` | Size, comparator, metric | Paired difference and interval |
| `data/processed/manuscript/matched_control_coefficient_sweeps.csv` | Size and coefficient | Fixed-parameter response-coefficient sweep |

## Trajectory, acquisition, and initialization records

| Location | Statistical unit | Main fields |
|---|---|---|
| `data/processed/supplement_figures/response_alignment/by_cycle_summary.csv` | Size, SNR, rule, cycle bin | Syndrome, instantaneous BER, flip/back-flip, lag/used-product, alignment summaries |
| `.../response_alignment/condition_summary.csv` | Condition | Endpoint/late-window means and intervals |
| `.../acquisition_stability/first_passage_curves.csv` | Size, rule, cycle | Censored cumulative reach |
| `.../acquisition_stability/trajectory_summary.csv` | Trajectory | Reach, first passage, residence, escape/return summaries |
| `data/processed/initialization_robustness/initialization_metadata.csv` | Paired trajectory | Initialization type, initial-state identity, channel/seed linkage |
| `.../condition_summary.csv` | Size, initialization, rule | Acquisition, decoded/late BER, residence/stability summaries |
| `.../first_passage_curves.csv` | Size, initialization, rule, cycle | Censored cumulative reach |

`NR` in publication figures means the transmitted codeword was not reached and
a conditional quantity is undefined. `N/A` means the diagnostic is not defined
for that update rule. Neither is encoded as a numerical zero.

## Cross-code transfer

| Location | Estimand | Main fields |
|---|---|---|
| `data/processed/cross_code_fixed_transfer/code_snr_three_way_summary.csv` | Three complete/fixed arms | Size, code, SNR, BER/FER for additive, pSA-specific, matched ablation |
| `.../code_level_pairwise_summary.csv` | Code-level package contrasts | BER/FER differences and ratios |
| `.../three_way_comparison_summary.csv` | Size/comparator | Median relative reductions and bootstrap intervals |
| `data/processed/matched_cross_code_ablation/code_level_summary.csv` | Matched rule ablation | Code-level BER/FER and reductions |
| `.../size_level_summary.csv` | Matched rule ablation | Size-level medians and code-bootstrap intervals |
| `configs/cross_code_fixed_transfer/parameter_comparison_table.csv` | Size and package | Schedule, activation, channel scaling, cycles, readout, memory coefficient |

## Binary-state self-feedback

| File | Purpose |
|---|---|
| `kappa_sweep.csv`, `kappa_sweep_by_seed.csv`, `kappa_sweep_by_ebno.csv` | Prespecified score selection across $\kappa$ |
| `best_kappa_selection.csv` | Score-selected values, including $N=192$, $\kappa=0.50$ |
| `ber_optimal_sensitivity_selection.csv` and related files | Separately retained $N=192$, $\kappa=0.25$ BER-favorable sensitivity |
| `equal_range_summary.csv` | Distinct $\kappa=\lambda$ amplitude-range estimand |
| `highstat_ber_fer.csv`, `paired_statistics.csv` | Publication-facing high-statistics values and paired intervals |
| `trajectory_*` | Targeted acquisition/freeze/stability diagnostics |

## Cross-problem controls

| Location | Contents |
|---|---|
| `data/processed/figure_source_data/maxcut/` | Selected MAX-CUT response-coefficient sweeps |
| `data/processed/manuscript/maxcut_summary.csv` | Publication summary of co-optimized outcomes |
| `data/raw/twosat/raw_trial_results.csv` | Matched-$k_w$ trial-level outcomes |
| `data/raw/twosat/raw_cell_formula_results.csv` | Formula/cell aggregates |
| `data/processed/supplement_figures/twosat_matched_kw/` | Response surface, formula-paired statistics, figure inputs |
| `data/processed/supplement_figures/twosat_initial/` | Initial jointly optimized study retained for transparent reinterpretation |

## Validation fields

Study-specific `validation.json`, `invariants.json`, `final_audit.json`, and
`manifest.json` files are immutable records from the corresponding completed
analysis. `tests/test_release_package.py` independently checks zero-coefficient
limits, shuffled-response invariants, instance hashes/ranks, claim records,
path independence, and plot-only regeneration.

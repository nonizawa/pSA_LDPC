# Numerical claim map

This file maps publication-facing values to immutable source records. It is a
provenance index, not a recalculation script.

| Claim | Values | Estimand / statistical unit | Canonical source |
|---|---|---|---|
| Representative independently optimized BER reduction | $N=96$: 33.5%; $N=192$: 74.8%; $N=288$: 81.8% | One representative matrix per size; mode-specific optimization; pooled equally over three SNRs | `data/processed/manuscript/representative_ldpc_summary.csv`; per-SNR records under `data/processed/figure_source_data/ldpc/` |
| Additive vs pSA-specific fixed-transfer BER reduction | 35.46% [32.29, 38.83%]; 76.61% [74.81, 77.17%]; 81.10% [80.66, 81.84%] | Complete-package transfer; median over ten code realizations per size; code bootstrap | `data/processed/cross_code_fixed_transfer/three_way_comparison_summary.csv`; `bootstrap_statistics.csv` |
| Additive vs matched $\lambda=0$ rule-ablation BER reduction | 59.90% [58.37, 61.14%]; 97.46% [97.29, 97.56%]; 97.61% [97.56, 97.70%] | Matched response-rule ablation; median over ten code realizations per size; code bootstrap | `data/processed/matched_cross_code_ablation/size_level_summary.csv` |
| Binary-state publication comparator, $N=96$ | $\kappa=0.25$; binary-minus-additive BER 0.014883 [0.014209, 0.015558] | Same nonmemory parameters; paired seed batches | `data/processed/binary_state_feedback/paired_statistics.csv`; `highstat_ber_fer.csv` |
| Binary-state publication comparator, $N=192$ | BER-favorable sensitivity $\kappa=0.25$; difference 0.204155 [0.203056, 0.205253] | Publication sensitivity; distinct from score-selected $\kappa=0.50$ | `data/processed/binary_state_feedback/ber_optimal_sensitivity_paired_statistics.csv`; `ber_optimal_sensitivity_selection.csv` |
| Binary-state publication comparator, $N=288$ | $\kappa=0.75$; difference 0.011927 [0.011260, 0.012593] | Same nonmemory parameters; paired seed batches | `data/processed/binary_state_feedback/paired_statistics.csv`; `highstat_ber_fer.csv` |
| Alternative-initialization acquisition | Additive 96%, 96%, 96%, and 97% for $N=192$ random/channel-hard and $N=288$ random/channel-hard; all three controls 0% | 200 trajectories per method/initialization/size; ten paired seed batches | `data/processed/initialization_robustness/condition_summary.csv`; `paired_statistics.csv` |
| Default-start acquisition | Additive 95.5% ($N=192$), 95.0% ($N=288$); matched pSA, gain-only, and shuffled 0% | 200 trajectories per method/size | `data/processed/manuscript/acquisition_natural_initialization_summary.csv` |
| Matched-$k_w$ 2-SAT primary contrast | $+3.944\times10^{-5}$ unsatisfied-clause rate, 95% formula-bootstrap interval $[1.333,6.611]\times10^{-5}$ | Formula-paired treatment minus $\rho=0$ at $k_w=16$, $\rho=0.08$; 30 formulas | `data/processed/manuscript/matched_kw_2sat_statistics.csv`; `data/raw/twosat/raw_cell_formula_results.csv` |

The complete claim ledger used for manuscript production is copied without
semantic changes to `docs/provenance/UPDATED_NUMERICAL_CLAIM_LEDGER.md`.

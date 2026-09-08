# pSA-specific fixed-transfer validation

**Predefined interpretation: Case 1.** Additive fixed transfer clearly outperforms the pSA-specific fixed-transfer comparator.

This study transfers one independently optimized, representative-matrix pSA parameter set per block length to the same ten archived random-regular (3,6) LDPC matrices used in the existing cross-code study. No matrix generation, parameter optimization, per-code retuning, or mechanism simulation was performed.

## 1. pSA-specific parameter sets

| Size | Candidate | cycles | schedule | I0 min | I0 max | shape | initial plateau | p_hold | kw | kr | channel scaling | burn-in | window | readout |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---|
| N96_M48 | 603 | 25600 | exponential | 0.030036 | 8.9073 | 1.1499 | 0.2475 | 0.66907 | 6.0716 | 8.7012 | snr:1.408 | 13349 | 3503 | majority |
| N192_M96 | 469 | 19200 | linear | 0.32982 | 7.8681 | 0.050627 | 0.23369 | 0.78653 | 0.34302 | 0.32625 | llr:1.0318 | 12836 | 3856 | majority |
| N288_M144 | 842 | 25600 | linear | 0.28843 | 9.2386 | 4.3386 | 0.82697 | 0.84396 | 6.5749 | 6.6783 | llr:0.95668 | 10774 | 7843 | best_state |

All three rows are the archived `mode=pSA`, `stage=final` selections used in the independently optimized representative-code benchmark. Their complete dictionaries match the manuscript release configuration exactly; lambda, finite-response, and output-memory coefficients are zero.

## 2. Additive versus pSA-specific fixed transfer

Across the 30 pooled code-level comparisons, additive wins/ties/loses in BER are **30/0/0**.

| Size | BER wins | Median relative BER reduction (95% code-bootstrap CI) | Mean reduction | Between-code SD | FER wins | Median relative FER reduction (95% CI) |
|---|---:|---:|---:|---:|---:|---:|
| N96_M48 | 10/10 | 35.455% [32.292, 38.830]% | 35.613% | 3.668 pp | 10/10 | 42.800% [38.842, 45.123]% |
| N192_M96 | 10/10 | 76.612% [74.810, 77.166]% | 76.303% | 1.621 pp | 10/10 | 81.813% [79.707, 82.845]% |
| N288_M144 | 10/10 | 81.105% [80.660, 81.839]% | 81.223% | 0.676 pp | 10/10 | 82.557% [82.082, 82.996]% |

The independent code realization is the only bootstrap unit. Individual bits and trials are not resampled as independent instances.

## 3. Effect relative to the matched lambda=0 ablation

| Size | Additive vs matched-ablation median BER reduction | Additive vs pSA-specific median BER reduction | Shrinkage | Retained fraction |
|---|---:|---:|---:|---:|
| N96_M48 | 59.904% | 35.455% | 24.449 pp | 59.19% |
| N192_M96 | 97.457% | 76.612% | 20.845 pp | 78.61% |
| N288_M144 | 97.614% | 81.105% | 16.510 pp | 83.09% |

The previously reported approximately 97% reductions at N=192 and 288 remain matched-ablation effects and are not relabeled as independently optimized pSA comparisons.

## 4. Protocol and validation

- Matrices: the archived C00--C09 files were loaded directly; all 30 SHA-256 hashes, generation seeds, degree checks, GF(2) ranks, and rates match the existing transfer registry.
- Seeds: the exact ten existing seed batches were reused: `[20260826, 20268848, 20276870, 20284892, 20292914, 20300936, 20308958, 20316980, 20325002, 20333024]`.
- Trials: 1000 per code, SNR, and arm at 2.0, 2.5, and 3.0 dB.
- Kernel: the existing `candidate_chunk` path and cross-code condition runner were reused unchanged.
- pSA-specific memory state: `implementation_mode=pSA`, lambda=0, response coefficient=0, and output-memory coefficient=0.
- Initialization, channel generation, stopping behavior, and readout are inherited unchanged from the existing cross-code runner; only the size-specific pSA parameter dictionary differs.
- Per-code retuning: none.
- Validation passed: **True**.
- New-arm runtime: 33.2 min.

## 5. Manuscript recommendations

- Figure 5: update to a three-way fixed-transfer comparison; retain the matched-ablation ratios as explicitly rule-level effects.
- Cross-code claim: it can be strengthened to state that additive fixed transfer remains superior to a separately optimized pSA fixed-transfer package within the tested code ensemble.
- Discussion limitation: delete the hypothetical sentence about a future separately optimized memoryless transfer and replace it with the observed three-way result and its tested scope.
- Additional trials: not indicated by the code-level result; the size-level median CIs are directionally resolved.
- Validation status: complete.
- Initialization-robustness study: may proceed; it is a separate question and does not alter this fixed-transfer result.

## 6. Output inventory

- `raw_psa_specific_seed_batches.csv`: new pSA-specific seed-batch counts and rates.
- `raw_three_way_seed_batches.csv`: all three fixed-transfer arms, including reused existing results.
- `code_snr_three_way_summary.csv`: absolute three-way BER/FER by code and SNR plus pooled rows.
- `code_level_pairwise_summary.csv`: all three pairwise comparisons.
- `pooled_code_level_summary.csv`: pooled code-level pairwise outcomes.
- `three_way_comparison_summary.csv`: size-level effects and code-bootstrap intervals.
- `overall_30_code_summary.csv`: descriptive 30-code pooled win counts and effects.
- `parameter_comparison_table.csv`, `matrix_seed_metadata.csv`, `seed_metadata.json`, and `validation.json`: provenance and checks.
- `figures/Fig_A_three_way_code_BER.*`, `Fig_B_size_relative_BER_reduction.*`, and `Fig_C_additive_vs_pSA_specific_log_BER_ratio.*`: manuscript candidates.

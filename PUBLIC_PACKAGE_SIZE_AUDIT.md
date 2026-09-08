# Public package size audit

Payload measurement excludes this size report and the two generated integrity manifests.

- Payload: 90,587,554 bytes (86.391 MiB)
- Payload files: 799

| Area | Bytes | MiB |
|---|---:|---:|
| src | 921,003 | 0.878 |
| configs | 132,712 | 0.127 |
| data/processed | 68,979,894 | 65.784 |
| data/raw | 17,084,064 | 16.293 |
| figures | 3,018,717 | 2.879 |
| docs | 178,040 | 0.170 |

## Largest 20 files

| File | Bytes |
|---|---:|
| `data/processed/supplement_figures/response_alignment/by_cycle_summary.csv` | 17,731,986 |
| `data/raw/twosat/raw_trial_results.csv` | 16,238,125 |
| `data/processed/supplement_figures/acquisition_stability/by_cycle_summary.csv` | 5,488,092 |
| `data/processed/binary_state_feedback/trajectory_by_cycle_summary.csv` | 3,924,818 |
| `data/processed/initialization_robustness/trajectory_summary.csv` | 3,861,514 |
| `data/processed/supplement_figures/response_alignment/trajectory_summary.csv` | 3,620,867 |
| `data/processed/initialization_robustness/alternative_trajectory_summary.csv` | 2,657,340 |
| `data/processed/representative_benchmark/search_records/N288_M144/lambda_pSA/deep_search_by_ebno.csv` | 1,228,373 |
| `data/processed/representative_benchmark/search_records/N192_M96/lambda_pSA/deep_search_by_ebno.csv` | 1,212,246 |
| `data/processed/representative_benchmark/search_records/N288_M144/tau_pSA/deep_search_by_ebno.csv` | 1,211,866 |
| `data/processed/representative_benchmark/search_records/N192_M96/tau_pSA/deep_search_by_ebno.csv` | 1,208,029 |
| `data/processed/representative_benchmark/search_records/N96_M48/lambda_pSA/deep_search_by_ebno.csv` | 1,188,610 |
| `data/processed/representative_benchmark/search_records/N96_M48/tau_pSA/deep_search_by_ebno.csv` | 1,173,823 |
| `data/processed/representative_benchmark/search_records/N192_M96/pSA/deep_search_by_ebno.csv` | 1,150,114 |
| `data/processed/representative_benchmark/search_records/N288_M144/pSA/deep_search_by_ebno.csv` | 1,148,810 |
| `data/processed/representative_benchmark/search_records/N96_M48/pSA/deep_search_by_ebno.csv` | 1,118,684 |
| `data/processed/supplement_figures/acquisition_stability/trajectory_summary.csv` | 1,053,829 |
| `data/processed/supplement_figures/response_alignment/by_seed_condition.csv` | 1,046,046 |
| `data/processed/cross_code_fixed_transfer/raw_three_way_seed_batches.csv` | 914,156 |
| `data/processed/supplement_figures/cross_code_transfer/mechanism_trajectory_summary.csv` | 665,138 |

## Files exceeding 10 MB (decimal)

Count: 2
- `data/processed/supplement_figures/response_alignment/by_cycle_summary.csv`: 17,731,986 bytes
- `data/raw/twosat/raw_trial_results.csv`: 16,238,125 bytes

## Files exceeding 50 MB (decimal)

Count: 0

## Files exceeding 100 MB (decimal)

Count: 0

## Archive policy

Large cycle-resolved source collections are excluded; see ZENODO_ARCHIVE_PLAN.md.
All included files fit the requested standard-repository size ceiling. No Git LFS setup was performed.

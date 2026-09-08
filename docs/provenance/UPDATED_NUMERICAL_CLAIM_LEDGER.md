# Numerical claim ledger

This ledger records each manuscript-facing numerical claim, its statistical unit, and its copied machine-readable source. Percentages are recomputed from stored rates. The independently optimized representative-code benchmark, matched lambda=0 controls, separately swept binary-state control, pSA-specific fixed transfer, and initialization-robustness validation remain separate protocols.

| Claim | Value used | Statistical unit / interval | Copied source |
|---|---:|---|---|
| Representative-code BER reduction, N96 | 33.5% | 10,000 trials/SNR/mode; one fixed matrix | `representative_ldpc_summary.csv` |
| Representative-code BER reduction, N192 | 74.8107% | same | `representative_ldpc_summary.csv` |
| Representative-code BER reduction, N288 | 81.8443% | same | `representative_ldpc_summary.csv` |
| Matched-control additive BER, N96/N192/N288 | 0.0154035 / 0.00736476 / 0.00759664 | ten paired seed batches | `matched_causal_control_summary.csv` |
| Binary best-performance BER, N96 | kappa=0.25: 0.0302868 | 10,000 trials/SNR; separately swept kappa | `phase7_results/.../highstat_ber_fer.csv` |
| Binary-minus-additive BER, N96 | 0.0148833 [0.0142088, 0.0155578] | 95% CI over ten paired seed-batch means | `phase7_results/.../paired_statistics.csv` |
| Binary BER-favorable sensitivity, N192 | kappa=0.25: 0.211519 | 10,000 trials/SNR; distinct from score-selected kappa=0.50 | `phase7_results/.../ber_optimal_sensitivity_by_ebno.csv` |
| Binary-minus-additive BER, N192 sensitivity | 0.204155 [0.203056, 0.205253] | 95% CI over ten paired seed-batch means | `phase7_results/.../paired_statistics.csv` |
| Binary best-performance BER, N288 | kappa=0.75: 0.0195231 | 10,000 trials/SNR; separately swept kappa | `phase7_results/.../highstat_ber_fer.csv` |
| Binary-minus-additive BER, N288 | 0.0119265 [0.0112603, 0.0125927] | 95% CI over ten paired seed-batch means | `phase7_results/.../paired_statistics.csv` |
| Equal-range binary BER, N96/N192/N288 | 0.499507 / 0.499814 / 0.347390 | kappa=lambda=0.95/0.95/0.90; separate amplitude-matched estimand | `phase7_results/.../highstat_ber_fer.csv` |
| Binary score-selection fairness | N96: 0.25; N192: 0.50; N288: 0.75 | score=BER=FER optimum at N96/N288 only | `phase7_results/.../kappa_selection_fairness_summary.csv` |
| Binary trajectory acquisition, N192/N288 | 0% / 93% | 200 trajectories/method/size at 2.5 dB; score-selected kappa values | `phase7_results/.../trajectory_condition_summary.csv` and report |
| Binary decoded BER, N192/N288 trajectory study | 0.49724 / 0.0105382 | same trajectory cells | same |
| Additive decoded BER, N192/N288 trajectory study | 0.0067969 / 0.0059549 | same trajectory cells | same |
| Shuffled-minus-additive BER at 2.5 dB | 0.0800 / 0.3763 / 0.3913 | 95% CI over ten paired seed-batch means | `response_alignment_condition_summary.csv` and paired statistics |
| Natural correct reach, N192/N288 | 95.5% / 95.0% | 200 trajectories/method/size | `acquisition_natural_initialization_summary.csv` |
| Natural post-hit residence, N192/N288 | 0.999986 / 0.999982 | reaching trajectories only | `acquisition_natural_initialization_summary.csv` |
| Correct-start additive residence, N192/N288 | 0.9176 / 0.9020 | separate 200-trajectory intervention | `post_acquisition_stability_summary.csv` |
| Alternative-init additive correct reach, N192 random / hard | 96.0% / 96.0% | 200 trajectories/method/initialization; paired streams | `initialization_condition_summary.csv` |
| Alternative-init additive correct reach, N288 random / hard | 96.0% / 97.0% | same | same |
| Alternative-init control correct reach | matched pSA, gain-only, shuffled: 0% in all four cells | same | same |
| Alternative-init acquisition difference, N192 random / hard | 96.0 pp [92.7, 99.3] / 96.0 pp [93.2, 98.8] | unbounded paired t interval over ten seed-batch differences | `initialization_paired_statistics.csv` |
| Alternative-init acquisition difference, N288 random / hard | 96.0 pp [91.6, 100.4] / 97.0 pp [94.0, 100.0] | same; interval not clipped | same |
| Alternative-init initial BER, random / hard | about 0.50 / 0.093 | trajectory summary | `initialization_summary.csv` |
| Alternative-init decoded BER, additive / matched pSA | 0.00345--0.00591 / 0.283--0.324 | 200 trajectories per cell | `initialization_condition_summary.csv` |
| Alternative-init additive post-hit residence | at least 0.999231 | reaching trajectories only | same |
| pSA-specific fixed-transfer BER/FER wins | 30/30 / 30/30 | independent code realization | `cross_code_three_way_summary.csv` |
| pSA-specific fixed-transfer median BER reduction, N96 | 35.46%, CI [32.29, 38.83]% | code bootstrap, 20,000 resamples | `cross_code_three_way_summary.csv` |
| pSA-specific fixed-transfer median BER reduction, N192 | 76.61%, CI [74.81, 77.17]% | code bootstrap | same |
| pSA-specific fixed-transfer median BER reduction, N288 | 81.10%, CI [80.66, 81.84]% | code bootstrap | same |
| Matched-ablation BER/FER wins | 30/30 / 30/30 | independent code realization | `cross_code_level_summary.csv` |
| Matched-ablation median BER reduction, N96 | 59.90%, CI [58.37, 61.14]% | code bootstrap, 20,000 resamples | `cross_code_size_summary.csv` |
| Matched-ablation median BER reduction, N192 | 97.46%, CI [97.29, 97.56]% | code bootstrap | same |
| Matched-ablation median BER reduction, N288 | 97.61%, CI [97.56, 97.70]% | code bootstrap | same |
| Largest evaluated mean normalized-cut gain | about 0.184% | co-optimized MAX-CUT condition | `maxcut_summary.csv` |
| Matched-$k_w$ selected-unsatisfied-rate change | +3.944e-5, CI [+1.333e-5, +6.611e-5] | paired formula bootstrap | `matched_kw_2sat_statistics.csv` |
| Matched-$k_w$ primary relative change | 1.867% worsening | same | same |
| Matched-$k_w$ terminal relative change | 11.351% worsening | same | same |
| Matched-$k_w$ SAT success change | -1.46 percentage points, CI [-3.77, +0.46] | 13 satisfiable formulas | same |
| Matched-$k_w$ unsatisfiable-formula rate change | +4.804e-5, CI [+0.882e-5, +8.627e-5] | 17 unsatisfiable formulas | same |

The package-transfer, initialization, and binary-state values were rechecked against their archived reports and machine-readable CSV/JSON records on August 30, 2026. No new simulation or numerical recomputation was used for this rewrite. Rerun the audit before submission only if an analysis output or figure source data file is regenerated.

# Manuscript-to-repository reproducibility map

This map is synchronized to the manuscript revision identified in
`docs/manuscript_version.txt`. Paths are relative to the repository root.
`scripts/reproduce_figures.py` regenerates all data-derived figure files listed
below in `figures/regenerated/`; the exact synchronized artwork is retained in
`figures/reference_current/`.

## Main manuscript

| Item | Scientific claim or role | Source data | Configuration | Script or source | Expected output |
|---|---|---|---|---|---|
| Fig. 1 | Update rules, stochastic activation, and causal-control logic | Not simulation-derived | `configs/controls/` | `figures/reference_current/supplement/Fig1_update_rules.tex` | `figures/reference_current/main/Fig1_update_rules.pdf` |
| Fig. 2 | Independently optimized representative-code BER/FER | `data/processed/figure_source_data/ldpc/*/final_comparison.csv` | `configs/ldpc/representative_benchmark.json` | `scripts/figure_sources/make_consistency_figures.py` | `Fig2b_LDPC_BER_N192_N288.pdf`, `Fig2c_LDPC_FER_N192_N288.pdf` |
| Fig. 3 | Matched causal rule controls and coefficient sweeps | `data/processed/manuscript/matched_causal_control_summary.csv`; `matched_control_coefficient_sweeps.csv` | `configs/controls/matched_causal_controls.json` | `scripts/figure_sources/make_publication_figures.py` | `Fig3_matched_causal_controls.pdf` |
| Fig. 4 | Correct-basin acquisition and post-acquisition stability | `data/processed/supplement_figures/acquisition_stability/`; `data/processed/manuscript/acquisition_natural_initialization_summary.csv`; `post_acquisition_stability_summary.csv` | `configs/controls/acquisition_stability.json` | `scripts/plot_archived_studies.py`; `scripts/figure_sources/make_fig4_acquisition_retention.py` | `Fig4a_channel_acquisition.pdf`; `Fig4b_acquisition_retention_summary.pdf` |
| Fig. 5 | Three-way fixed transfer: additive package, pSA-specific package, and matched $\lambda=0$ ablation | `data/processed/cross_code_fixed_transfer/code_snr_three_way_summary.csv`; `code_level_pairwise_summary.csv` | `configs/cross_code_fixed_transfer/` | `scripts/plot_archived_studies.py` | `Fig5_cross_code_three_way_BER.pdf` |
| Table I | Distinct numerical designs and supported inference | `docs/provenance/PROTOCOL_PROVENANCE_TABLE.md` | All study configs | Descriptive table in `docs/source_records/main.tex` | Main Table I |
| Table II | Representative-code pooled BER and reduction | `data/processed/manuscript/representative_ldpc_summary.csv` | `configs/ldpc/representative_benchmark.json` | Descriptive table in `docs/source_records/main.tex` | Main Table II |
| Table III | Fixed additive transfer package fields | `data/processed/manuscript/cross_code_parameter_comparison.csv` | `configs/cross_code_fixed_transfer/parameter_comparison_table.csv` | Descriptive table in `docs/source_records/main.tex` | Main Table III |

## Supplemental figures

| Item | Scientific role | Source data | Configuration | Script | Expected output |
|---|---|---|---|---|---|
| Fig. S1 | Representative $N=96$ BER | `data/processed/figure_source_data/ldpc/N96_M48/final_comparison.csv` | `configs/ldpc/representative_benchmark.json` | `make_consistency_figures.py` | `Fig2a_LDPC_BER_N96.pdf` |
| Fig. S2 | Binary-state $\kappa$ selection sweep | `data/processed/binary_state_feedback/kappa_sweep.csv` | `configs/binary_state_feedback/` | `plot_archived_studies.py` | `FigS_binary_kappa_sweep.pdf` |
| Fig. S3 | Binary-state high-statistics comparison | `data/processed/binary_state_feedback/highstat_ber_fer.csv`; sensitivity records | `configs/binary_state_feedback/` | `plot_archived_studies.py` | `FigS_binary_highstat_comparison.pdf` |
| Fig. S4 | Binary-state trajectory mechanism | `data/processed/binary_state_feedback/trajectory_by_cycle_summary.csv` | `configs/binary_state_feedback/` | `plot_archived_studies.py` | `FigS_binary_trajectory_mechanism.pdf` |
| Fig. S5 | Response-alignment endpoint diagnostics, including true-zero/N/A semantics | `data/processed/manuscript/response_alignment_condition_summary.csv` | `configs/controls/trajectory_analysis.json` | `make_supplement_visual_cleanup_figures.py` | `FigS_response_alignment_summary.pdf` |
| Fig. S6 | Size-resolved response and trajectory dynamics | `data/processed/supplement_figures/response_alignment/by_cycle_summary.csv` | `configs/controls/trajectory_analysis.json` | `plot_archived_studies.py` | `FigS_response_dynamics_by_size.pdf` |
| Fig. S7 | Censored first-passage curves | `data/processed/supplement_figures/acquisition_stability/first_passage_curves.csv` | `configs/controls/acquisition_stability.json` | `plot_archived_studies.py` | `FigS_acquisition_first_passage.pdf` |
| Fig. S8 | Correct-start post-acquisition stability | `data/processed/manuscript/post_acquisition_stability_summary.csv` | `configs/controls/acquisition_stability.json` | `make_supplement_visual_cleanup_figures.py` | `FigS_post_acquisition_stability.pdf` |
| Fig. S9 | Initialization-dependent acquisition | `data/processed/manuscript/initialization_condition_summary.csv` | `configs/initialization_robustness/` | `make_supplement_visual_cleanup_figures.py` | `FigS_initialization_acquisition.pdf` |
| Fig. S10 | Initialization-dependent censored first passage | `data/processed/initialization_robustness/first_passage_curves.csv` | `configs/initialization_robustness/` | `plot_archived_studies.py` | `FigS_initialization_first_passage.pdf` |
| Fig. S11 | Matched-$\lambda=0$ cross-code size summary | `data/processed/matched_cross_code_ablation/code_level_summary.csv`; `size_level_summary.csv` | `configs/cross_code/parameters_and_seeds.json` | `plot_archived_studies.py` | `FigS_cross_code_size_summary.pdf` |
| Fig. S12 | Cross-code mechanism sampling | `data/processed/manuscript/cross_code_mechanism_summary.csv` | `configs/cross_code/parameters_and_seeds.json` | `make_publication_figures.py` | `FigS_cross_code_mechanism.pdf` |
| Fig. S13 | MAX-CUT selected quality | `data/processed/manuscript/maxcut_summary.csv` | `configs/maxcut/boundary_experiments.json` | `make_consistency_figures.py` | `FigS_MAXCUT_quality.pdf` |
| Fig. S14 | MAX-CUT response-coefficient sweeps | `data/processed/figure_source_data/maxcut/*/*/memory_sweep_summary.csv` | `configs/maxcut/boundary_experiments.json` | `make_consistency_figures.py` | `FigS_MAXCUT_memory_sweep.pdf` |
| Fig. S15 | Initial jointly optimized 2-SAT mode distribution | `data/processed/supplement_figures/twosat_initial/initial_joint_optimization_summary.csv` | `configs/twosat/initial_tau_parameters.csv` | `make_consistency_figures.py` | `FigS_2SAT_initial_mode_distribution.pdf` |
| Fig. S16 | Initial jointly optimized 2-SAT selected $\rho$ | `data/processed/figure_source_data/twosat/initial_joint_optimization_stats.csv` | `configs/twosat/initial_tau_parameters.csv` | `make_consistency_figures.py` | `FigS_2SAT_initial_rho_vs_alpha.pdf` |
| Fig. S17 | Matched-$k_w$ finite-response slices | `data/processed/manuscript/matched_kw_response_surface.csv`; `matched_kw_2sat_statistics.csv` | `configs/twosat/matched_kw_control.json` | `make_consistency_figures.py` | `FigS_2SAT_matched_kw_slices.pdf` |
| Fig. S18 | Matched-$k_w$ finite-response heat map | `data/processed/supplement_figures/twosat_matched_kw/kw_rho_aggregate_summary.csv`; `statistics_ci.csv` | `configs/twosat/matched_kw_control.json` | `plot_archived_studies.py` | `FigS_2SAT_matched_kw_heatmap.pdf` |

The script names in the table omit the common prefix
`scripts/figure_sources/` where applicable.

## Supplemental tables

| Item | Content | Canonical record(s) |
|---|---|---|
| Table S1 | Study-level reproducibility roles | `docs/provenance/PROTOCOL_PROVENANCE_TABLE.md`; study configs |
| Table S2 | Additive and pSA-specific transfer packages | `configs/cross_code_fixed_transfer/parameter_comparison_table.csv` |
| Table S3 | Representative-code per-SNR outcomes | `data/processed/figure_source_data/ldpc/*/final_comparison.csv` |
| Table S4 | Matched causal-control summary | `data/processed/manuscript/matched_causal_control_summary.csv` |
| Table S5 | Binary-state publication comparison | `data/processed/binary_state_feedback/highstat_ber_fer.csv`; `ber_optimal_sensitivity_summary.csv` |
| Table S6 | Response-alignment diagnostics | `data/processed/manuscript/response_alignment_condition_summary.csv` |
| Table S7 | Acquisition and post-hit residence | `data/processed/manuscript/acquisition_natural_initialization_summary.csv`; `post_acquisition_stability_summary.csv` |
| Table S8 | Initial-state diagnostics | `data/processed/initialization_robustness/initialization_summary.csv` |
| Table S9 | Alternative-initialization outcomes | `data/processed/initialization_robustness/condition_summary.csv` |
| Table S10 | Fixed-transfer effects against both comparators | `data/processed/cross_code_fixed_transfer/three_way_comparison_summary.csv` |
| Table S11 | Thirty matched-ablation code-level outcomes | `data/processed/matched_cross_code_ablation/code_level_summary.csv` |
| Table S12 | Selected MAX-CUT outcomes | `data/processed/manuscript/maxcut_summary.csv` |
| Table S13 | Matched-$k_w$ 2-SAT response surface | `data/processed/manuscript/matched_kw_response_surface.csv` |

Tables rendered directly in TeX are traceable to these records but are not
rewritten by the plot-only command. Their values are checked by the lightweight
validation suite and the synchronized numerical-claim ledger.

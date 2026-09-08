# Methods Clarification Audit

Date: 2026-08-31  
Source revision: `pra_manuscript_targeted_cleanup_20260831/`  
Audited revision: `pra_manuscript_methods_clarification_20260831/`

The descriptions below were recovered from the archived commands, result CSV files, and the implementation in `LDPC_pbit/ldpc_pbit.py`, `LDPC_pbit/auto_match_bp.py`, `LDPC_pbit/compare_lambda_memory.py`, and `run_paper_experiments.py`. No unrecorded method detail was inferred.

## BP reference decoder

1. **Exact BP algorithm.** Custom binary sum-product belief propagation. Check-to-variable messages use the standard tanh/atanh product update, and variable/posterior messages are LLR sums.

2. **Update schedule.** Flooding. All check-to-variable messages are updated before the variable-to-check and posterior updates of the iteration; it is not layered or sequential decoding.

3. **Maximum iterations.** 50 complete flooding iterations.

4. **Stopping criterion.** After each complete iteration, hard decisions are formed and decoding stops early when their syndrome is zero. Otherwise the result after iteration 50 is returned.

5. **LLR definition.** BPSK maps 0 to +1 and 1 to -1. With the actual generator dimension defining `R = dim(G)/N`, the simulation uses `sigma^2 = 1/[2 R 10^(Eb/N0/10)]` and `L_i = 2 y_i/sigma^2`. Input and edge LLRs are clipped to +/-50, and the atanh product is clipped to `[-1+1e-12, 1-1e-12]`.

6. **Same H/channel relation.** BP uses the same fixed representative H, generator-derived rate, BPSK/AWGN model, and SNR values as the corresponding p-bit benchmark. It uses separate message/noise Monte Carlo streams and therefore is not paired trial by trial to the p-bit points. The reference contains 5000 trials per SNR. Conditional on a received vector, BP has no random update.

7. **External library.** None. The decoder is a custom NumPy implementation. NumPy supplies numerical array operations only.

8. **Supplement location.** Sec. S2.B, “BP reference decoder,” immediately after the representative-code optimization protocol.

9. **BP recomputation.** No. Existing BP CSV files and implementation records were inspected; no BP curve or trial was rerun.

## Representative-code optimization

10. **Initial candidate count.** 1000 candidates for every block length and each of pSA, additive lambda-pSA, and finite-response tau-pSA.

11. **Trials and seeds.** Each initial candidate uses 160 trials per SNR and one candidate-specific mode-specific stream. The SNRs are 2.0, 2.5, and 3.0 dB. Refinement and final-confirmation seed usage is described below.

12. **Refinement stage.** The 32 lowest-score candidates are reevaluated with 2000 total trials per SNR divided across six new seed streams. “2000” is the total per SNR, not 2000 per seed.

13. **Top-k selection.** `k = 32`, ranked first by the composite score and then by its BER log-gap component.

14. **Selection objective.** For an equal average over the three SNRs,

    `S = mean(|log10(BER+eps)-log10(BP_BER+eps)|) + 0.2 mean(|log10(FER+eps)-log10(BP_FER+eps)|) + 0.5 mean(nonzero-syndrome fraction)`,

    with `eps = 1e-12`. Shape and floor weights are zero in this protocol.

15. **Tie break.** The implemented secondary key is the BER-only log-gap term. No additional scientific tie-break is specified for an exact residual tie in search/refinement. In the final coefficient sweep, records are ordered by ascending coefficient before the same two-key minimum, so an exact residual tie selects the lower coefficient.

16. **Final high-statistics protocol.** The lowest-score refined package is evaluated with 10,000 total trials per SNR divided across ten further seed streams. For pSA, that record is the reported endpoint. For additive and finite-response modes, all other fields are fixed and the response coefficient is then evaluated on a coarse 0.05 grid plus a focused 0.02 grid within +/-0.15 of the confirmed coefficient, at 10,000 trials per SNR and grid point. The score-minimizing grid point supplies the reported endpoint.

17. **Selection/evaluation stream relation.** Initial search, top-32 refinement, ten-stream confirmation, and response-coefficient sweep use distinct streams. The pSA endpoint is evaluated on streams not used to select its package during search/refinement. For additive and finite-response modes, however, the published endpoint is the selected 10,000-trial coefficient-grid point itself; there is no additional held-out evaluation after coefficient selection. The manuscript now says this explicitly and does not call the response-state endpoint fully held out.

18. **Independent optimization by mode.** Yes. Each mode uses a separate Optuna TPE study with mode-specific sampling and stochastic streams and selects its own complete package on the same representative H for that size.

19. **Supplement location.** Sec. S2.A, “Representative-code parameter optimization.” Main Sec. IV and the Fig. 2 caption point to this description.

20. **Unverified or guessed information.** None was inserted. Search budgets were checked against archived commands and the 1033-record per-mode outputs (1000 search + 32 refine + 1 confirmation). The response-grid and seed relationships were checked in both source and archived sweep CSV files.

## Stochastic-channel limitation

21. **Exact Discussion text.** “Because the LDPC implementation resamples stochastic channel evidence at every cycle, part of the observed benefit may also reflect temporal integration of channel information in addition to stabilization of the parity-constrained parallel dynamics. These contributions were not separately isolated; their separation under a deterministic channel field remains an open question.”

22. **Is channel integration limited to a possible contribution?** Yes. The text says “may also reflect” and does not identify channel integration as the explanation.

23. **Is separation from the parity-dynamics contribution unresolved?** Yes. The second sentence states explicitly that the contributions were not separately isolated.

24. **Deterministic-channel simulation.** Not performed.

25. **Cross-problem interpretation.** No causal connection was claimed between the weak MAX-CUT/2-SAT outcomes and stochastic-channel integration.

## Overall verification

26. **Main page count.** 13 pages, unchanged.

27. **Supplement page count.** 26 pages (previously 24); the two-page increase is due only to the added reproducibility text.

28. **Numerical-result changes.** None. Representative, control, binary-state, cross-code, initialization, MAX-CUT, and 2-SAT values are unchanged.

29. **Figure changes.** None. All files under `figures/` are byte-identical to the source revision.

30. **New simulation/optimization/aggregation.** None.

31. **Scientific-conclusion changes.** None. The added Discussion text narrows mechanism interpretation without changing the reported evidence or conclusions.

32. **Remaining major scientific concern from a PRApplied reviewer perspective.** No internal protocol inconsistency remains in these three targeted areas. A reviewer could still request (i) a deterministic-channel ablation to separate channel-evidence integration from parity-dynamics stabilization or (ii) a fully held-out confirmation after the final response-coefficient grid selection. These are now transparent scope limitations rather than undocumented design features. The benchmark is also intentionally limited to one representative matrix per size, with transfer evidence supplied separately; no broader ensemble universality is claimed.

## Compilation and visual QA

- `main.tex` and `supplement.tex` compile successfully with REVTeX 4.2.
- No undefined references/citations, duplicate labels, overfull boxes, stuck floats, or fatal errors were found.
- The existing nonfatal `nameref` compatibility warning and BibTeX `jnrlst` warning remain unchanged in character.
- All 13 Main pages and all 26 Supplemental pages were rendered to PNG and visually inspected. No clipping, missing glyph, broken equation, displaced float, or abnormal page layout was found.


# Binary-State Self-Feedback Control

## Executive conclusion

**Predefined interpretation: Case 1: binary-state self-feedback is clearly worse than additive.**

Within the fixed representative-LDPC matched framework, the binary-state control does not reproduce the additive saturated-response result.

This study is a stored-quantity control, not a PIMI reproduction. It retains the existing partial activation, channel resampling, local-field evaluation, uniform stochastic threshold, schedule, initialization, readout, representative H, SNR grid, and seed-batch framework.

## 1–3. Implemented rule, timing, and zero-coefficient validation

The exact active-bit update is

```text
q_i^(t) = tanh(I0^(t) F_i^(t))
s_i^(t) = 2 w_i^(t) - 1  in {-1,+1}
d_i^(t) = q_i^(t) + kappa s_i^(t)
w_i^(t+1) = 1[d_i^(t) + xi_i^(t) >= 0],  xi_i^(t) ~ Uniform[-1,1]
```

`w_i^(t)` is the common pre-update/current state copied at the start of the Jacobi cycle. The manuscript convention is used: stored binary one maps to spin +1. A held bit skips candidate evaluation and retains `w_i`; there is no separate binary-memory commit. The production endpoint kernel has no response-history array. The trajectory logger computes response diagnostics, but its variant-5 update uses only `w_prev` and never `q_prev`.

At kappa=0, decoded endpoints matched the existing optimized pSA kernel for all test seeds, and the validation logger produced trajectory-identical state histories and threshold discriminants. All validation checks passed; see `validation.json`.

## 4–7. Equal-range, selected kappa, and high-statistics results

At kappa=lambda, both additive and binary-state rules span the same deterministic range [-(1+lambda), +(1+lambda)]. The 2,000-trial equal-range results were:

| Size | kappa=lambda | Additive BER | Binary-state BER | Matched pSA BER |
|---|---:|---:|---:|---:|
| N96_M48 | 0.95 | 0.0157569 | 0.499507 | 0.0358142 |
| N192_M96 | 0.95 | 0.00748785 | 0.499814 | 0.280872 |
| N288_M144 | 0.90 | 0.00740162 | 0.34739 | 0.322364 |

The predeclared sweep used the existing score/aggregation objective. Selected coefficients and 10,000-trial/SNR results pooled equally over the three SNRs are:

| Size | Best kappa | Additive BER / FER | Binary BER / FER | Matched pSA BER / FER | Binary-additive BER [95% CI] |
|---|---:|---:|---:|---:|---:|
| N96_M48 | 0.25 | 0.0154035 / 0.184667 | 0.0302868 / 0.307567 | 0.0351976 / 0.3724 | 0.0148833 [0.0142088, 0.0155578] |
| N192_M96 | 0.50 | 0.00736476 / 0.0599667 | 0.499692 / 1 | 0.281805 / 0.999933 | 0.492328 [0.491688, 0.492968] |
| N288_M144 | 0.75 | 0.00759664 / 0.0723333 | 0.0195231 / 0.134033 | 0.322369 / 1 | 0.0119265 [0.0112603, 0.0125927] |

**Selection-objective sensitivity (N192_M96).** The primary existing score selects kappa=0.50 because the wrong all-zero valid codeword has zero syndrome. The BER-minimizing point on the same predeclared grid is kappa=0.25. Its independent 10,000-trial/SNR confirmation gives pooled BER=0.211519 and FER=0.947067; binary-minus-additive BER=0.204155 [0.203056, 0.205253]. This secondary sensitivity does not change the primary objective and shows that the stored-quantity conclusion is not an artifact of selecting the freezing point.

SNR-resolved BER/FER values are in `highstat_ber_fer.csv`; both contrast directions and paired seed-batch 95% intervals are in `paired_statistics.csv`. The BER-optimal sensitivity records, when applicable, are stored in the correspondingly named sensitivity CSV files.

## 8. Did binary-state feedback reproduce the additive benefit?

- N96_M48: binary/additive BER ratio = 1.966; fraction of the pSA-to-additive BER improvement reproduced = 0.248.
- N192_M96: binary/additive BER ratio = 67.849; fraction of the pSA-to-additive BER improvement reproduced = -0.794.
- N288_M144: binary/additive BER ratio = 2.570; fraction of the pSA-to-additive BER improvement reproduced = 0.962.

The size dependence matters: at N=288 the binary-state arm reproduces much of the pSA-to-additive improvement, although its residual BER remains significantly above the additive response-state arm. Binary inertia is therefore a material contributor in that case, not an irrelevant control; it is insufficient to reproduce the full benefit.

No. Even after coefficient selection over the full predeclared grid, binary-state feedback remains clearly above additive at every size.

## 9. Targeted trajectory mechanism

A targeted 200-trajectory/condition diagnostic was run at 2.5 dB for N=192 and 288 using additive, binary-state, and matched pSA. Key values are:

| Size | Method | Correct acquisition | Decoded BER | Late BER | Late syndrome/M | Late back-flip | Channel alignment | Post-hit residence |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| N192_M96 | additive | 0.9550 | 0.00679687 | 0.00655573 | 0.00330826 | 0.000262306 | 0.695183 | 0.999986 |
| N192_M96 | binary_state | 0.0000 | 0.49724 | 0.49724 | 0 | 0 | 0.00736552 | nan |
| N192_M96 | pSA | 0.0000 | 0.289193 | 0.327935 | 0.431796 | 0.144407 | 0.325945 | nan |
| N288_M144 | additive | 0.9500 | 0.00595486 | 0.00609441 | 0.00254845 | 0.00016524 | 0.726681 | 0.999982 |
| N288_M144 | binary_state | 0.9300 | 0.0105382 | 0.0109163 | 0.00609605 | 0.000282442 | 0.723596 | 1 |
| N288_M144 | pSA | 0.0000 | 0.324549 | 0.340717 | 0.452193 | 0.175142 | 0.315184 | nan |

## 10–15. Novelty and manuscript implications

10. **Stored quantity as the novelty center.** Yes, within the tested matched LDPC framework. The experiment supports a stored-quantity-specific temporal-reinforcement interpretation: same-bit temporal feedback and equal range alone are insufficient, while storing and reusing the real-valued saturated response is associated with the full benefit. This remains a scoped empirical claim, not a proof of uniqueness across all p-bit systems.

11. **Recommended PIMI wording.** Describe PIMI as prior binary-state inertia with a closely related post-tanh/pre-threshold reinjection point. State that the present study instead stores a bit-specific real-valued saturated response under stochastic partial activation and studies LDPC basin acquisition/stability. Call this arm a `binary-state self-feedback control` or `PIMI-inspired control`, never a PIMI reproduction; Gaussian noise, fully active synchronous updates, and the PIMI schedule were not reproduced.

12. **Main Fig. 3/control section.** Yes. Add binary-state feedback as one additional point/bar in the matched-control panel and state its best-kappa comparison in the text; it answers the nearest-prior-work question more directly than gain-only or normalized memory.

13. **Supplement only?** No. The full sweep, validation, and trajectory panels belong in the Supplement, but the central best-kappa result and its interpretation should appear in Main because PIMI is the closest conceptual comparator.

14. **Further simulation needed?** No additional large simulation is indicated by this study. The coefficient boundary was prespecified and evaluated, high-stat validation used 10,000 trials/SNR, the score-versus-BER selection sensitivity was checked where it differed, and the targeted mechanism diagnostic used paired trajectories. A new simulation would be justified only by a new reviewer question or a changed claim.

15. **Proceed to novelty/related-work rewrite?** Yes. Preserve the distinction between this matched stored-quantity control and PIMI as a complete algorithm, and do not claim temporal feedback, bit alignment, real-valued local state, or the reinjection point alone as novel.

## Files

- `kappa_sweep.csv`, `kappa_sweep_by_ebno.csv`, `kappa_sweep_by_seed.csv`
- `equal_range_summary.csv`, `best_kappa_selection.csv`
- `highstat_ber_fer.csv`, `highstat_by_seed.csv`, `paired_statistics.csv`
- `ber_optimal_sensitivity_*.csv` (secondary, only where the predeclared score and BER selections differ)
- `validation.json`, `parameter_config_record.json`
- `trajectory_raw.csv.gz`, trajectory summaries and paired statistics
- `figures/Fig_A_mean_BER_vs_kappa.*`, `Fig_B_highstat_matched_comparison.*`, and `Fig_C_targeted_trajectory_mechanism.*`

No manuscript source or manuscript PDF was modified by this work.

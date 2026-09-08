# Initialization Robustness Study

**Predefined interpretation: Case A.**

This matched-parameter mechanism study tests whether the additive correct-basin acquisition advantage depends on starting from the valid all-zero codeword. Existing all-zero trajectories are reused; only independent-random and channel-hard-decision initializations are newly simulated. No parameter optimization, pSA-specific transfer package, new parity-check matrix, SNR sweep, or manuscript edit was performed.

## 1. Initialization definitions

- **Random:** every stored binary bit is drawn independently from Bernoulli(1/2) using a dedicated `SeedSequence([seed, 0x494E4954])` stream. This stream is independent of the channel and p-bit streams, and the same state is supplied to all four methods in a paired trajectory.
- **Channel hard decision:** bit 1 iff the received BPSK sample satisfies `y_i < 0`; because every selected channel scaling is positive and tanh is monotone, this is exactly equivalent to `z_i < 0` in the simulated channel input.
- **Response state:** zero for every bit, method, size, and initialization.

## 2. Initial-state diagnostics

| Size | Initialization | Initial BER | Syndrome fraction | Valid | Initially correct | Channel alignment |
|---|---|---:|---:|---:|---:|---:|
| N192/M96 | All-zero | 0.4972 | 0 | 100.0% | 0.0% | 0.007366 |
| N192/M96 | Random | 0.5017 | 0.5007 | 0.0% | 0.0% | 0.0001516 |
| N192/M96 | Channel hard decision | 0.09237 | 0.3564 | 0.0% | 0.0% | 0.7691 |
| N288/M144 | All-zero | 0.4969 | 0 | 100.0% | 0.0% | 0.006547 |
| N288/M144 | Random | 0.4996 | 0.4971 | 0.0% | 0.0% | 0.001465 |
| N288/M144 | Channel hard decision | 0.09297 | 0.3522 | 0.0% | 0.0% | 0.81 |

## 3. Correct-codeword acquisition

| Size | Initialization | matched pSA | additive | gain-only | shuffled | Additive minus pSA (95% paired seed-batch CI) |
|---|---|---:|---:|---:|---:|---:|
| N192/M96 | All-zero | 0.0% | 95.5% | 0.0% | 0.0% | 95.5% [92.4%, 98.6%] |
| N192/M96 | Random | 0.0% | 96.0% | 0.0% | 0.0% | 96.0% [92.7%, 99.3%] |
| N192/M96 | Channel hard decision | 0.0% | 96.0% | 0.0% | 0.0% | 96.0% [93.2%, 98.8%] |
| N288/M144 | All-zero | 0.0% | 95.0% | 0.0% | 0.0% | 95.0% [90.9%, 99.1%] |
| N288/M144 | Random | 0.0% | 96.0% | 0.0% | 0.0% | 96.0% [91.6%, 100.4%] |
| N288/M144 | Channel hard decision | 0.0% | 97.0% | 0.0% | 0.0% | 97.0% [94.0%, 100.0%] |

The confidence intervals are paired t intervals over ten seed-batch differences and are not clipped to the natural [-100%, 100%] range; consequently, an upper endpoint can be slightly above 100% because of finite-sample uncertainty.

## 4. Decoded and late BER

| Size | Initialization | Method | Decoded BER | Late instantaneous BER |
|---|---|---|---:|---:|
| N192/M96 | All-zero | matched pSA | 0.2892 | 0.3279 |
| N192/M96 | All-zero | additive | 0.006797 | 0.006556 |
| N192/M96 | All-zero | gain-only | 0.4972 | 0.4972 |
| N192/M96 | All-zero | shuffled | 0.3869 | 0.3987 |
| N192/M96 | Random | matched pSA | 0.2831 | 0.3281 |
| N192/M96 | Random | additive | 0.005833 | 0.005681 |
| N192/M96 | Random | gain-only | 0.2678 | 0.3182 |
| N192/M96 | Random | shuffled | 0.3857 | 0.3987 |
| N192/M96 | Channel hard decision | matched pSA | 0.2836 | 0.328 |
| N192/M96 | Channel hard decision | additive | 0.005911 | 0.005819 |
| N192/M96 | Channel hard decision | gain-only | 0.2687 | 0.3181 |
| N192/M96 | Channel hard decision | shuffled | 0.3886 | 0.3987 |
| N288/M144 | All-zero | matched pSA | 0.3245 | 0.3407 |
| N288/M144 | All-zero | additive | 0.005955 | 0.006094 |
| N288/M144 | All-zero | gain-only | 0.3225 | 0.3407 |
| N288/M144 | All-zero | shuffled | 0.3961 | 0.4005 |
| N288/M144 | Random | matched pSA | 0.3237 | 0.3409 |
| N288/M144 | Random | additive | 0.004583 | 0.004995 |
| N288/M144 | Random | gain-only | 0.323 | 0.3406 |
| N288/M144 | Random | shuffled | 0.3941 | 0.4005 |
| N288/M144 | Channel hard decision | matched pSA | 0.3208 | 0.3407 |
| N288/M144 | Channel hard decision | additive | 0.003455 | 0.003796 |
| N288/M144 | Channel hard decision | gain-only | 0.3186 | 0.3406 |
| N288/M144 | Channel hard decision | shuffled | 0.3946 | 0.4006 |

Primary endpoint contrasts below are matched-pSA minus additive, so a positive value favors additive dynamics.

| Size | Initialization | Decoded-BER reduction (95% paired seed-batch CI) | Late-BER reduction (95% paired seed-batch CI) |
|---|---|---:|---:|
| N192/M96 | All-zero | 0.2824 [0.2715, 0.2933] | 0.3214 [0.3172, 0.3256] |
| N192/M96 | Random | 0.2773 [0.2708, 0.2837] | 0.3224 [0.3184, 0.3264] |
| N192/M96 | Channel hard decision | 0.2777 [0.2711, 0.2842] | 0.3222 [0.3185, 0.3258] |
| N288/M144 | All-zero | 0.3186 [0.3107, 0.3265] | 0.3346 [0.33, 0.3393] |
| N288/M144 | Random | 0.3191 [0.3146, 0.3236] | 0.3359 [0.331, 0.3407] |
| N288/M144 | Channel hard decision | 0.3174 [0.3116, 0.3231] | 0.3369 [0.334, 0.3398] |

## 5. First passage and post-acquisition stability

Unreached trajectories remain censored. Reach fractions are reported above, and the following first-passage summaries use reached trajectories only; no artificial terminal cycle is assigned.
Because matched pSA reaches the correct codeword in no trajectory, an additive-versus-pSA conditional FPT difference is undefined rather than assigned an artificial value; Figure C compares the censored cumulative reach curves.

| Size | Initialization | Method | Reached / 200 | Correct FPT median / mean | Pooled correct residence | Escape probability/cycle | Return probability | Mean return time |
|---|---|---|---:|---:|---:|---:|---:|---:|
| N192/M96 | All-zero | matched pSA | 0 | undefined / undefined | undefined | undefined | undefined | undefined |
| N192/M96 | All-zero | additive | 191 | 444 / 1335 | 1 | 5.275e-06 | 1 | 2.722 |
| N192/M96 | All-zero | gain-only | 0 | undefined / undefined | undefined | undefined | undefined | undefined |
| N192/M96 | All-zero | shuffled | 0 | undefined / undefined | undefined | undefined | undefined | undefined |
| N192/M96 | Random | matched pSA | 0 | undefined / undefined | undefined | undefined | undefined | undefined |
| N192/M96 | Random | additive | 192 | 436.5 / 1431 | 0.9992 | 5.867e-06 | 1 | 131.2 |
| N192/M96 | Random | gain-only | 0 | undefined / undefined | undefined | undefined | undefined | undefined |
| N192/M96 | Random | shuffled | 0 | undefined / undefined | undefined | undefined | undefined | undefined |
| N192/M96 | Channel hard decision | matched pSA | 0 | undefined / undefined | undefined | undefined | undefined | undefined |
| N192/M96 | Channel hard decision | additive | 192 | 500.5 / 1399 | 1 | 4.096e-06 | 1 | 3.5 |
| N192/M96 | Channel hard decision | gain-only | 0 | undefined / undefined | undefined | undefined | undefined | undefined |
| N192/M96 | Channel hard decision | shuffled | 0 | undefined / undefined | undefined | undefined | undefined | undefined |
| N288/M144 | All-zero | matched pSA | 0 | undefined / undefined | undefined | undefined | undefined | undefined |
| N288/M144 | All-zero | additive | 190 | 425 / 1261 | 1 | 3.46e-06 | 1 | 5.062 |
| N288/M144 | All-zero | gain-only | 0 | undefined / undefined | undefined | undefined | undefined | undefined |
| N288/M144 | All-zero | shuffled | 0 | undefined / undefined | undefined | undefined | undefined | undefined |
| N288/M144 | Random | matched pSA | 0 | undefined / undefined | undefined | undefined | undefined | undefined |
| N288/M144 | Random | additive | 192 | 548 / 1337 | 1 | 2.147e-06 | 1 | 1.9 |
| N288/M144 | Random | gain-only | 0 | undefined / undefined | undefined | undefined | undefined | undefined |
| N288/M144 | Random | shuffled | 0 | undefined / undefined | undefined | undefined | undefined | undefined |
| N288/M144 | Channel hard decision | matched pSA | 0 | undefined / undefined | undefined | undefined | undefined | undefined |
| N288/M144 | Channel hard decision | additive | 194 | 457 / 1678 | 1 | 2.801e-06 | 1 | 13.69 |
| N288/M144 | Channel hard decision | gain-only | 0 | undefined / undefined | undefined | undefined | undefined | undefined |
| N288/M144 | Channel hard decision | shuffled | 0 | undefined / undefined | undefined | undefined | undefined | undefined |

The N=192 gain-only endpoint depends on initialization: its late BER is 0.4972 from the valid all-zero start and 0.3182/0.3181 from random/channel-hard starts. Nevertheless, gain-only never acquires the transmitted codeword under any tested initialization, so the initialization dependence of its wrong-valid freeze does not restore the additive mechanism.
Shuffled memory also has zero acquisition in all six size-by-initialization cells (0% maximum), preserving the requirement for bit-specific temporal alignment.
For additive dynamics, pooled post-hit correct residence remains between 0.999231 and 0.999996. All observed additive escapes return to the correct codeword; the microscopic return-time decomposition changes most visibly for N=192 random initialization, but post-acquisition stability remains high.

## 6. Interpretation and required answers

1. **Random initialization definition:** independent equiprobable binary bits from a dedicated paired initialization stream; exact seed construction is recorded above and in `parameters_and_seeds.json`.
2. **Channel-hard definition:** BPSK hard decision, bit 1 iff `y_i < 0`, verified equivalent to the sign of the actual channel value used by the p-bit kernel.
3. **Initial diagnostics:** reported in Sec. 2 and in `initialization_summary.csv`/`initialization_metadata.csv`.
4. **N=192 acquisition:** reported for every initialization and method in Sec. 3.
5. **N=288 acquisition:** reported for every initialization and method in Sec. 3.
6. **Additive vs matched pSA effect and CI:** Sec. 3 uses ten paired seed-batch differences; bits are never treated as independent CI units.
7. **Decoded/late BER:** Sec. 4 and the complete CSV summaries report both endpoints.
8. **First passage:** reach and conditional FPT are separated; cumulative censored curves are Figure C.
9. **Gain-only:** the N=192 wrong-valid all-zero freeze is initialization dependent, but gain-only acquires the transmitted codeword in 0/1,200 tested trajectories across all sizes and initializations.
10. **Shuffled memory:** it acquires the transmitted codeword in 0/1,200 trajectories while preserving the existing derangement and delayed-response-distribution invariants.
11. **Post-acquisition stability:** additive pooled residence remains above 0.9992 in every tested cell; the rare-escape/return decomposition varies somewhat with initialization, especially for N=192 random starts, without changing the high-retention conclusion.
12. **All-zero artifact question:** the additive acquisition advantage persists for every tested alternative initialization and is not specific to the valid all-zero start. All four alternative size-by-initialization acquisition CIs exclude zero.
13. **Discussion limitation:** replace the current outside-scope sentence with the observed, tested-scope robustness statement.
14. **Main Fig. 4:** retain the current integrated mechanism figure; use Figure A as a compact supplementary robustness panel unless the editor/reviewer specifically requests a main-text panel.
15. **Supplement additions:** initialization definitions, initial-state diagnostics, acquisition table, paired effect table, Figure A, and Figure C; Figure B may be included as endpoint confirmation.
16. **Additional simulation:** not required for this targeted reviewer concern.
17. **Completion:** validation passed = **True**; predefined interpretation = **Case A**.
18. **Next step:** stop large-scale simulation and proceed to manuscript revision.

## 7. Protocol and validation

- New trajectories: 3200 (random and channel-hard only); reused all-zero trajectories: 1600.
- Summed wall time recorded by the 16 newly simulated conditions: 3.1 min; final aggregation/figure/report invocation: 0.2 min.
- Matrices: representative generation-seed-0 matrices; hashes agree with the archived C00 registry.
- Parameters: the fixed additive-optimized matched-control package is shared by all methods; matched pSA sets only lambda to zero. No pSA-specific fixed-transfer package is used.
- Stochastic pairing: channel, p-bit core seed, and shuffled mapping follow the existing seed construction. Random initial states use a separate stream and therefore do not perturb those sequences.
- Validation passed: **True**.

## 8. Output inventory

- `trajectory_summary.csv`: standardized 4,800-row data set spanning three initializations.
- `raw_alternative_trajectory_metrics.csv.gz`: cycle-binned raw metrics for newly simulated initializations.
- `initialization_metadata.csv` and `initialization_summary.csv`: identity hashes and initial diagnostics.
- `condition_summary.csv`, `seed_batch_summary.csv`, `paired_statistics.csv`, and `first_passage_curves.csv`: reported analysis.
- `figures/Fig_A_initialization_acquisition.*`, `Fig_B_initialization_BER.*`, and `Fig_C_initialization_first_passage.*`: manuscript candidates.
- `validation.json`, `matrix_metadata.csv`, `parameters_and_seeds.json`, and `manifest.json`: provenance and audit records.

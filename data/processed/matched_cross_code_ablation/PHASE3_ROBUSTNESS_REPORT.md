# Phase 3: Multiple independent LDPC code realizations robustness

**Outcome: Strong success.**

The performance gain and its acquisition–retention mechanism are robust across independently generated random-regular LDPC code realizations under size-specific transferable parameters.

## Fixed-parameter performance results

All primary results use the Phase 1/2 size-specific lambda-anchor parameters without per-code retuning. Results are pooled with equal trial counts over Eb/N0 = 2.0, 2.5, and 3.0 dB unless stated otherwise.

| Size | BER wins | Median relative BER reduction (95% code bootstrap CI) | Mean relative BER reduction | Between-code SD | FER wins |
|---|---:|---:|---:|---:|---:|
| N96_M48 | 10/10 | 0.599 [0.5837, 0.6114] | 0.5963 | 0.01759 | 10/10 |
| N192_M96 | 10/10 | 0.9746 [0.9729, 0.9756] | 0.9745 | 0.001786 | 10/10 |
| N288_M144 | 10/10 | 0.9761 [0.9756, 0.977] | 0.9762 | 0.0009177 | 10/10 |

Positive reduction means additive lambda-pSA is better. The bootstrap unit is the independent code realization (20,000 resamples). Seed-batch paired t intervals are in `paired_statistics.csv`.

All 240/240 BER/FER code-level paired intervals (three SNRs plus pooled) exclude zero in the additive-favoring direction. Therefore, no condition required adaptive trial extension beyond 1,000 trials/SNR/mode.

## Parameter transferability and retuning

The same co-designed parameter set was transferred to every H matrix of a given size. No per-code full retuning was performed. Limited retuning was not used in the primary analysis.

## Lightweight acquisition–retention replication

Mechanism direction was reproduced in all sampled realizations: additive had higher correct-codeword acquisition in 9/9, lower late correct-to-incorrect bit back-flip in 9/9, and post-hit residence at least 0.9 and no worse than pSA when pSA retention was defined in 9/9.

| Size/code | Mode | Correct acquisition | Correct residence | Escape probability/cycle | Return probability | Mean return cycles | Late back-flip rate |
|---|---|---:|---:|---:|---:|---:|---:|
| N192_M96/existing_00 | additive | 0.98 | 1 | 7.88e-06 | 1 | 1.286 | 0 |
| N192_M96/existing_00 | pSA | 0 | undefined | undefined | undefined | undefined | 0.1439 |
| N192_M96/new_01 | additive | 0.98 | 1 | 9.094e-06 | 1 | 1.25 | 0 |
| N192_M96/new_01 | pSA | 0 | undefined | undefined | undefined | undefined | 0.1441 |
| N192_M96/new_02 | additive | 0.96 | 1 | 4.813e-06 | 1 | 2.5 | 0.0001611 |
| N192_M96/new_02 | pSA | 0 | undefined | undefined | undefined | undefined | 0.1452 |
| N288_M144/existing_00 | additive | 0.92 | 1 | 2.749e-06 | 1 | 1.667 | 0.0002751 |
| N288_M144/existing_00 | pSA | 0 | undefined | undefined | undefined | undefined | 0.1755 |
| N288_M144/new_01 | additive | 0.92 | 1 | 8.839e-07 | 1 | 1 | 0.0002457 |
| N288_M144/new_01 | pSA | 0 | undefined | undefined | undefined | undefined | 0.1759 |
| N288_M144/new_02 | additive | 0.98 | 1 | 3.455e-06 | 1 | 1 | 6.222e-05 |
| N288_M144/new_02 | pSA | 0 | undefined | undefined | undefined | undefined | 0.1756 |
| N96_M48/existing_00 | additive | 0.8 | 0.9154 | 0.0005624 | 0.9955 | 69.09 | 0.001083 |
| N96_M48/existing_00 | pSA | 0.68 | 0.915 | 0.0002855 | 0.9936 | 205.1 | 0.005108 |
| N96_M48/new_01 | additive | 0.84 | 0.9608 | 0.0004882 | 1 | 83.61 | 0.0007427 |
| N96_M48/new_01 | pSA | 0.7 | 0.9001 | 0.0002335 | 0.9919 | 348.5 | 0.006892 |
| N96_M48/new_02 | additive | 0.92 | 0.9668 | 0.0007089 | 1 | 48.51 | 0.0001114 |
| N96_M48/new_02 | pSA | 0.76 | 0.9241 | 0.000289 | 0.9941 | 169.6 | 0.00441 |

For N192/N288, pSA never acquired the correct codeword, while additive acquisition was 0.92–0.98 and post-hit residence was 0.999988–0.999999; every observed additive escape returned within a mean 1.0–2.5 cycles. N96 shows a size-dependent retention regime: additive escape events were more frequent than pSA, but returns were much faster (48.5–83.6 versus 169.6–348.5 cycles on the two new codes), yielding equal-or-higher residence and substantially lower late back-flip. Thus the retention signature is robust, but its escape/return decomposition is not size-invariant.

## N dependence

The median relative BER reduction rises from 0.599 at N96 to 0.9746 at N192 and 0.9761 at N288, while the between-code SD falls from 0.0176 to 0.00179 and 0.000918. This supports a stronger and less code-sensitive benefit at N192/N288 within the tested design. It is an N-dependence/robustness comparison, not finite-size scaling: only three sizes and ten code realizations per size were tested.

## Required final answers

1. **Reproduction:** Yes. BER and FER improved for every independently generated code realization tested.
2. **Win counts:** N96 10/10, N192 10/10, N288 10/10 for both BER and FER.
3. **Magnitude/variability:** Median relative BER reductions are 0.599, 0.9746, and 0.9761; between-code SDs are 0.0176, 0.00179, and 0.000918 for N96, N192, and N288, respectively. Means are reported in the table above.
4. **Transferability:** The result holds with one fixed Phase 1/2 parameter set per size across all ten matrices.
5. **Retuning:** No limited or full per-code retuning was required or performed.
6. **Mechanism:** Higher acquisition and lower late back-flip reproduced in 9/9 and 9/9 sampled codes. Retention is near-perfect at N192/N288; N96 retains a favorable residence/return signature but not a uniformly lower escape rate.
7. **N dependence:** Benefit is markedly larger and code-to-code variability smaller at N192/N288 than N96; this is not claimed as finite-size scaling.
8. **Figures:** Use Figure A for per-code robustness and Figure B for size-level distributions in the main text; use the mechanism figure and detailed tables in the Supplement.
9. **Phase 3 status:** Complete.
10. **Phase 4:** Proceed to the 2-SAT kw-confound-removal study.

## PRApplied figures

- Main Figure A: `figures/Fig_A_code_realization_BER_reduction.pdf` — per-code log BER ratio at each SNR and pooled.
- Main Figure B: `figures/Fig_B_size_robustness_summary.pdf` — BER/FER improvement distributions, medians, and win counts by size.
- Supplement: `figures/Fig_S_mechanism_robustness.pdf`, `code_level_summary.csv`, and the per-SNR rows in the summary tables.

## Completion and next phase

Phase 3 completion criterion: **met**. Full 10-code design present: True; mechanism sampling sufficient: True; output validation passed: True.

Phase 4 (2-SAT kw confound removal): **proceed**.

## Validation and provenance

- Phase 3 version: `2026-08-27-v1`
- Config hash: `f96f4a3806b405334aec28e1f5816dc049eecee5f618b8e2b6684c6a61b06da4`
- Raw row/schema/pairing validation: `{"actual_row_count": 1800, "all_passed": true, "expected_row_count": 1800, "paired_modes_ok": true, "row_count_ok": true, "selected_matrix_count": 30, "unique_key_count": 1800, "unique_keys_ok": true, "value_ranges_ok": true}`
- Adaptive CI check: 0 conditions require additional trials; details are in `ci_diagnostics.csv`.
- Matrix metadata records degree checks, GF(2) rank, actual rate, duplicate rows/columns, generation seed, and SHA-256 hash.
- The legacy N96 matrix has one duplicate column pair; it is retained and explicitly flagged as an existing-realization exception. All newly generated matrices exclude duplicate rows and columns and have full row rank.

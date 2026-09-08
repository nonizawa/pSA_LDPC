# Kappa-Selection Fairness Audit

## Executive conclusion

Both publication-facing choices pass **CASE A**. In the predeclared 2,000-trial/SNR grid, N=96 kappa=0.25 and N=288 kappa=0.75 are each simultaneously the score-selected, pooled-BER-best, and pooled-FER-best binary-state settings. No more favorable observed grid point was withheld, so no targeted high-stat simulation was triggered.

This audit only reads completed results. It did not run a decoder, change a kappa grid, modify an existing result, or modify the manuscript.

## Existing sweep, including sampling uncertainty

Intervals are 95% Student-t intervals across the existing eight paired seed batches, with each seed batch pooled equally across 2.0, 2.5, and 3.0 dB. This is the same seed-batch uncertainty convention used by the completed matched-control study.

### N96/M48

| kappa | BER 2.0 | BER 2.5 | BER 3.0 | Pooled BER [95% CI] | FER 2.0 | FER 2.5 | FER 3.0 | Pooled FER | Score | Role |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0.00 | 0.0593281 | 0.0328646 | 0.01525 | 0.0358142 [0.0343089, 0.0373196] | 0.55 | 0.3675 | 0.2145 | 0.377333 | 0.625565 | - |
| 0.25 | 0.0513802 | 0.029724 | 0.0143802 | 0.0318281 [0.0303158, 0.0333405] | 0.4505 | 0.3035 | 0.1905 | 0.314833 | 0.512531 | score-selected, BER-best, FER-best |
| 0.50 | 0.498042 | 0.500854 | 0.499625 | 0.499507 [0.497645, 0.501369] | 1 | 1 | 1 | 1 | 1.7963 | - |
| 0.75 | 0.498042 | 0.500854 | 0.499625 | 0.499507 [0.497645, 0.501369] | 1 | 1 | 1 | 1 | 1.7963 | - |
| 0.90 | 0.498042 | 0.500854 | 0.499625 | 0.499507 [0.497645, 0.501369] | 1 | 1 | 1 | 1 | 1.7963 | - |
| 0.95 | 0.498042 | 0.500854 | 0.499625 | 0.499507 [0.497645, 0.501369] | 1 | 1 | 1 | 1 | 1.7963 | equal-range |
| 1.00 | 0.498042 | 0.500854 | 0.499625 | 0.499507 [0.497645, 0.501369] | 1 | 1 | 1 | 1 | 1.7963 | - |
| 1.10 | 0.498042 | 0.500854 | 0.499625 | 0.499507 [0.497645, 0.501369] | 1 | 1 | 1 | 1 | 1.7963 | - |
| 1.25 | 0.498042 | 0.500854 | 0.499625 | 0.499507 [0.497645, 0.501369] | 1 | 1 | 1 | 1 | 1.7963 | - |

The selected and BER-best coefficients are identical (kappa=0.25), so selected-minus-best is exactly 0 with paired 95% CI [0, 0]. The next-lowest observed BER is kappa=0.00, BER=0.0358142; the selected setting is better by 0.00398611 (11.1% relative to the runner-up), with selected-minus-runner paired CI [-0.00549387, -0.00247836].

**Decision: CASE A.** Score-selected kappa is exactly the observed pooled-ber and fer optimum. The publication-facing binary comparator remains kappa=0.25; targeted high-stat validation is not required.

### N288/M144

| kappa | BER 2.0 | BER 2.5 | BER 3.0 | Pooled BER [95% CI] | FER 2.0 | FER 2.5 | FER 3.0 | Pooled FER | Score | Role |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0.00 | 0.327516 | 0.322977 | 0.316599 | 0.322364 [0.321447, 0.323281] | 1 | 1 | 1 | 1 | 2.888 | - |
| 0.25 | 0.281851 | 0.272686 | 0.263094 | 0.272543 [0.271354, 0.273733] | 1 | 0.9995 | 0.9955 | 0.998333 | 2.81398 | - |
| 0.50 | 0.151644 | 0.0942639 | 0.0430347 | 0.0963142 [0.093911, 0.0987175] | 0.7735 | 0.5155 | 0.2435 | 0.510833 | 1.997 | - |
| 0.75 | 0.0451892 | 0.0116424 | 0.00124826 | 0.01936 [0.0182027, 0.0205173] | 0.309 | 0.085 | 0.009 | 0.134333 | 0.650312 | score-selected, BER-best, FER-best |
| 0.90 | 0.356505 | 0.344951 | 0.340714 | 0.34739 [0.341857, 0.352923] | 0.791 | 0.715 | 0.686 | 0.730667 | 2.41485 | equal-range |
| 0.95 | 0.500566 | 0.499896 | 0.499944 | 0.500135 [0.498768, 0.501503] | 1 | 1 | 1 | 1 | 2.57878 | - |
| 1.00 | 0.500566 | 0.499896 | 0.499944 | 0.500135 [0.498768, 0.501503] | 1 | 1 | 1 | 1 | 2.57878 | - |
| 1.10 | 0.500566 | 0.499896 | 0.499944 | 0.500135 [0.498768, 0.501503] | 1 | 1 | 1 | 1 | 2.57878 | - |
| 1.25 | 0.500566 | 0.499896 | 0.499944 | 0.500135 [0.498768, 0.501503] | 1 | 1 | 1 | 1 | 2.57878 | - |

The selected and BER-best coefficients are identical (kappa=0.75), so selected-minus-best is exactly 0 with paired 95% CI [0, 0]. The next-lowest observed BER is kappa=0.50, BER=0.0963142; the selected setting is better by 0.0769543 (79.9% relative to the runner-up), with selected-minus-runner paired CI [-0.0788063, -0.0751023].

**Decision: CASE A.** Score-selected kappa is exactly the observed pooled-ber and fer optimum. The publication-facing binary comparator remains kappa=0.75; targeted high-stat validation is not required.

## Existing high-stat comparison after the fairness decision

Because both choices are already BER-best on the sweep grid, the completed high-stat runs are the relevant favorable binary-state comparisons:

| Size | Publication kappa | Binary BER / FER | Additive BER / FER | Binary-additive BER [paired 95% CI] |
|---|---:|---:|---:|---:|
| N96_M48 | 0.25 | 0.0302868 / 0.307567 | 0.0154035 / 0.184667 | 0.0148833 [0.0142088, 0.0155578] |
| N288_M144 | 0.75 | 0.0195231 / 0.134033 | 0.00759664 / 0.0723333 | 0.0119265 [0.0112603, 0.0125927] |

Both intervals are strictly above zero. The additive advantage therefore remains statistically clear after giving the binary-state comparator its observed BER-best coefficient on the predeclared grid.

## Equal-range and best-performance controls answer different questions

For N=96, equal range uses kappa=0.95 and gives binary BER=0.499507; best-performance uses kappa=0.25. For N=288, equal range uses kappa=0.90 and gives binary BER=0.34739; best-performance uses kappa=0.75. Equal range tests whether binary-state feedback reproduces additive response memory at the same maximum deterministic range. Best performance tests whether coefficient tuning allows binary-state feedback to reproduce the additive result. These estimands must remain separate.

## Consistency with N=192

At N=192 the score-selected kappa=0.50 is not a reasonable BER comparator because a wrong-valid-codeword freeze lowers its syndrome contribution. The already-completed BER-optimal sensitivity at kappa=0.25 gives pooled BER=0.211519; binary-minus-additive BER=0.204155 [0.203056, 0.205253]. Using this sensitivity value for fairness while preserving the original score-selected result for provenance is consistent with the present audit. N=96 and N=288 need no such substitution because their score-selected settings are already BER-best.

## Required final answers

1. **N96_M48 score-selected kappa:** 0.25.

2. **N96_M48 observed BER-best kappa:** 0.25.

3. **N96_M48 selected pooled sweep BER:** 0.0318281.

4. **N96_M48 BER-best pooled sweep BER:** 0.0318281.

5. **Is the difference material?** No difference exists: selected and BER-best are the same grid point.

6. **Was targeted high-stat required?** No; this is CASE A.

7. **Publication comparator:** kappa=0.25.

8. **N288_M144 score-selected kappa:** 0.75.

9. **N288_M144 observed BER-best kappa:** 0.75.

10. **N288_M144 selected pooled sweep BER:** 0.01936.

11. **N288_M144 BER-best pooled sweep BER:** 0.01936.

12. **Is the difference material?** No difference exists: selected and BER-best are the same grid point.

13. **Was targeted high-stat required?** No; this is CASE A.

14. **Publication comparator:** kappa=0.75.

15. **Equal-range role:** It is a matched deterministic-range control, not a best-performance binary baseline.

16. **N=192 consistency:** Retain score-selected kappa=0.50 for provenance and use the already validated BER-optimal kappa=0.25 sensitivity when judging comparator fairness.

17. **All-size reproduction:** No. At the favorable publication coefficients N=96, N=192 sensitivity, and N=288 binary-state feedback remains worse than additive.

18. **Statistical clarity:** Yes. The paired 95% binary-minus-additive BER interval is strictly positive for N=96 and N=288 here, and for the N=192 BER-optimal sensitivity in the completed study.

19. **Stored-quantity interpretation:** Maintained, scoped to the tested matched LDPC framework. N=288 still shows that binary inertia contributes materially, so the claim must be insufficiency for the full benefit, not irrelevance of binary self-feedback.

20. **Further simulation:** None is indicated by this fairness audit.

21. **Proceed to rewrite:** Yes. The novelty/related-work rewrite can proceed while transparently separating score selection, BER-fair comparator selection, and equal range.

## Outputs

- `kappa_selection_fairness_all_kappa.csv`
- `kappa_selection_fairness_summary.csv`
- `kappa_selection_fairness_paired_differences.csv`
- `kappa_selection_fairness_validation.json`

No new simulation was run. Existing phase-7 results and manuscript files were not modified.

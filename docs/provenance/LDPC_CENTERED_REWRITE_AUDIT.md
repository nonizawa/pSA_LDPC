# LDPC-centered rewrite audit

## 1. How was the central story made LDPC-centered?

The evidence is now ordered as: parallel p-bit model → direct LDPC factor/channel formulation → independently optimized representative benchmark → causal/stored-quantity controls → acquisition and post-acquisition mechanism → initialization robustness → fixed-package transfer across independent codes. MAX-CUT and 2-SAT are no longer coequal applications; they appear only as a short negative-generalization paragraph in Discussion and as detailed controls in the Supplemental Material.

## 2. How is Volpe et al. positioned?

Volpe et al. is the bridge establishing that p-bit LDPC decoding already exists. The revised Introduction uses it to distinguish their deeply pipelined, logically sequential p-computer from the present Jacobi-type field evaluation with stochastic partial activation. No first-p-bit-LDPC claim remains.

## 3. Was Volpe’s logical update organization confirmed from the primary source?

Yes. The complete Version-of-Record-associated arXiv full text states that arithmetic reordering does not alter logical p-bit update order and that the architecture preserves strictly sequential-update semantics while overlapping evaluation and local-field refresh. The pipeline is therefore not described as a Jacobi all-spin state commit.

## 4. Does Volpe directly compare with BP?

No direct BP, sum-product, or min-sum comparison was found in the full article/supplement text. “Software reference” denotes the software implementation of the same PIM formulation used to validate hardware/quantized behavior. The present Fig. 2 BP curve remains a conventional decoding reference, and no BP-equivalent claim is attributed to Volpe et al.

## 5. Is the PIMI distinction stated as stored quantity?

Yes. The manuscript acknowledges the close reinjection structure: PIMI adds the same-bit binary spin after the nonlinear response and before stochastic decision. The principal distinction is that the present additive rule stores the real-valued saturated response. Reinjection location alone is not claimed as novel.

## 6. Is the SFA distinction clear?

Yes. SFA is described as forming a real-valued adaptive state from binary activity and returning negative input/energy-gradient feedback for local heating and escape. The present rule retains the saturated response and reuses it additively at the response path for LDPC acquisition/stability.

## 7. Is the TEC distinction clear?

Yes. TEC is cited as same-bit temporal interaction between successive binary configurations. This explicitly concedes that same-bit temporal feedback is not unique; the present stored quantity is real-valued saturated response.

## 8. Where was binary-state self-feedback added in Main?

The exact rule is added in Sec. II-D, Causal controls. Publication-facing endpoint results are summarized in Sec. V, and trajectory interpretation is added in Sec. VI-B. Full selection, equal-range, statistics, and trajectory evidence are in Supplemental Sec. S4.

## 9. How are N=192 score-selected and BER-favorable kappa handled?

The archived score selection `kappa=0.50` is retained and identified as a wrong-valid-codeword freeze. Main and the publication-facing table use the independently high-statistics-validated, BER-favorable `kappa=0.25` sensitivity as a conservative comparator. This substitution is explicitly unique to N=192 and does not overwrite the prespecified score selection.

## 10. Is the N=288 binary-inertia nuance retained?

Yes. Binary feedback at `kappa=0.75` reaches the correct word in 93% of 200 trajectories and has perfect sampled post-hit residence, showing a material contribution. Its decoded BER remains above additive, so the conclusion is insufficiency to reproduce the full saturated-response benefit, not uselessness.

## 11. Are equal-range and best-kappa controls separated?

Yes. Equal-range uses `kappa=lambda` and tests deterministic amplitude. Best-kappa uses a separate sweep and tests the strongest observed binary endpoint. They have separate paragraphs and are never pooled.

## 12. What happened to Fig. 3?

Fig. 3 was left unchanged. It contains matched lambda-based rule controls whose only intervention is the response rule. Adding the separately swept best-kappa binary arm would mix selection protocols in one bar plot and invalidate the existing “only the response rule changes” caption. The binary result is instead a compact numerical paragraph in Main and three focused Supplemental figures.

## 13. Was the independent Main MAX-CUT / 2-SAT section removed?

Yes. “Cross-problem boundaries” was deleted as an independent section.

## 14. Where are MAX-CUT / 2-SAT mappings now?

They are in Supplemental Sec. S9, “Cross-problem boundary controls,” with exact objective and local-field equations and storage conventions.

## 15. Was Table III removed?

Yes. The former problem-boundary table was removed.

## 16. How much negative cross-problem evidence remains in Main?

One concise Discussion paragraph states that all tested MAX-CUT mean changes are below 0.2%, some selected finite-response points are at `rho=0`, and matched-kw 2-SAT favors `rho=0` at every tested clause weight with a slightly detrimental prespecified contrast. All detailed numbers, mappings, surfaces, and figures remain in the Supplemental Material.

## 17. Added references

Seven references were added: Caccioli–Franz–Marsili memory Ising (2008); Xu et al. SFA (2025); Du et al. TEC (2026); Gibeault et al. programmable stochastic-MTJ coupling (2024); Tawada–Tanaka–Togawa Ising LDPC (2020); Dikopoulos et al. RXO-LDPC (2025); and Volpe et al. pipelined p-computer / LDPC (2026).

## 18. Removed references

The later targeted citation-hygiene cleanup removed three uncited, obsolete bibliography entries: Kirkpatrick et al. (simulated annealing), Aspvall et al. (2-SAT), and Goemans--Williamson (MAX-CUT). Their removal does not change any in-text citation or scientific positioning. The PIMI arXiv category remains corrected to `cs.ET`.

## 19. Final reference count

26 BibTeX entries: 25 cited external scientific references plus one cited Supplemental Material record. There are no unused entries.

## 20. Closest prior work in the multidimensional comparison

PIMI is closest because both place a same-bit self-term after the nonlinear response and before stochastic decision. The main distinction is stored binary spin versus stored real-valued saturated response. SFA is the closest precedent for a real-valued local adaptive state, but its state source, feedback sign, and reinjection side differ.

## 21. Does any “first” claim remain?

No novelty sentence uses “first.” Text searches find “first” only in technical terms such as first-passage time or first visit.

## 22. Main / Supplemental page count

Source revision: 12 / 20 pages. Revised version: 13 / 24 pages. The additional Main page supports LDPC-related-work positioning and the stored-quantity control; the four Supplemental pages contain the archived binary-control evidence and moved cross-problem mappings.

## 23. Numerical-claim mismatch

No mismatch was found. The representative optimized reductions (33.5/74.8/81.8%), pSA-package transfer reductions (35.46/76.61/81.10%), matched-lambda-zero reductions (59.90/97.46/97.61%), initialization acquisition (96–97% versus 0%), and binary excess-BER intervals remain tied to separate protocols in `UPDATED_NUMERICAL_CLAIM_LEDGER.md`.

## 24. Were new simulations run?

No. Existing Phase-7 PDFs were copied into the revision, and all values came from archived CSV/JSON/report outputs. Only TeX compilation, text extraction, and PDF rendering were performed.

## 25. Largest remaining novelty concern from a PRApplied reviewer’s perspective

The main risk is that the additive rule could still be viewed as a specific heuristic combination of enlarged deterministic range and one-step response history rather than a broadly transferable physical principle. The matched gain/normalization/shuffle/binary controls and mechanistic trajectories substantially narrow that concern, but the paper has no physical implementation and tests one random-regular LDPC ensemble and limited channel/initialization conditions. The manuscript now states this scope directly and does not claim hardware efficiency or universal superiority.

## Compile and layout status

- Main and Supplemental PDFs compile with no undefined references or citations and no overfull boxes.
- The only Main box notices are noncritical underfull bibliography/Appendix lines.
- REVTeX/hyperref emits the standard nonfatal `nameref` label-definition warning in both builds; no manuscript reference is unresolved. The earlier Supplemental stuck-float warning was removed by stabilizing the two opening table placements.
- All new citation keys resolve, equation/figure/table numbering is stable, and the final 13-page Main and 24-page Supplemental PDFs were rendered and visually checked. The copied binary figures, three-way transfer figure, and moved cross-problem figures are readable at publication scale without clipping.
- Human submission checks remain: corresponding-author email, AI model/version identifiers, final GitHub/archival URL, and a final Version-of-Record check for 2026 preprints/early-view metadata.

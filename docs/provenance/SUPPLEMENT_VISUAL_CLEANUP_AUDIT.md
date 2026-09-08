# Supplemental Visual Cleanup Audit

## Scope and source verification

The source revision was `pra_manuscript_crossproblem_fig4_cleanup_20260907/`, the latest complete revision at the start of this task.  Its `main.tex`, `supplement.tex`, PDFs, figure scripts, and copied machine-readable source summaries were inspected.  The revised `source_data/` directory remains byte-for-byte identical to that source revision.  The final table and plots are direct renderings of those records.

## Table S11

1. **Original readability problem.** The original longtable followed the internal row order `N=192`, `N=288`, `N=96`, split across pages 17--19, repeated a continuation heading, and was interleaved with Figs. S11 and S12.  The repeated `N` column and dense seven-column layout made code-to-code scanning difficult.
2. **Block placement.** The revised logical table has consecutive blocks (a) `N=96`, (b) `N=192`, and (c) `N=288`.  Blocks (a) and (b) occupy the first dedicated table page; block (c) occupies the immediately following page.
3. **`N` column.** Removed.  The block heading carries the block length, leaving six columns per block.
4. **Rows retained.** Yes.  All 30 rows (`C00`--`C09` at each of three block lengths) are present.  A parser comparison against the pooled rows of `source_data/cross_code_level_summary.csv` returned `30/30` exact formatted matches.
5. **Figure interruption.** Removed.  Figs. S11 and S12 are flushed before Table S11; no figure appears between its two pages, and no automatic “continued” fragment is used.
6. **Font size.** Improved to `\small` with `\arraystretch=1.05`; the table is no longer forced into a single crowded page or a succession of small fragments.
7. **Later numbering.** Preserved.  The auxiliary cross-reference record gives Table S11 for the code-level table, Table S12 for MAX-CUT, and Table S13 for the matched-`k_w` 2-SAT table.

## Figure S5

8. **Small-positive panels.** The additive late correct-to-incorrect rates were `5.15e-4`, `3.63e-4`, and `1.70e-4`, which were nearly indistinguishable from zero on the previous linear bars.  The shuffled used-product values (`1.36e-3`, `1.06e-4`, and `6.04e-5`) were also visually compressed relative to the additive and normalized values.
9. **Revised display.** Grouped bars were replaced by points with the existing intervals.  Decoded BER remains logarithmic.  Late syndrome, late correct-to-incorrect rate, and used product use symmetric-log coordinates with a linear neighborhood containing true zero.  Small nonzero values are labeled in scientific notation.
10. **Used-product N/A cells.** Yes.  The source implementation sets `q_used_source_product` to `NaN` unless `temporal_memory` is true; that flag is true only for additive, normalized, and shuffled variants.  Thus pSA and gain-only have no applicable used-product metric.
11. **N/A versus zero.** pSA and gain-only are labeled `N/A`, not plotted at zero.  Applicable exact zeros elsewhere are plotted with method markers and a `0` annotation.

## Figure S8

12. **Exact-zero count.** Eight across the 2 sizes × 4 methods × 4 displayed metrics: five residence values, one escape probability, one late state BER, and one late back-flip rate.
13. **Very-small-positive count.** Four under the declared display threshold `0 < value < 10^-3`: the two additive escape probabilities (`6.19e-5`, `4.66e-5`) and two additive late back-flip rates (`3.47e-4`, `1.44e-4`).
14. **Point plot.** Yes.  All four panels now use method-specific point markers; the two metrics with archived intervals retain those intervals.
15. **Log-axis zero handling.** Panels (b)--(d) use symmetric-log coordinates with a true linear region around zero.  Exact zeros remain at numerical zero and are labeled `0`; no positive floor or epsilon is used.

## Figure S9

16. **Zero acquisition display.** Replaced zero-height bars with point markers at `y=0` on an explicitly drawn zero baseline.
17. **Visibility.** Yes.  Eighteen zero-acquisition cells are visible through distinct method colors and marker shapes; the y-axis extends slightly below zero so the markers are not clipped.
18. **Wilson intervals.** Preserved.  The script reads the existing low/high endpoints from `initialization_condition_summary.csv`; no interval was recalculated.  Boundary-scale floating-point differences are clipped only when converted to a nonnegative drawing length, not in the stored endpoints.

## Whole-Supplement audit

19. **Other ambiguity found.** No additional publication-facing figure required modification.  Figs. S7 and S10 use censored cumulative-reach curves and explicitly state that conditional first-passage values are not imputed.  Fig. S12 already uses visible point markers for zero acquisition.  The MAX-CUT and 2-SAT figures do not present undefined quantities as numerical zero.
20. **Other figure changes.** None.  In particular, Fig. S13 and the cross-problem figures were not redesigned.
21. **NR versus N/A.** The concepts remain distinct.  `N/A` is used only when a metric does not apply to an update rule (Fig. S5 used product).  No revised Supplemental panel needs an `NR` label: correct-start metrics in Fig. S8 are defined for all methods, acquisition in Fig. S9 is defined even when zero, and first-passage plots retain censoring rather than a numerical replacement.  Main Fig. 4 continues to use `NR` for an undefined conditional first-passage quantity after nonreach.
22. **Arbitrary epsilon.** None is used.  Symmetric-log axes represent exact zero directly.
23. **Numerical values.** Unchanged.  The source-data directory is identical to the base revision, the 30 Table S11 rows match the source CSV exactly at the displayed precision, and figure generators read existing mean/interval columns directly.
24. **Simulation or reaggregation.** None.  Only figure rendering, TeX layout, compilation, and validation were performed.
25. **Main scientific content.** Unchanged.  `main.tex` and `main.pdf` are byte-identical to the base revision.
26. **Final page counts.** Main manuscript: 13 pages.  Supplemental Material: 28 pages (previously 26); the increase is the permitted readability cost of keeping Table S11 in ordered, uninterrupted blocks and allowing longer semantic captions.
27. **Reviewer-facing interpretability.** Yes.  In the affected figures, exact zero is a visible marker, undefined/nonreaching conditional quantities are not imputed, N/A is textually identified, and small positive values are separated from zero by symmetric-log placement and/or scientific-notation labels.

## Compilation and visual verification

- `main.tex` and `supplement.tex` compile successfully with REVTeX 4.2.
- No undefined reference, undefined citation, overfull box, oversized float, stuck float, figure clipping, or numbering error is present.
- The known nonfatal REVTeX/hyperref `nameref` warning remains.  The BibTeX style also emits its existing `jnrlst` dependency warning; neither was introduced by this cleanup.
- All 13 Main pages and all 28 Supplemental pages were rendered and visually inspected.  Focused inspection covered Figs. S5, S8, and S9; both Table S11 pages; and the surrounding Figs. S11/S12 placement.

## Human final check

At final submission zoom, the author should make one preference-level check: whether the symmetric-log tick density in Fig. S5(d) is visually optimal for the journal production size.  The numerical semantics and labels are already unambiguous, so this is not a scientific or reproducibility issue.

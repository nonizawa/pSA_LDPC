# Public release audit

## Executive assessment

This directory is a clean, path-independent release candidate synchronized to
the final manuscript and Supplemental Material. It is technically ready to be
uploaded to a new GitHub repository. **It should not be made public until the
author selects and adds software and data/documentation licenses, supplies a
public contact address, and chooses the repository identity.** No GitHub or
Zenodo record was created.

The completed tree is approximately 86.7 MiB including integrity manifests;
processed data occupy 65.8 MiB and compact raw records 16.3 MiB. Exact payload
sizes and the largest files are in `PUBLIC_PACKAGE_SIZE_AUDIT.md`.

## Required audit questions

### 1. Was an existing GitHub staging package found?

Yes. The sibling source-project directory `github_release_memory_pbit_20260828`
was created on 2026-08-28 and was approximately 51 MiB. It contained early code,
configs, instance metadata, compact raw/processed data, documentation, and
validation, but its manuscript map ended at Supplemental Fig. S13 and predated
the pSA-specific transfer, initialization-robustness, and binary-state studies.

### 2. Was it reused or rebuilt?

It was used as a structural and early-study base, copied into a new staging
directory, and then substantially resynchronized. The earlier package remains
unchanged. Stale manuscript mappings and the old two-arm cross-code figure were
not carried forward as current artifacts.

### 3. Which manuscript revision was synchronized?

`pra_manuscript_supplement_visual_cleanup_20260907`: 13 Main pages and 28
Supplemental pages. Content checks confirmed the LDPC-centered story,
binary-state control, BP and representative-search methods, stochastic-channel
limitation, Main Fig. 4 zero/NR semantics, the reorganized Table S11, and
Supplemental zero/N/A/small-positive visual cleanup. Exact hashes are in
`docs/manuscript_version.txt`.

### 4. Is the pSA-specific fixed-transfer validation included?

Yes. The complete publication-facing processed records, seed/config provenance,
raw seed-batch tables, code-level summaries, bootstrap results, validation,
figures, report, and source driver are included under
`data/processed/cross_code_fixed_transfer/`,
`configs/cross_code_fixed_transfer/`, and `src/studies/`.

### 5. Is initialization robustness included?

Yes for full manuscript reproduction: initialization metadata, condition and
trajectory summaries, censored first-passage curves, paired statistics,
validation, figures, report, configs, and driver are included. The redundant
28-MiB compressed event-level table and per-method raw shards are intentionally
reserved for a large-data archive because the included summaries reproduce all
reported figures and claims.

### 6. Is binary-state self-feedback included?

Yes. The $\kappa$ sweep, high-statistics records, paired statistics,
BER-favorable sensitivity, equal-range comparison, trajectory source/summary,
selection-fairness audit and validation, figures, exact parameter record, and
driver are included.

### 7. Are the two $N=192$ binary choices kept distinct?

Yes. Score-selected $\kappa=0.50$ is stored in `best_kappa_selection.csv` and
its fairness/trajectory records. Publication-facing BER-favorable sensitivity
$\kappa=0.25$ is separately stored in `ber_optimal_sensitivity_*`. Neither
overwrites the other; the equal-range $\kappa=\lambda$ estimand is also separate.

### 8. Are matched $\lambda=0$ and pSA-specific fixed transfer separated?

Yes. They have separate directories, configs, filenames, documentation, and
claim-map rows. The former is a rule-isolating matched ablation; the latter is a
complete-package transfer-performance comparator whose independently selected
readout and nonmemory fields remain intact.

### 9. Is the BP reference implementation included?

Yes: `src/ldpc/auto_match_bp.py`, `src/ldpc/ldpc_pbit.py`, the exact BP baseline
CSV files, and plot mapping. It is the custom NumPy binary sum-product decoder
with flooding, 50-iteration cap, zero-syndrome early termination, clipping, and
no external decoder library.

### 10. Are representative optimization configs/protocols included?

Yes. Exact selected packages are in `configs/ldpc/`, and candidate,
refinement, coefficient-sweep, final, and BP records for all three sizes and
modes are under `data/processed/representative_benchmark/search_records/`.
The staged Supplemental source and provenance audit describe the 1000-candidate,
top-32, 2000-trial refinement, score, tie-break, and 10,000-trial endpoint use.

### 11. Is there a Main Fig. 1--5 reproduction map?

Yes: `docs/REPRODUCIBILITY_MAP.md`. Fig. 1 is explicitly identified as a TikZ
schematic rather than simulation output; Figs. 2--5 map to data/config/scripts.

### 12. Is there a Supplemental Fig. S1--S18 map?

Yes. Every figure has a current source-data/config/script/output row.

### 13. Is table provenance mapped?

Yes. Main Tables I--III and Supplemental Tables S1--S13 are mapped. Table S11
maps to all 30 matched-ablation code rows, while the three-way fixed-transfer
effects map separately to Table S10 and Main Fig. 5.

### 14. Do any private absolute paths remain?

No. Public text/data were scanned for personal home and cloud-sync paths,
username, and host name. Historical path-valued provenance fields use the
neutral relative label `source_project/`. See `docs/SANITIZATION_AUDIT.md`.

### 15. Are credentials or private-network data present?

No. Credential-shaped strings, private keys, bearer tokens, private URLs,
hostnames, and IP-like values were scanned. Only documented false positives
were found; none is secret or network data.

### 16. What is the largest file?

`data/processed/supplement_figures/response_alignment/by_cycle_summary.csv`,
17,731,986 bytes (16.91 MiB).

### 17. Are there files over 100 MB?

No. There are no files over 50 MB or 100 MB. Two files exceed 10 MiB; both are
ordinary CSV records needed for reproducibility. See
`PUBLIC_PACKAGE_SIZE_AUDIT.md`.

### 18. Were scientifically important files excluded from GitHub?

Only large, redundant event-level trajectory/source collections were excluded.
Their manuscript-derived summaries, validation, parameters, seeds, and figure
inputs are included. Nothing was deleted from the research workspace.

### 19. Are exclusions recorded as Zenodo candidates?

Yes, with approximate size, scientific role, and reproduction requirement in
`ZENODO_ARCHIVE_PLAN.md`.

### 20. Was the environment derived from actual code/environment records?

Yes. Python 3.10.7 and the pinned numerical/plotting versions were read from the
recorded environment and actual imports. NetworkX 2.8.6 is documented as an
optional structural-MAX-CUT dependency available in the source host's general
environment, rather than falsely claimed as present in the paper venv.

### 21. Did lightweight tests pass?

Yes. Twelve automated package tests passed, all Python files compiled, and all
seven smoke entry points completed from the staging directory. See
`docs/VALIDATION_SUMMARY.md`.

### 22. Is plot-only reproduction possible?

Yes. `python scripts/reproduce_figures.py` regenerates all 24 data-derived Main
and Supplemental figure PDFs from staged data without simulation. The exact
current reference artwork is also included. Main Fig. 1 remains a TikZ source.

### 23. Was any new scientific simulation run?

No manuscript-scale simulation, optimization, or statistical reanalysis was
run. Only explicitly allowed smoke-sized implementation checks (two trials, or
one trial per 2-SAT cell) and plot-only reproduction were executed. Their
temporary outputs were removed.

### 24. Were scientific outputs changed?

No. Archived numerical CSV/JSON records, claims, protocols, statistical units,
and figure data were copied without scientific modification. Changes were
limited to packaging, relative paths, public documentation, launcher path
resolution, and regeneration of plots from existing processed data.
The final manuscript source-data CSVs were compared cell by cell: only nine
`source_parameter_file` provenance strings in the parameter comparison table
were sanitized. Every numerical field and row count is unchanged.

### 25. Is the license finalized?

No source-project license was found. `LICENSE_RECOMMENDATION.md` records options,
but the author must select the software and data/documentation licenses and add
the full legal text before public release.

### 26. Was `CITATION.cff` created?

Yes. It names Naoya Onizawa, the affiliation, repository title, and preferred
manuscript citation. DOI, repository URL, volume/pages, and publication date are
omitted because they do not yet exist.

### 27. Can this directory be uploaded directly to GitHub?

Technically yes: imports, plots, smoke commands, links, paths, sizes, metadata,
and checksums have been audited. It can be uploaded as a private release
candidate immediately. Public activation is conditional on license and public
contact decisions; repository name/account and release tag also remain human
choices.

### 28. What must be reflected in the manuscript after release?

The verified public GitHub URL, immutable release/tag, Zenodo DOI, and any
separate full-trajectory archive DOI. `DATA_AVAILABILITY_UPDATE_GUIDE.md`
identifies the replacement fields without inventing identifiers.

### 29. What must be reflected after Zenodo deposition?

Insert the verified DOI for the immutable release and, if applicable, the
separate trajectory archive DOI. No DOI has been minted by this work.

### 30. What remains for human review?

The author decisions below, ownership/license compatibility, public contact
metadata, and the final release artifact list require review.

## Configuration correction during packaging

The old staging copy contained a mislabeled N=96 additive sweep/config: its
fields came from the finite-response mode. The research archive's original
CSV records were correct. The public selected-config JSON was reconstructed
from mode-specific final-comparison rows, and staged source CSVs were replaced
with byte-identical copies of the correct research archive files. This corrects
release metadata only; no scientific result was recomputed. A regression test
checks each mode's parameter fields and staged sweep against the archive.

## Human decisions remaining

- GitHub account or organization and repository name;
- public corresponding-author email;
- software license and data/documentation license;
- final release version/tag (candidate: `v1.0.0`);
- whether to archive the optional full trajectories/sweeps and whether to use a
  single or separate Zenodo record.

# Release manifest

- Package label: `github_public_release_20260907`
- Creation/synchronization date: 2026-09-07
- Manuscript revision label: `pra_manuscript_supplement_visual_cleanup_20260907`
- Package state: local release candidate; not pushed or published
- Per-file machine-readable manifest: `release_manifest.json`
- Integrity list: `CHECKSUMS.sha256`

## Scientific source groups

| Package path | Role |
|---|---|
| `src/`, `scripts/` | Validated kernels, BP implementation, study drivers, smoke launchers, and plotting tools |
| `configs/` | Exact selected parameter packages, seed policies, and protocol settings |
| `data/metadata/` | Matrix/formula identities, arrays, ranks, degrees, seeds, and hashes |
| `data/processed/representative_benchmark/` | Optimization, refinement, coefficient-sweep, and final records |
| `data/processed/matched_causal_controls/` | Fixed-parameter rule controls |
| `data/processed/cross_code_fixed_transfer/` | pSA-specific complete-package transfer comparison |
| `data/processed/matched_cross_code_ablation/` | Matched $\lambda=0$ response-rule ablation |
| `data/processed/initialization_robustness/` | Alternative-start acquisition and stability validation |
| `data/processed/binary_state_feedback/` | Score-selected, BER-sensitivity, equal-range, high-statistics, and trajectory records |
| `data/processed/manuscript/` | Current compact figure/table source layer |
| `data/raw/` | GitHub-suitable compact raw/count records |
| `figures/reference_current/` | Exact artwork synchronized to the manuscript revision |
| `docs/source_records/` | Synchronized TeX/Bib records for artifact mapping; no manuscript PDF |

The manifest does not contain a private source path. Original source labels and
their scientific roles are documented in `docs/SOURCE_PROVENANCE.md`.

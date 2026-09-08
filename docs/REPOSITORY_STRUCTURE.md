# Repository structure

```text
README.md                         Entry point and three reproduction levels
CITATION.cff                      Citation metadata without invented DOI/URL
configs/                          Exact selected packages and study settings
data/metadata/                    Matrices, formulas, seeds, hashes, and ranks
data/processed/                   Current plot/table inputs and study outputs
data/raw/                         Compact raw/count records suitable for GitHub
docs/                             Study, data, claim, and release documentation
figures/reference_current/        Artwork used by the synchronized manuscript
figures/regenerated/              Outputs of plot-only reproduction
scripts/                          Public launchers, plotters, and audit utilities
src/core/                         Response-rule helpers
src/ldpc/                         Validated LDPC/p-bit and BP implementation
src/controls/                     Control helpers
src/maxcut/, src/twosat/          Boundary-control kernels and instance tools
src/studies/                      Current fixed-transfer, initialization, and binary drivers
src/reference/                    Validated historical driver module boundaries
tests/                            Lightweight invariants and package checks
```

`src/reference/` deliberately preserves some source-level names used by the
validated research drivers. Public documentation and entry points use the
scientific study names. Generated smoke outputs belong in `outputs/` and are not
part of the release.

The package contains no manuscript PDF. Exact synchronized TeX/Bib source
records are retained under `docs/source_records/` for artifact mapping only;
the source revision and PDF hashes are recorded in `docs/manuscript_version.txt`.

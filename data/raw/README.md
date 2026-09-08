# Raw-data inclusion policy

This directory contains compact records sufficient to recompute the principal paired and cross-instance statistics. It intentionally excludes the complete cycle-resolved trajectory collections.

Measured source collections before staging:

| Source category | Size |
|---|---:|
| Response-alignment study directory | 550 MiB |
| Natural-initialization acquisition directory | 152 MiB |
| Correct-start stability directory | 30 MiB |
| Cross-code transfer directory | 4.1 MiB |
| Matched-$k_w$ 2-SAT directory | 77 MiB |
| Initial 2-SAT jointly optimized results | 79 MiB |
| MAX-CUT structural results | 61 MiB |

The excluded response-alignment and acquisition collections include 163 MiB and 51 MiB seed-by-cycle CSVs plus compressed per-condition trajectory streams. These should be placed in an external immutable archive. The GitHub package retains the cycle-bin, seed-condition, trajectory-summary, first-passage, and manuscript-facing files needed for figure generation and claim checking.

No raw output was regenerated while preparing this package.

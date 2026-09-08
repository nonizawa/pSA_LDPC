# Validation summary

Validation was run from the staging directory without importing the research
workspace. No manuscript-scale simulation or optimization was launched.

## Automated package tests

`python scripts/run_validation.py` completed 12 tests successfully:

1. additive $\lambda=0$ trajectory equivalence to pSA;
2. finite-response $\rho=0$ response-rule equivalence;
3. binary-state $\kappa=0$ equivalence;
4. normalized-memory and gain-only formula checks;
5. shuffled derangement, no-self-link, and moment preservation;
6. all 30 LDPC matrix hashes, GF(2) ranks, and degree constraints;
7. all 30 random 2-SAT identities and 13/17 SAT/UNSAT labels;
8. processed-data schemas and 30/30 matched-ablation code wins;
9. current pSA-specific fixed-transfer, initialization, and binary-selection records;
10. absence of personal absolute paths in public text files; and
11. plot-only regeneration of all 24 data-derived manuscript figures; and
12. selected representative config and staged sweep agreement with the archived mode records.

The final recorded test runtime was 16.8 s. The plot-only command was also run
directly and completed successfully.

## Smoke execution

The smoke-sized entry points for representative LDPC, matched controls,
trajectory diagnostics, acquisition/stability, cross-code transfer, MAX-CUT,
and matched-$k_w$ 2-SAT all completed from the staged package. These runs use
two trials (or one trial per 2-SAT cell), shortened cycles, and one worker; they
are implementation checks, not new estimates of manuscript results. Temporary
smoke outputs were removed from the release directory after validation.

The first 2-SAT smoke attempt exposed two obsolete source-tree dependencies in
the preserved driver: its selected-parameter file and source-record hash list.
Both were redirected to the equivalent staged configuration and provenance
records. The rerun then completed. Scientific parameters and archived outputs
were not changed.

## Visual and file checks

Every regenerated PDF is one page and nonempty. Main Fig. 5 and Supplemental
Figs. S5, S8, and S9 were rendered for visual inspection. Legends, zero markers,
N/A labels, small-positive annotations, and three-way method identities were
readable and unclipped.

All Python sources compile under the recorded Python 3.10 environment. Cache,
log, and temporary output directories produced by validation are excluded from
the final staging copy.

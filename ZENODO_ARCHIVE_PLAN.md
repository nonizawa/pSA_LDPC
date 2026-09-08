# Large-data archive plan

The GitHub package contains all code, exact configurations, instance files,
compact raw/count records, processed summaries, validation records, and data
needed for plot-only reproduction. Full trajectory archives are preserved only
in the research workspace and were not copied into this release candidate.

| Excluded source collection | Approximate size | Scientific role | Needed for current figure/table regeneration? | Recommended archive action |
|---|---:|---|---|---|
| Response-alignment full trajectory collection | 550 MiB | Cycle/trajectory-level diagnostic provenance | No; processed by-cycle and trajectory summaries are included | Archive in Zenodo if full event-level reanalysis is desired |
| Natural-initialization acquisition trajectories | 152 MiB | First-passage and residence source trajectories | No; censored curves and summaries are included | Archive in Zenodo |
| Correct-start stability trajectories | 30 MiB | Escape/return source trajectories | No | Archive with acquisition trajectories |
| Alternative-initialization raw trajectory table | 28 MiB compressed, plus per-method source shards | Random/channel-hard event-level records | No; metadata, condition, trajectory, and first-passage summaries are included | Optional Zenodo addition |
| Initial jointly optimized 2-SAT archive | 79 MiB | Exploratory optimizer behavior retained for transparency | No; publication summaries are included | Optional supporting archive |
| Completed MAX-CUT structural sweep archive | 61 MiB | Full boundary-control sweep | No; selected summaries and sweep inputs are included | Optional supporting archive |

No file was deleted from the research workspace. A separate immutable archive
is preferable to Git LFS for these scientific source collections because it can
receive a citable DOI and preserve the manuscript-linked snapshot.

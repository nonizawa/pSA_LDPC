#!/usr/bin/env python3
"""Plot-only reproduction of the manuscript's data-derived figures."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = [
    ROOT / "scripts" / "figure_sources" / "make_consistency_figures.py",
    ROOT / "scripts" / "figure_sources" / "make_publication_figures.py",
    ROOT / "scripts" / "figure_sources" / "make_fig4_acquisition_retention.py",
    ROOT / "scripts" / "figure_sources" / "make_supplement_visual_cleanup_figures.py",
    ROOT / "scripts" / "plot_archived_studies.py",
]
EXPECTED = {
    "Fig2a_LDPC_BER_N96.pdf",
    "Fig2b_LDPC_BER_N192_N288.pdf",
    "Fig2c_LDPC_FER_N192_N288.pdf",
    "Fig3_matched_causal_controls.pdf",
    "Fig4a_channel_acquisition.pdf",
    "Fig4b_acquisition_retention_summary.pdf",
    "Fig5_cross_code_three_way_BER.pdf",
    "FigS_binary_kappa_sweep.pdf",
    "FigS_binary_highstat_comparison.pdf",
    "FigS_binary_trajectory_mechanism.pdf",
    "FigS_response_alignment_summary.pdf",
    "FigS_response_dynamics_by_size.pdf",
    "FigS_acquisition_first_passage.pdf",
    "FigS_post_acquisition_stability.pdf",
    "FigS_initialization_acquisition.pdf",
    "FigS_initialization_first_passage.pdf",
    "FigS_cross_code_size_summary.pdf",
    "FigS_cross_code_mechanism.pdf",
    "FigS_MAXCUT_quality.pdf",
    "FigS_MAXCUT_memory_sweep.pdf",
    "FigS_2SAT_initial_mode_distribution.pdf",
    "FigS_2SAT_initial_rho_vs_alpha.pdf",
    "FigS_2SAT_matched_kw_slices.pdf",
    "FigS_2SAT_matched_kw_heatmap.pdf",
}


def main() -> None:
    output = ROOT / "figures" / "regenerated"
    output.mkdir(parents=True, exist_ok=True)
    for stale in output.glob("*.pdf"):
        stale.unlink()
    for script in SCRIPTS:
        subprocess.run([sys.executable, str(script)], cwd=ROOT, check=True)
    missing = sorted(name for name in EXPECTED if not (output / name).is_file())
    if missing:
        raise SystemExit(f"Missing regenerated figures: {', '.join(missing)}")
    for extra in output.glob("*.pdf"):
        if extra.name not in EXPECTED:
            extra.unlink()
    print(f"Regenerated {len(EXPECTED)} data-derived figure files in {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

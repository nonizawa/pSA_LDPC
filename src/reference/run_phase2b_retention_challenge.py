#!/usr/bin/env python3
"""Counterfactual retention challenge initialized at the transmitted codeword.

This companion to the natural Phase 2b run removes the acquisition bottleneck:
all methods start from the same transmitted codeword with zero response memory,
the same channel observations, and paired Phase 2 seeds.  It is an explicitly
separate intervention and does not replace the natural zero-state trajectories.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import logging
import math
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
LDPC_DIR = ROOT / "src" / "ldpc"
if str(LDPC_DIR) not in sys.path:
    sys.path.insert(0, str(LDPC_DIR))

from auto_match_bp import split_total_trials  # noqa: E402
from ldpc_pbit import (  # noqa: E402
    add_awgn, awgn_channel_llr, bpsk_modulate, encode_message,
    make_stochastic_channel_values, parity_check_to_generator,
    random_regular_ldpc_parity_check,
)
from run_phase1_controls import (  # noqa: E402
    SIZE_SPECS, adjusted_params, config_hash, environment_record, load_anchor,
    locate_completed_size, parse_list, result_roots, write_csv, write_json,
)
from run_phase2_ldpc_dynamics import (  # noqa: E402
    FINAL_LAMBDAS, IMPLEMENTATION_MODES, VARIANT_CODES, _decision_code,
    _prepare_kernel_inputs, _shuffle_mapping,
)
from run_phase2b_acquisition_retention import (  # noqa: E402
    E_EXACT_ESCAPES, E_EXACT_OPPORTUNITIES, EXTRA_METRIC_COUNT,
    LOW_BER_THRESHOLD, PRIMARY_VARIANTS, SYNDROME_FRACTION_THRESHOLD,
    X_BACKFLIP_COUNT, X_BIT_ERROR_COUNT, X_CHANNEL_GAP_TO_TRUE, X_CHANNEL_SCORE,
    X_CORRECT_CYCLES, X_FLIP_COUNT, X_HARD_CHANNEL_DISAGREEMENTS,
    X_PREV_CORRECT_BITS, X_RELAXED_BASIN_CYCLES, X_SYNDROME_WEIGHT,
    _decode_events_channel_numba, _event_trajectory_fields, _fields_for,
    _t95, run_invariants,
)


PROFILE_DEFAULTS = {
    "smoke": {"sizes": ["N192_M96"], "trials": 2, "seed_count": 1,
              "cycles_override": 200, "cycle_bin": 20},
    "pilot": {"sizes": ["N192_M96"], "trials": 200, "seed_count": 10,
              "cycles_override": None, "cycle_bin": 100},
    "targeted": {"sizes": ["N192_M96", "N288_M144"], "trials": 200,
                 "seed_count": 10, "cycles_override": None, "cycle_bin": 100},
}

CYCLE_METRICS = [
    "state_BER", "syndrome_weight_fraction", "bit_flip_rate", "backflip_rate",
    "channel_alignment", "channel_alignment_gap_to_transmitted",
    "hard_channel_disagreement_rate", "exact_correct_fraction",
    "relaxed_basin_fraction",
]

CYCLE_FIELDS = ["size", "variant", "lambda_mem", "seed", "trial_index",
                "cycle_start", "cycle_end", "cycle_mid", "cycles_in_bin",
                *CYCLE_METRICS]

TRAJECTORY_METRICS = [
    "correct_residence_fraction", "correct_remaining_cycles_after_hit", "correct_cycles_after_hit",
    "correct_escape_probability", "correct_transition_opportunities", "correct_escape_count",
    "correct_return_probability", "correct_return_count", "correct_mean_return_time",
    "correct_first_residence_time",
    "correct_first_residence_censored", "relaxed_basin_residence_fraction",
    "relaxed_basin_escape_probability", "late_channel_alignment",
    "late_channel_alignment_gap_to_transmitted", "late_hard_channel_disagreement_rate",
    "mean_state_BER", "late_state_BER", "mean_syndrome_weight_fraction",
    "late_syndrome_weight_fraction", "mean_bit_flip_rate", "late_bit_flip_rate",
    "mean_backflip_rate", "late_backflip_rate",
]

TRAJECTORY_FIELDS = ["size", "variant", "implementation_mode", "lambda_mem", "seed",
                     "trial_index", "n_cycles", *TRAJECTORY_METRICS]


def _ratio(a, b):
    return float(a / b) if b > 0 else float("nan")


def _raw_cycles(n_cycles, cycle_bin):
    n_bins = (n_cycles + cycle_bin - 1) // cycle_bin
    raw = np.zeros((n_bins, 20), dtype=float)
    for index in range(n_bins):
        raw[index, 0] = min(cycle_bin, n_cycles - index * cycle_bin)
    return raw


def _aggregate_extra(raw, extra, n_bits, n_checks, late=False):
    if late:
        start = max(0, int(math.floor(0.75 * len(raw))))
        raw = raw[start:]; extra = extra[start:]
    cycles = float(np.sum(raw[:, 0])); totals = np.sum(extra, axis=0)
    return {
        "state_BER": _ratio(totals[X_BIT_ERROR_COUNT], cycles * n_bits),
        "syndrome_weight_fraction": _ratio(totals[X_SYNDROME_WEIGHT], cycles * n_checks),
        "bit_flip_rate": _ratio(totals[X_FLIP_COUNT], cycles * n_bits),
        "backflip_rate": _ratio(totals[X_BACKFLIP_COUNT], totals[X_PREV_CORRECT_BITS]),
        "channel_alignment": _ratio(totals[X_CHANNEL_SCORE], cycles * n_bits),
        "channel_alignment_gap_to_transmitted": _ratio(
            totals[X_CHANNEL_GAP_TO_TRUE], cycles * n_bits),
        "hard_channel_disagreement_rate": _ratio(
            totals[X_HARD_CHANNEL_DISAGREEMENTS], cycles * n_bits),
        "exact_correct_fraction": _ratio(totals[X_CORRECT_CYCLES], cycles),
        "relaxed_basin_fraction": _ratio(totals[X_RELAXED_BASIN_CYCLES], cycles),
    }


def _worker(task):
    P = np.asarray(task["P"], dtype=np.uint8); condition = task["condition"]
    params = condition["params"]; seed = int(task["seed"]); n_bits = P.shape[1]
    G = parity_check_to_generator(P); rate = G.shape[0] / n_bits
    channel_seed, pbit_seed = np.random.SeedSequence(seed).spawn(2)
    channel_rng = np.random.default_rng(channel_seed); pbit_rng = np.random.default_rng(pbit_seed)
    shuffle_rng = np.random.default_rng(np.random.SeedSequence([seed, 0x53485546]))
    prepared = _prepare_kernel_inputs(P, params)
    (P_packed, group_indices, group_counts, group_lengths,
     check_indices, check_lengths, I0_history) = prepared
    cycle_rows, trajectory_rows = [], []
    n_cycles = int(params["n_cycles"]); cycle_bin = int(task["cycle_bin"])
    raw_template = _raw_cycles(n_cycles, cycle_bin)
    for trial_index in range(int(task["n_trials"])):
        message = channel_rng.integers(0, 2, size=G.shape[0], dtype=np.uint8)
        codeword = encode_message(message, G)
        y = add_awgn(bpsk_modulate(codeword), 2.5, rate=rate, rng=channel_rng)
        llr = awgn_channel_llr(y, 2.5, rate=rate)
        channel_values = make_stochastic_channel_values(
            y, alpha=params["alpha"], bit_width=task["fixed_bit_width"],
            input_mode=task["channel_input_mode"], alpha_mode=params["alpha_mode"],
            ebno_db=2.5, rate=rate, channel_llr=llr)
        order, rank, offsets = _shuffle_mapping(n_bits, n_cycles, shuffle_rng)
        core_seed = int(pbit_rng.integers(0, 2**31 - 1))
        _decoded, _final, extra, events = _decode_events_channel_numba(
            P_packed, group_indices, group_counts, group_lengths, check_indices, check_lengths,
            np.asarray(channel_values, dtype=float), np.asarray(codeword, dtype=np.uint8),
            float(params["kw"]), float(params["kr"]), n_cycles, I0_history,
            float(params["psa_p"]), float(params.get("lambda_mem", 0.0)),
            int(VARIANT_CODES[condition["variant"]]), int(_decision_code(params["decision_method"])),
            int(params["burn_in"]), int(params["sample_window"]), cycle_bin,
            order, rank, offsets, 1, float(LOW_BER_THRESHOLD),
            float(SYNDROME_FRACTION_THRESHOLD), core_seed)
        raw = raw_template.copy()
        for bin_index in range(len(raw)):
            cycles = raw[bin_index, 0]; totals = extra[bin_index]
            start = bin_index * cycle_bin + 1; end = min(n_cycles, (bin_index + 1) * cycle_bin)
            metrics = {
                "state_BER": _ratio(totals[X_BIT_ERROR_COUNT], cycles * n_bits),
                "syndrome_weight_fraction": _ratio(totals[X_SYNDROME_WEIGHT], cycles * P.shape[0]),
                "bit_flip_rate": _ratio(totals[X_FLIP_COUNT], cycles * n_bits),
                "backflip_rate": _ratio(totals[X_BACKFLIP_COUNT], totals[X_PREV_CORRECT_BITS]),
                "channel_alignment": _ratio(totals[X_CHANNEL_SCORE], cycles * n_bits),
                "channel_alignment_gap_to_transmitted": _ratio(totals[X_CHANNEL_GAP_TO_TRUE], cycles * n_bits),
                "hard_channel_disagreement_rate": _ratio(totals[X_HARD_CHANNEL_DISAGREEMENTS], cycles * n_bits),
                "exact_correct_fraction": _ratio(totals[X_CORRECT_CYCLES], cycles),
                "relaxed_basin_fraction": _ratio(totals[X_RELAXED_BASIN_CYCLES], cycles),
            }
            cycle_rows.append({"size": condition["size"], "variant": condition["variant"],
                               "lambda_mem": condition["lambda_mem"], "seed": seed,
                               "trial_index": trial_index, "cycle_start": start, "cycle_end": end,
                               "cycle_mid": 0.5 * (start + end), "cycles_in_bin": int(cycles), **metrics})
        event_fields = _event_trajectory_fields(events, raw, extra, n_bits, n_cycles)
        overall = _aggregate_extra(raw, extra, n_bits, P.shape[0], False)
        late = _aggregate_extra(raw, extra, n_bits, P.shape[0], True)
        trajectory_rows.append({
            "size": condition["size"], "variant": condition["variant"],
            "implementation_mode": IMPLEMENTATION_MODES[condition["variant"]],
            "lambda_mem": condition["lambda_mem"], "seed": seed, "trial_index": trial_index,
            "n_cycles": n_cycles,
            **{field: event_fields[field] for field in TRAJECTORY_METRICS if field in event_fields},
            "mean_state_BER": overall["state_BER"], "late_state_BER": late["state_BER"],
            "mean_syndrome_weight_fraction": overall["syndrome_weight_fraction"],
            "late_syndrome_weight_fraction": late["syndrome_weight_fraction"],
            "mean_bit_flip_rate": overall["bit_flip_rate"], "late_bit_flip_rate": late["bit_flip_rate"],
            "mean_backflip_rate": overall["backflip_rate"], "late_backflip_rate": late["backflip_rate"],
        })
    return cycle_rows, trajectory_rows


class Stat:
    def __init__(self): self.values = []
    def add(self, value):
        try: value = float(value)
        except (TypeError, ValueError): return
        if math.isfinite(value): self.values.append(value)
    def summary(self):
        if not self.values:
            return {"n": 0, "mean": float("nan"), "sd": float("nan"), "sem": float("nan"),
                    "ci_low": float("nan"), "ci_high": float("nan"), "median": float("nan")}
        x = np.asarray(self.values); mean = float(np.mean(x)); n = len(x)
        sd = float(np.std(x, ddof=1)) if n > 1 else float("nan")
        sem = sd / math.sqrt(n) if n > 1 else float("nan"); half = 1.96 * sem if n > 1 else float("nan")
        return {"n": n, "mean": mean, "sd": sd, "sem": sem, "ci_low": mean-half,
                "ci_high": mean+half, "median": float(np.median(x))}


def _aggregate(rows, metrics, keys):
    groups = {}
    for row in rows:
        key = tuple(row[field] for field in keys)
        groups.setdefault(key, {metric: Stat() for metric in metrics})
        for metric in metrics: groups[key][metric].add(row.get(metric))
    output = []
    for key in sorted(groups):
        item = dict(zip(keys, key))
        for metric, stat in groups[key].items():
            for suffix, value in stat.summary().items(): item[f"{metric}_{suffix}"] = value
        output.append(item)
    return output


def _add_pooled(summary, rows, keys):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in keys)].append(row)
    for item in summary:
        subset = groups.get(tuple(item[field] for field in keys), [])
        remaining = sum(float(row["correct_remaining_cycles_after_hit"]) for row in subset)
        occupied = sum(float(row["correct_cycles_after_hit"]) for row in subset)
        opportunities = sum(float(row["correct_transition_opportunities"]) for row in subset)
        escapes = sum(float(row["correct_escape_count"]) for row in subset)
        returns = sum(float(row["correct_return_count"]) for row in subset)
        item["correct_residence_fraction_pooled"] = _ratio(occupied, remaining)
        item["correct_escape_probability_pooled"] = _ratio(escapes, opportunities)
        item["correct_return_probability_pooled"] = _ratio(returns, escapes)
    return summary


def _paired(rows):
    metrics = ["correct_residence_fraction", "correct_escape_probability", "correct_escape_count",
               "correct_return_probability", "late_state_BER", "late_syndrome_weight_fraction",
               "late_backflip_rate", "late_channel_alignment"]
    indexed = defaultdict(dict)
    for row in rows: indexed[(row["size"], row["seed"], row["trial_index"])][row["variant"]] = row
    diffs = defaultdict(list)
    for (size, seed, _), variants in indexed.items():
        if "additive" not in variants: continue
        for control, row in variants.items():
            if control == "additive": continue
            for metric in metrics:
                try: delta = float(row[metric]) - float(variants["additive"][metric])
                except (KeyError, ValueError, TypeError): continue
                if math.isfinite(delta): diffs[(size, control, metric, seed)].append(delta)
    seed_means = defaultdict(list); counts = defaultdict(int)
    for (size, control, metric, _seed), values in diffs.items():
        if values:
            seed_means[(size, control, metric)].append(float(np.mean(values)))
            counts[(size, control, metric)] += len(values)
    out = []
    for key in sorted(seed_means):
        x = np.asarray(seed_means[key]); n = len(x); mean = float(np.mean(x))
        sd = float(np.std(x, ddof=1)) if n > 1 else float("nan"); sem = sd/math.sqrt(n) if n>1 else float("nan")
        half = _t95(n)*sem if n>1 else float("nan")
        out.append({"size": key[0], "control": key[1], "reference": "additive",
                    "metric": key[2], "contrast_definition": "control_minus_additive",
                    "mean_difference": mean, "ci95_low": mean-half, "ci95_high": mean+half,
                    "seed_batch_sd": sd, "n_seed_batches": n, "n_paired_trajectories": counts[key]})
    return out


def _gzip_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_suffix(path.suffix+".tmp")
    with gzip.open(tmp, "wt", newline="") as h:
        w=csv.DictWriter(h,fieldnames=fields,extrasaction="ignore"); w.writeheader(); w.writerows(rows)
    tmp.replace(path)


def _figures(condition, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out_dir.mkdir(parents=True,exist_ok=True)
    colors={"pSA":"#4C78A8","additive":"#E45756","gain_only":"#F2CF5B","shuffled":"#B279A2"}
    variants=[v for v in PRIMARY_VARIANTS if any(r["variant"]==v for r in condition)]
    sizes=[s for s in ["N192_M96","N288_M144"] if any(r["size"]==s for r in condition)]
    specs=[("correct_residence_fraction_pooled","Correct-state residence fraction",False),
           ("correct_escape_probability_pooled","Escape probability / cycle",True),
           ("late_state_BER_mean","Late state BER",True),
           ("late_backflip_rate_mean","Late bit back-flip rate",True)]
    fig,axes=plt.subplots(2,2,figsize=(10.8,7.4)); x=np.arange(len(sizes)); width=.82/len(variants)
    for ax,(field,ylabel,log) in zip(axes.flat,specs):
        positives=[]
        for vi,v in enumerate(variants):
            means=[]; errs=[]
            for s in sizes:
                r=next(z for z in condition if z["size"]==s and z["variant"]==v)
                mean=float(r[field]); lo=float(r[field.replace("_mean","_ci_low")]); hi=float(r[field.replace("_mean","_ci_high")])
                means.append(mean); errs.append(max(mean-lo,hi-mean));
                if math.isfinite(mean) and mean>0: positives.append(mean)
            ax.bar(x-.41+width/2+vi*width,means,width,color=colors[v],label=v,yerr=errs,
                   error_kw={"linewidth":.8,"capsize":2})
        ax.set_xticks(x,[s.split('_')[0] for s in sizes]); ax.set_ylabel(ylabel); ax.grid(axis="y",alpha=.22)
        if log and positives: ax.set_yscale("log")
    handles,labels=axes[0,0].get_legend_handles_labels(); fig.suptitle("Correct-start retention challenge",y=.985)
    fig.legend(handles,labels,loc="upper center",bbox_to_anchor=(.5,.935),ncol=len(variants),frameon=False)
    fig.subplots_adjust(left=.09,right=.985,bottom=.08,top=.86,hspace=.30,wspace=.23)
    for ext in ["png","pdf"]: fig.savefig(out_dir/f"Fig_phase2b_retention_challenge.{ext}",dpi=300,bbox_inches="tight")
    plt.close(fig)


def parser():
    p=argparse.ArgumentParser(); p.add_argument("--profile",choices=PROFILE_DEFAULTS,default="smoke")
    p.add_argument("--sizes",default=None); p.add_argument("--trials",type=int,default=None)
    p.add_argument("--seed-count",type=int,default=None); p.add_argument("--seed",type=int,default=20260826)
    p.add_argument("--matrix-seed",type=int,default=0); p.add_argument("--n-workers",type=int,default=8)
    p.add_argument("--cycles-override",type=int,default=None); p.add_argument("--cycle-bin",type=int,default=None)
    p.add_argument("--fixed-bit-width",type=int,default=8); p.add_argument("--channel-input-mode",default="float")
    p.add_argument("--output-dir",default=str(ROOT/"phase2b_results")); p.add_argument("--run-name",default=None)
    p.add_argument("--snapshot-only",action="store_true",
                   help="Write the complete parameter/seed snapshot without rerunning trajectories")
    return p


def main():
    a=parser().parse_args(); d=PROFILE_DEFAULTS[a.profile]
    sizes=parse_list(a.sizes) if a.sizes else list(d["sizes"]); trials=a.trials or d["trials"]
    seed_count=a.seed_count or d["seed_count"]; cycles=a.cycles_override if a.cycles_override is not None else d["cycles_override"]
    cycle_bin=a.cycle_bin or d["cycle_bin"]; seeds=[a.seed+8022*i for i in range(seed_count)]
    run=Path(a.output_dir).expanduser().resolve()/(a.run_name or f"retention_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    run.mkdir(parents=True,exist_ok=True); log=logging.getLogger("retention_challenge"); log.handlers.clear(); log.setLevel(logging.INFO)
    fmt=logging.Formatter("%(asctime)s %(levelname)s %(message)s"); sh=logging.StreamHandler(sys.stdout); sh.setFormatter(fmt)
    fh=logging.FileHandler(run/"run.log"); fh.setFormatter(fmt); log.addHandler(sh); log.addHandler(fh)
    roots=result_roots(None); base={}; files={}
    for s in sizes:
        source=locate_completed_size(s,roots); base[s],path=load_anchor(source,"lambda"); files[s]=str(path)
    config={"protocol":"transmitted-codeword initial state with zero response memory","sizes":sizes,
            "variants":PRIMARY_VARIANTS,"trials":trials,"seeds":seeds,"cycle_bin":cycle_bin,
            "cycles_override":cycles,"EbNo_dB":2.5,"source_files":files}
    digest=config_hash(config); write_json(run/"effective_config.json",{**config,"config_hash":digest})
    conditions=[]
    for s in sizes:
        for v in PRIMARY_VARIANTS:
            lam=0.0 if v=="pSA" else FINAL_LAMBDAS[s]
            conditions.append({"size":s,"variant":v,"lambda_mem":lam,"params":adjusted_params(base[s],lam,cycles)})
    write_json(run/"parameters_and_seeds.json",{
        "config_hash":digest,"protocol":config["protocol"],"matrix_seed":int(a.matrix_seed),
        "trajectory_seed_construction":"seed + 8022 * seed_batch_index","seeds":seeds,
        "fixed_bit_width":int(a.fixed_bit_width),"channel_input_mode":a.channel_input_mode,
        "anchor_parameters":{
            s:{"source_parameter_file":files[s],"loaded_parameters":base[s]} for s in sizes
        },
        "effective_condition_parameters":[
            {"size":c["size"],"variant":c["variant"],
             "implementation_mode":IMPLEMENTATION_MODES[c["variant"]],
             "lambda_mem":c["lambda_mem"],"parameters":c["params"]}
            for c in conditions
        ],
    })
    if a.snapshot_only:
        log.info("Parameter/seed snapshot written without rerunning trajectories: %s",run)
        return
    inv=run_invariants(a.matrix_seed); write_json(run/"invariants.json",inv)
    if not inv["all_passed"]: raise RuntimeError("baseline invariant failed")
    write_json(run/"manifest.json",{"status":"running","conditions":len(conditions),"started_utc":datetime.now(timezone.utc).isoformat(),
                                    "environment":environment_record(),"config_hash":digest})
    start=time.perf_counter(); all_c=[]; all_t=[]
    for idx,c in enumerate(conditions,1):
        P=random_regular_ldpc_parity_check(SIZE_SPECS[c["size"]]["n_bits"],SIZE_SPECS[c["size"]]["n_checks"],3,6,seed=a.matrix_seed)
        tasks=[{"P":P,"condition":c,"seed":seed,"n_trials":n,"cycle_bin":cycle_bin,
                "fixed_bit_width":a.fixed_bit_width,"channel_input_mode":a.channel_input_mode}
               for seed,n in split_total_trials(trials,seeds)]
        log.info("[%d/%d] %s %s",idx,len(conditions),c["size"],c["variant"]); results=[]
        with ProcessPoolExecutor(max_workers=min(a.n_workers,len(tasks))) as ex:
            for f in as_completed([ex.submit(_worker,t) for t in tasks]): results.append(f.result())
        for cy,tr in results: all_c.extend(cy); all_t.extend(tr)
    cycle_summary=_aggregate(all_c,CYCLE_METRICS,["size","variant","lambda_mem","cycle_start","cycle_end","cycle_mid"])
    seed_cycle=_aggregate(all_c,CYCLE_METRICS,["size","variant","lambda_mem","seed","cycle_start","cycle_end","cycle_mid"])
    condition=_add_pooled(
        _aggregate(all_t,TRAJECTORY_METRICS,["size","variant","lambda_mem"]),
        all_t,["size","variant","lambda_mem"])
    seed_summary=_add_pooled(
        _aggregate(all_t,TRAJECTORY_METRICS,["size","variant","lambda_mem","seed"]),
        all_t,["size","variant","lambda_mem","seed"])
    paired=_paired(all_t)
    _gzip_csv(run/"raw_trajectory_metrics.csv.gz",all_c,CYCLE_FIELDS)
    write_csv(run/"trajectory_summary.csv",all_t,TRAJECTORY_FIELDS); write_csv(run/"condition_summary.csv",condition,_fields_for(condition))
    write_csv(run/"seed_batch_summary.csv",seed_summary,_fields_for(seed_summary)); write_csv(run/"paired_statistics.csv",paired,_fields_for(paired))
    write_csv(run/"by_cycle_summary.csv",cycle_summary,_fields_for(cycle_summary)); write_csv(run/"by_seed_cycle.csv",seed_cycle,_fields_for(seed_cycle))
    _figures(condition,run/"figures"); elapsed=time.perf_counter()-start
    write_json(run/"manifest.json",{"status":"complete","conditions":len(conditions),"trajectory_count":len(all_t),
                                    "elapsed_seconds":elapsed,"finished_utc":datetime.now(timezone.utc).isoformat(),"config_hash":digest})
    log.info("Done in %.2f s: %s",elapsed,run)


if __name__=="__main__": main()

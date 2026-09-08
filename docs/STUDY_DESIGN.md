# Study design

The studies answer different questions and must not be pooled into a single
effect estimate.

## Independently optimized representative-code benchmark

One fixed random-regular (3,6) matrix is used at each of $N=96,192,288$.
Memoryless pSA, additive $\lambda$-pSA, and finite-response $\tau$-pSA are
searched separately, refined, and evaluated at 2.0, 2.5, and 3.0 dB. The exact
selected dictionaries are in `configs/ldpc/representative_benchmark.json` and
candidate/refinement records are under
`data/processed/representative_benchmark/search_records/`. This is a
best-achieved mode comparison, not a matched response-rule intervention.

## Matched causal controls

The additive-selected nonmemory fields and readout are fixed while the response
rule is changed among matched pSA, additive, normalized, gain-only, and shuffled
response. Shared seed batches support paired contrasts. These controls test
gain, normalization, history, and bit identity without claiming independent
optimization of every arm.

## Binary-state self-feedback

The stored-quantity control uses $d_i^{(t)}=q_i^{(t)}+\kappa x_i^{(t)}$ with
the matched nonmemory parameters. Only $\kappa$ is swept. The score-selected
values are 0.25, 0.50, and 0.75 for $N=96,192,288$. At $N=192$, the manuscript
also retains the separately validated BER-favorable $\kappa=0.25$ sensitivity;
it does not overwrite the score-selected $\kappa=0.50$ record. The equal-range
$\kappa=\lambda$ comparison is a third, distinct estimand.

## Response alignment, acquisition, and stability

Trajectory diagnostics use fixed matched-control parameters. Natural all-zero
starts measure acquisition; a separate correct-start intervention measures
post-acquisition stability. Nonreaching first-passage observations are
censored, not assigned a terminal time. Initialization robustness repeats the
mechanism analysis from paired random and channel-hard-decision states on the
representative $N=192$ and $N=288$ matrices at 2.5 dB.

## Cross-code fixed transfer

Ten independently generated matrices (C00--C09) are used at each size. The
complete additive and independently selected pSA-specific packages are
transferred intact, including their selected readouts, with no per-code tuning.
This is a package-level transfer-performance comparison. A separate matched
$\lambda=0$ arm keeps the additive nonmemory fields and readout fixed and removes
only response reinforcement; it is the rule-isolating ablation. The two effects
must not be interchanged.

## Boundary controls

MAX-CUT reports co-optimized endpoint behavior and is marginal or conditional.
Random 2-SAT uses a matched-$k_w$ finite-response $\rho$ control and finds no
independent positive finite-response effect. Neither study is a matched causal
test of the additive LDPC rule on another problem.

## Conventional BP reference

The reference curve uses the included custom NumPy binary sum-product decoder
on the same representative matrices: flooding updates, a maximum of 50
iterations, zero-syndrome early stopping, clipped LLR/messages, and no external
decoder library. It is a conventional reference, not a hardware/runtime claim.

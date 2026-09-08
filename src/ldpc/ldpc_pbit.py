import argparse
import csv
import sys
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy import sparse

try:
    from numba import njit
except Exception:  # pragma: no cover - optional acceleration
    njit = None


# ============================================================
# GF(2) linear algebra
# ============================================================

def gf2_rref(A):
    A = np.array(A, dtype=np.uint8) % 2
    m, n = A.shape

    pivot_cols = []
    row = 0

    for col in range(n):
        pivot = None
        for r in range(row, m):
            if A[r, col] == 1:
                pivot = r
                break

        if pivot is None:
            continue

        if pivot != row:
            A[[row, pivot]] = A[[pivot, row]]

        for r in range(m):
            if r != row and A[r, col] == 1:
                A[r] ^= A[row]

        pivot_cols.append(col)
        row += 1

        if row == m:
            break

    return A, pivot_cols


def gf2_rank(A):
    _, pivot_cols = gf2_rref(A)
    return len(pivot_cols)


def parity_check_to_generator(P):
    """
    P G^T = 0 を満たす generator matrix G をGF(2)上で生成する。
    P: shape = (M, N)
    G: shape = (K, N), K = N - rank(P)
    """
    H = np.array(P, dtype=np.uint8) % 2
    M, N = H.shape

    rref, pivot_cols = gf2_rref(H)
    free_cols = [c for c in range(N) if c not in pivot_cols]

    G_rows = []

    for free_col in free_cols:
        x = np.zeros(N, dtype=np.uint8)
        x[free_col] = 1

        for row, pivot_col in enumerate(pivot_cols):
            x[pivot_col] = rref[row, free_col]

        G_rows.append(x)

    G = np.array(G_rows, dtype=np.uint8)

    return G


def encode_message(message_bits, G):
    message_bits = np.array(message_bits, dtype=np.uint8) % 2
    G = np.array(G, dtype=np.uint8) % 2

    if len(message_bits) != G.shape[0]:
        raise ValueError("message_bits length must match G.shape[0]")

    codeword = message_bits @ G
    return codeword % 2


def syndrome(P, codeword):
    P = np.array(P, dtype=np.uint8) % 2
    codeword = np.array(codeword, dtype=np.uint8) % 2
    return (P @ codeword) % 2


# ============================================================
# Parity-check matrix examples / generators
# ============================================================

def toy_parity_check_3x6():
    return np.array([
        [1, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 0, 1],
        [1, 0, 0, 1, 1, 0],
    ], dtype=np.uint8)


def hamming_parity_check(m):
    """
    Hamming code parity-check matrix.
    m=3 -> (7,4), m=4 -> (15,11)
    """
    if m < 2:
        raise ValueError("m must be >= 2")

    n_bits = 2 ** m - 1
    H = np.zeros((m, n_bits), dtype=np.uint8)

    for col in range(1, n_bits + 1):
        for row in range(m):
            H[row, col - 1] = (col >> row) & 1

    return H


def random_regular_ldpc_parity_check(
    n_bits,
    n_checks,
    variable_degree=3,
    check_degree=None,
    seed=None,
    max_attempts=1000,
):
    """
    Configuration-model random regular LDPC parity-check matrix.
    P has shape (n_checks, n_bits).
    """
    n_bits = int(n_bits)
    n_checks = int(n_checks)
    variable_degree = int(variable_degree)

    if n_bits <= 0 or n_checks <= 0:
        raise ValueError("n_bits and n_checks must be positive")

    if variable_degree <= 0:
        raise ValueError("variable_degree must be positive")

    total_edges = n_bits * variable_degree

    if check_degree is None:
        if total_edges % n_checks != 0:
            raise ValueError("n_bits * variable_degree must be divisible by n_checks")
        check_degree = total_edges // n_checks
    else:
        check_degree = int(check_degree)

    if check_degree <= 0:
        raise ValueError("check_degree must be positive")

    if total_edges != n_checks * check_degree:
        raise ValueError("n_bits * variable_degree must equal n_checks * check_degree")

    rng = np.random.default_rng(seed)
    var_sockets = np.repeat(np.arange(n_bits), variable_degree)
    check_sockets = np.repeat(np.arange(n_checks), check_degree)

    for _ in range(max_attempts):
        rng.shuffle(var_sockets)
        rng.shuffle(check_sockets)
        H = np.zeros((n_checks, n_bits), dtype=np.uint8)
        duplicate = False

        for row, col in zip(check_sockets, var_sockets):
            if H[row, col] == 1:
                duplicate = True
                break
            H[row, col] = 1

        if duplicate:
            continue

        if (
            np.all(np.sum(H, axis=0) == variable_degree)
            and np.all(np.sum(H, axis=1) == check_degree)
        ):
            return H

    raise RuntimeError("Failed to generate random regular LDPC matrix; try another seed or size")


def load_parity_check_matrix(path):
    delimiter = "," if str(path).lower().endswith(".csv") else None
    H = np.loadtxt(path, dtype=np.uint8, delimiter=delimiter)

    if H.ndim != 2:
        raise ValueError("Parity-check matrix file must contain a 2D matrix")

    return H % 2


def make_parity_check_matrix(args):
    if args.p_matrix_file is not None:
        return load_parity_check_matrix(args.p_matrix_file)

    if args.p_matrix == "toy":
        return toy_parity_check_3x6()

    if args.p_matrix == "hamming74":
        return hamming_parity_check(3)

    if args.p_matrix == "hamming1511":
        return hamming_parity_check(4)

    if args.p_matrix == "random_regular":
        return random_regular_ldpc_parity_check(
            n_bits=args.n_bits,
            n_checks=args.n_checks,
            variable_degree=args.variable_degree,
            check_degree=args.check_degree,
            seed=args.matrix_seed,
        )

    raise ValueError(f"Unknown parity-check matrix type: {args.p_matrix}")


# ============================================================
# Channel model
# ============================================================

def bpsk_modulate(bits):
    """
    0 -> +1
    1 -> -1
    """
    bits = np.array(bits, dtype=float)
    return 1.0 - 2.0 * bits


def awgn_noise_variance(ebno_db, rate=1.0):
    ebno = 10 ** (ebno_db / 10.0)
    return 1.0 / (2.0 * rate * ebno)


def add_awgn(x, ebno_db, rate=1.0, rng=None):
    sigma = np.sqrt(awgn_noise_variance(ebno_db, rate))
    if rng is None:
        noise = np.random.normal(0.0, sigma, size=x.shape)
    else:
        noise = rng.normal(0.0, sigma, size=x.shape)
    return x + noise


def awgn_channel_llr(y, ebno_db, rate=1.0):
    """
    BPSK: 0 -> +1, 1 -> -1
    AWGNの受信値yからBP用のチャネルLLR log(P(y|0)/P(y|1)) を作る。
    """
    sigma2 = awgn_noise_variance(ebno_db, rate)
    return 2.0 * y / sigma2


def quantize_fixed_signed(x, bit_width):
    """
    [-1, 1] の値を signed fixed-point 相当に量子化
    """
    levels = 2 ** (bit_width - 1) - 1
    x_clip = np.clip(x, -1.0, 1.0)
    q = np.round(x_clip * levels)
    return q / levels


def stochastic_bits_from_values(values, rng=None):
    if rng is None:
        rnd = np.random.uniform(-1.0, 1.0, size=values.shape)
    else:
        rnd = rng.uniform(-1.0, 1.0, size=values.shape)

    return (values < rnd).astype(np.uint8)


def y_to_fixed_stochastic_values(y, alpha=1.0, bit_width=8):
    """
    チャネル出力yを固定小数点化したストカスティック入力値に変換する。
    bipolar表現では E[1 - 2r] = zq となる。
    """
    z = np.tanh(alpha * y)
    return quantize_fixed_signed(z, bit_width)


def y_to_float_stochastic_values(y, alpha=1.0):
    """
    チャネル出力yから直接ストカスティック入力値を作る。
    固定小数点化は行わない。
    """
    return np.tanh(alpha * y)


def channel_argument(
    y,
    alpha=1.0,
    alpha_mode="fixed",
    ebno_db=None,
    rate=1.0,
    channel_llr=None,
):
    if alpha_mode == "fixed":
        return alpha * y

    if alpha_mode == "snr":
        if ebno_db is None:
            raise ValueError("ebno_db is required when alpha_mode='snr'")
        sigma2 = awgn_noise_variance(ebno_db, rate)
        return alpha * y / sigma2

    if alpha_mode == "llr":
        if channel_llr is None:
            if ebno_db is None:
                raise ValueError("ebno_db is required when alpha_mode='llr'")
            channel_llr = awgn_channel_llr(y, ebno_db, rate=rate)
        return alpha * channel_llr

    raise ValueError("alpha_mode must be 'fixed', 'snr', or 'llr'")


def make_stochastic_channel_values(
    y,
    alpha=1.0,
    bit_width=8,
    input_mode="float",
    alpha_mode="fixed",
    ebno_db=None,
    rate=1.0,
    channel_llr=None,
):
    arg = channel_argument(
        y,
        alpha=alpha,
        alpha_mode=alpha_mode,
        ebno_db=ebno_db,
        rate=rate,
        channel_llr=channel_llr,
    )
    z = np.tanh(arg)

    if input_mode == "float":
        return z
    if input_mode == "fixed":
        return quantize_fixed_signed(z, bit_width)
    raise ValueError("input_mode must be 'float' or 'fixed'")


def y_to_stochastic_input(
    y,
    alpha=1.0,
    bit_width=8,
    rng=None,
    input_mode="float",
    alpha_mode="fixed",
    ebno_db=None,
    rate=1.0,
):
    """
    z_x = tanh(alpha * y_x)
    r_x = (z_x < rnd[-1, 1]) ? 1 : 0
    """
    channel_values = make_stochastic_channel_values(
        y,
        alpha=alpha,
        bit_width=bit_width,
        input_mode=input_mode,
        alpha_mode=alpha_mode,
        ebno_db=ebno_db,
        rate=rate,
    )
    r = stochastic_bits_from_values(channel_values, rng=rng)
    return r, channel_values


# ============================================================
# LDPC p-bit connection
# ============================================================

def build_pbit_connections(P):
    """
    Pから各p-bitの接続を生成する。

    あるcheck nodeに [w0, w1, w2, w3] が接続されている場合，
    w0の更新には [w1, w2, w3] のXORを使う。
    """
    P = np.array(P, dtype=np.uint8) % 2
    n_bits = P.shape[1]

    connections = []

    for i in range(n_bits):
        groups = []
        rows = np.where(P[:, i] == 1)[0]

        for row in rows:
            cols = np.where(P[row, :] == 1)[0]
            others = [int(c) for c in cols if c != i]

            if len(others) > 0:
                groups.append(others)

        connections.append(groups)

    return connections


def xor_bits(bits):
    result = 0
    for b in bits:
        result ^= int(b)
    return result


_PBIT_CONNECTION_CACHE = {}


def _packed_pbit_connections(P):
    P = np.array(P, dtype=np.uint8) % 2
    key = (P.shape, P.tobytes())
    cached = _PBIT_CONNECTION_CACHE.get(key)
    if cached is not None:
        return cached

    connections = build_pbit_connections(P)
    max_groups = max((len(groups) for groups in connections), default=0)
    max_group_len = max(
        (len(group) for groups in connections for group in groups),
        default=0,
    )
    group_indices = np.full(
        (P.shape[1], max_groups, max_group_len),
        -1,
        dtype=np.int32,
    )
    group_counts = np.zeros(P.shape[1], dtype=np.int32)
    group_lengths = np.zeros((P.shape[1], max_groups), dtype=np.int32)

    for bit, groups in enumerate(connections):
        group_counts[bit] = len(groups)
        for group_idx, group in enumerate(groups):
            group_lengths[bit, group_idx] = len(group)
            for pos, other_bit in enumerate(group):
                group_indices[bit, group_idx, pos] = int(other_bit)

    cached = (P, group_indices, group_counts, group_lengths)
    _PBIT_CONNECTION_CACHE[key] = cached
    return cached


if njit is not None:
    @njit(cache=True)
    def _syndrome_weight_numba(P, bits):
        weight = 0
        for row in range(P.shape[0]):
            parity = 0
            for col in range(P.shape[1]):
                if P[row, col] == 1:
                    parity ^= int(bits[col])
            weight += parity
        return weight


    @njit(cache=True)
    def _channel_score_numba(bits, channel_values):
        score = 0.0
        for i in range(bits.shape[0]):
            score += channel_values[i] * (1.0 - 2.0 * bits[i])
        return score


    @njit(cache=True)
    def _decode_pbits_psa_numba(
        P,
        group_indices,
        group_counts,
        group_lengths,
        channel_values,
        kw,
        kr,
        n_cycles,
        I0_history,
        psa_p,
        lambda_mem,
        psa_variant,
        decision_code,
        burn_in,
        sample_window,
        seed,
    ):
        np.random.seed(seed)
        n_bits = P.shape[1]
        w = np.zeros(n_bits, dtype=np.uint8)
        Itanh_hold = np.zeros(n_bits, dtype=np.float64)
        decoded = np.zeros(n_bits, dtype=np.uint8)
        ones_count = np.zeros(n_bits, dtype=np.int64)
        sample_count = 0
        best_state = np.zeros(n_bits, dtype=np.uint8)
        best_syndrome_weight = P.shape[0] + 1
        best_channel_score = -1.0e300

        sample_start = burn_in
        available_samples = n_cycles - burn_in
        if available_samples <= 0:
            sample_start = n_cycles - 1
        elif sample_window > 0 and sample_window < available_samples:
            sample_start = n_cycles - sample_window

        for cycle in range(n_cycles):
            I0 = I0_history[cycle]
            w_prev_cycle = w.copy()
            Itanh_hold_prev_cycle = Itanh_hold.copy()

            for bit in range(n_bits):
                parity_sum = 0
                for group_idx in range(group_counts[bit]):
                    target = 0
                    for pos in range(group_lengths[bit, group_idx]):
                        other_bit = group_indices[bit, group_idx, pos]
                        target ^= int(w_prev_cycle[other_bit])
                    if target == 1:
                        parity_sum += 1
                    else:
                        parity_sum -= 1

                rnd_channel = -1.0 + 2.0 * np.random.random()
                channel_bit = 1 if channel_values[bit] < rnd_channel else 0
                channel_term = 1 if channel_bit == 1 else -1

                if psa_p > 0.0 and np.random.random() < psa_p:
                    w[bit] = w_prev_cycle[bit]
                    continue

                I = kw * parity_sum + kr * channel_term
                Itanh_hold_bit = np.tanh(I0 * I)
                rnd = -1.0 + 2.0 * np.random.random()
                if psa_variant == 1 or psa_variant == 4:
                    deterministic = lambda_mem * Itanh_hold_prev_cycle[bit] + Itanh_hold_bit
                elif psa_variant == 2:
                    deterministic = (
                        lambda_mem * Itanh_hold_prev_cycle[bit] + Itanh_hold_bit
                    ) / (1.0 + lambda_mem)
                elif psa_variant == 3:
                    deterministic = (1.0 + lambda_mem) * Itanh_hold_bit
                else:
                    deterministic = Itanh_hold_bit
                w[bit] = 1 if deterministic + rnd >= 0.0 else 0
                Itanh_hold[bit] = Itanh_hold_bit

            if cycle >= sample_start:
                if decision_code == 1:
                    sample_count += 1
                    for bit in range(n_bits):
                        ones_count[bit] += int(w[bit])
                elif decision_code == 2:
                    syndrome_weight = _syndrome_weight_numba(P, w)
                    channel_score = _channel_score_numba(w, channel_values)
                    if (
                        syndrome_weight < best_syndrome_weight
                        or (
                            syndrome_weight == best_syndrome_weight
                            and channel_score > best_channel_score
                        )
                    ):
                        best_syndrome_weight = syndrome_weight
                        best_channel_score = channel_score
                        best_state[:] = w[:]

        if decision_code == 1:
            if sample_count <= 0:
                decoded[:] = w[:]
            else:
                for bit in range(n_bits):
                    decoded[bit] = 1 if 2 * ones_count[bit] > sample_count else 0
        elif decision_code == 2:
            decoded[:] = best_state[:]
        else:
            decoded[:] = w[:]

        return decoded


    @njit(cache=True)
    def _decode_pbits_ssa_numba(
        P,
        group_indices,
        group_counts,
        group_lengths,
        channel_values,
        kw,
        kr,
        n_cycles,
        I0_history,
        nrnd,
        lambda_mem,
        lambda_out,
        use_lambda_mem,
        use_dual_memory,
        decision_code,
        burn_in,
        sample_window,
        seed,
    ):
        np.random.seed(seed)
        n_bits = P.shape[1]
        w = np.zeros(n_bits, dtype=np.uint8)
        Itanh = np.zeros(n_bits, dtype=np.float64)
        Itanh_hold = np.zeros(n_bits, dtype=np.float64)
        decoded = np.zeros(n_bits, dtype=np.uint8)
        ones_count = np.zeros(n_bits, dtype=np.int64)
        sample_count = 0
        best_state = np.zeros(n_bits, dtype=np.uint8)
        best_syndrome_weight = P.shape[0] + 1
        best_channel_score = -1.0e300

        sample_start = burn_in
        available_samples = n_cycles - burn_in
        if available_samples <= 0:
            sample_start = n_cycles - 1
        elif sample_window > 0 and sample_window < available_samples:
            sample_start = n_cycles - sample_window

        for cycle in range(n_cycles):
            I0 = I0_history[cycle]
            w_prev_cycle = w.copy()
            Itanh_prev_cycle = Itanh.copy()
            Itanh_hold_prev_cycle = Itanh_hold.copy()

            for bit in range(n_bits):
                parity_sum = 0
                for group_idx in range(group_counts[bit]):
                    target = 0
                    for pos in range(group_lengths[bit, group_idx]):
                        other_bit = group_indices[bit, group_idx, pos]
                        target ^= int(w_prev_cycle[other_bit])
                    if target == 1:
                        parity_sum += 1
                    else:
                        parity_sum -= 1

                rnd_channel = -1.0 + 2.0 * np.random.random()
                channel_bit = 1 if channel_values[bit] < rnd_channel else 0
                channel_term = 1 if channel_bit == 1 else -1

                I = kw * parity_sum + kr * channel_term
                rnd = -1.0 if np.random.random() < 0.5 else 1.0
                I_vector = I + nrnd * rnd

                if use_lambda_mem == 1:
                    discriminant = lambda_mem * Itanh_prev_cycle[bit] + I_vector
                else:
                    discriminant = Itanh_prev_cycle[bit] + I_vector

                if discriminant >= I0:
                    Itanh[bit] = I0
                elif discriminant < -I0:
                    Itanh[bit] = -I0
                else:
                    Itanh[bit] = discriminant

                if use_dual_memory == 1:
                    Itanh_hold[bit] = np.tanh(Itanh[bit])
                    rnd_out = -1.0 + 2.0 * np.random.random()
                    w[bit] = 1 if Itanh_hold[bit] + lambda_out * Itanh_hold_prev_cycle[bit] + rnd_out >= 0.0 else 0
                else:
                    Itanh_hold[bit] = Itanh[bit]
                    w[bit] = 1 if Itanh[bit] >= 0.0 else 0

            if cycle >= sample_start:
                if decision_code == 1:
                    sample_count += 1
                    for bit in range(n_bits):
                        ones_count[bit] += int(w[bit])
                elif decision_code == 2:
                    syndrome_weight = _syndrome_weight_numba(P, w)
                    channel_score = _channel_score_numba(w, channel_values)
                    if (
                        syndrome_weight < best_syndrome_weight
                        or (
                            syndrome_weight == best_syndrome_weight
                            and channel_score > best_channel_score
                        )
                    ):
                        best_syndrome_weight = syndrome_weight
                        best_channel_score = channel_score
                        best_state[:] = w[:]

        if decision_code == 1:
            if sample_count <= 0:
                decoded[:] = w[:]
            else:
                for bit in range(n_bits):
                    decoded[bit] = 1 if 2 * ones_count[bit] > sample_count else 0
        elif decision_code == 2:
            decoded[:] = best_state[:]
        else:
            decoded[:] = w[:]

        return decoded
else:
    _decode_pbits_psa_numba = None
    _decode_pbits_ssa_numba = None


def decode_pbits_fast(
    P,
    channel_values,
    kw=2.0,
    kr=4.0,
    n_cycles=100,
    mode="SSA",
    I0_min=0.1,
    I0_max=5.0,
    psa_p=0.0,
    lambda_mem=0.0,
    response_lambda=0.0,
    lambda_out=0.0,
    I0_schedule_type="linear",
    I0_schedule_shape=1.0,
    I0_hold_fraction=0.0,
    decision_method="last",
    burn_in=0,
    sample_window=0,
    rng=None,
    nrnd=0.0,
):
    if mode not in {
        "pSA",
        "lambda_pSA",
        "lambda_pSA_normalized",
        "gain_only_pSA",
        "lambda_pSA_extended",
        "tau_pSA",
        "SSA",
        "lambda_SSA",
        "dual_memory",
    }:
        return None

    if channel_values is None:
        return None

    decision_codes = {"last": 0, "majority": 1, "best_state": 2}
    if decision_method not in decision_codes:
        return None

    if mode == "tau_pSA":
        decoded, _, _, _ = update_pbits_jacobi_vectorized(
            P=P,
            channel_values=channel_values,
            kw=kw,
            kr=kr,
            n_cycles=n_cycles,
            mode=mode,
            I0_min=I0_min,
            I0_max=I0_max,
            nrnd=nrnd,
            psa_p=psa_p,
            lambda_mem=0.0,
            response_lambda=response_lambda,
            lambda_out=0.0,
            I0_schedule_type=I0_schedule_type,
            I0_schedule_shape=I0_schedule_shape,
            I0_hold_fraction=I0_hold_fraction,
            decision_method=decision_method,
            burn_in=burn_in,
            sample_window=sample_window,
            rng=rng,
            store_history=decision_method in {"majority", "best_state"},
        )
        return decoded

    P, group_indices, group_counts, group_lengths = _packed_pbit_connections(P)
    channel_values = np.array(channel_values, dtype=np.float64)
    I0_history = make_I0_schedule(
        I0_min,
        I0_max,
        int(n_cycles),
        I0_schedule_type,
        shape=I0_schedule_shape,
        hold_fraction=I0_hold_fraction,
    ).astype(np.float64)

    if rng is None:
        seed = int(np.random.randint(0, 2**31 - 1))
    else:
        seed = int(rng.integers(0, 2**31 - 1))

    if mode in {
        "pSA",
        "lambda_pSA",
        "lambda_pSA_normalized",
        "gain_only_pSA",
        "lambda_pSA_extended",
    }:
        if _decode_pbits_psa_numba is None:
            return None
        psa_variant = {
            "pSA": 0,
            "lambda_pSA": 1,
            "lambda_pSA_normalized": 2,
            "gain_only_pSA": 3,
            "lambda_pSA_extended": 4,
        }[mode]
        return _decode_pbits_psa_numba(
            P,
            group_indices,
            group_counts,
            group_lengths,
            channel_values,
            float(kw),
            float(kr),
            int(n_cycles),
            I0_history,
            float(psa_p),
            float(lambda_mem),
            int(psa_variant),
            int(decision_codes[decision_method]),
            int(burn_in),
            int(sample_window),
            seed,
        )

    if _decode_pbits_ssa_numba is None:
        return None
    return _decode_pbits_ssa_numba(
        P,
        group_indices,
        group_counts,
        group_lengths,
        channel_values,
        float(kw),
        float(kr),
        int(n_cycles),
        I0_history,
        float(nrnd),
        float(lambda_mem),
        float(lambda_out),
        1 if mode in {"lambda_SSA", "dual_memory"} else 0,
        1 if mode == "dual_memory" else 0,
        int(decision_codes[decision_method]),
        int(burn_in),
        int(sample_window),
        seed,
    )


# ============================================================
# pSA / SSA p-bit operation
# ============================================================

def fun_Itanh(Itanh_prev, I_vector, I0):
    discriminant = Itanh_prev + I_vector

    if discriminant >= I0:
        return I0
    elif discriminant < -I0:
        return -I0
    else:
        return discriminant


PBIT_MODES = [
    "pSA",
    "SSA",
    "lambda_pSA",
    "lambda_pSA_normalized",
    "gain_only_pSA",
    "lambda_pSA_extended",
    "tau_pSA",
    "lambda_SSA",
    "dual_memory",
]


def is_psa_mode(mode):
    return mode in {
        "pSA",
        "lambda_pSA",
        "lambda_pSA_normalized",
        "gain_only_pSA",
        "lambda_pSA_extended",
        "tau_pSA",
    }


def is_ssa_mode(mode):
    return mode in {"SSA", "lambda_SSA", "dual_memory"}


def is_lambda_mode(mode):
    return mode in {"lambda_pSA", "lambda_SSA", "dual_memory"}


def pbit_operation(
    I,
    Itanh_prev,
    I0,
    mode="SSA",
    nrnd=0.0,
    rng=None,
    lambda_mem=0.0,
    response_lambda=0.0,
    lambda_out=0.0,
    Itanh_hold_prev=0.0,
):
    """
    mode = "pSA", "SSA", "lambda_pSA", "tau_pSA", "lambda_SSA", or "dual_memory"
    """

    if mode == "pSA":
        if rng is None:
            rnd = np.random.uniform(-1.0, 1.0)
        else:
            rnd = rng.uniform(-1.0, 1.0)
        Itanh_hold = np.tanh(I0 * I)
        Itanh = Itanh_hold + rnd

    elif mode == "lambda_pSA":
        if rng is None:
            rnd = np.random.uniform(-1.0, 1.0)
        else:
            rnd = rng.uniform(-1.0, 1.0)
        Itanh_hold = np.tanh(I0 * I)
        Itanh = lambda_mem * Itanh_hold_prev + Itanh_hold + rnd

    elif mode == "lambda_pSA_normalized":
        if rng is None:
            rnd = np.random.uniform(-1.0, 1.0)
        else:
            rnd = rng.uniform(-1.0, 1.0)
        Itanh_hold = np.tanh(I0 * I)
        Itanh = (lambda_mem * Itanh_hold_prev + Itanh_hold) / (1.0 + lambda_mem) + rnd

    elif mode == "gain_only_pSA":
        if rng is None:
            rnd = np.random.uniform(-1.0, 1.0)
        else:
            rnd = rng.uniform(-1.0, 1.0)
        Itanh_hold = np.tanh(I0 * I)
        Itanh = (1.0 + lambda_mem) * Itanh_hold + rnd

    elif mode == "lambda_pSA_extended":
        if rng is None:
            rnd = np.random.uniform(-1.0, 1.0)
        else:
            rnd = rng.uniform(-1.0, 1.0)
        Itanh_hold = np.tanh(I0 * I)
        Itanh = lambda_mem * Itanh_hold_prev + Itanh_hold + rnd

    elif mode == "tau_pSA":
        if rng is None:
            rnd = np.random.uniform(-1.0, 1.0)
        else:
            rnd = rng.uniform(-1.0, 1.0)
        H_new = np.tanh(I0 * I)
        Itanh_hold = response_lambda * Itanh_hold_prev + (1.0 - response_lambda) * H_new
        Itanh = Itanh_hold + rnd

    elif mode == "SSA":
        if rng is None:
            rnd = np.random.choice([-1.0, 1.0])
        else:
            rnd = rng.choice([-1.0, 1.0])
        I_vector = I + nrnd * rnd
        Itanh = fun_Itanh(Itanh_prev, I_vector, I0)
        Itanh_hold = Itanh

    elif mode == "lambda_SSA":
        if rng is None:
            rnd = np.random.choice([-1.0, 1.0])
        else:
            rnd = rng.choice([-1.0, 1.0])
        I_vector = I + nrnd * rnd
        Itanh = fun_Itanh(lambda_mem * Itanh_prev, I_vector, I0)
        Itanh_hold = Itanh

    elif mode == "dual_memory":
        if rng is None:
            rnd_int = np.random.choice([-1.0, 1.0])
            rnd_out = np.random.uniform(-1.0, 1.0)
        else:
            rnd_int = rng.choice([-1.0, 1.0])
            rnd_out = rng.uniform(-1.0, 1.0)
        I_vector = I + nrnd * rnd_int
        Itanh = fun_Itanh(lambda_mem * Itanh_prev, I_vector, I0)
        Itanh_hold = np.tanh(Itanh)
        w = 1 if Itanh_hold + lambda_out * Itanh_hold_prev + rnd_out >= 0 else 0
        return w, Itanh, Itanh_hold

    else:
        raise ValueError(f"unsupported p-bit mode: {mode}")

    w = 1 if Itanh >= 0 else 0

    return w, Itanh, Itanh_hold


def make_I0_schedule(
    I0_min,
    I0_max,
    n_cycles,
    schedule="linear",
    shape=1.0,
    hold_fraction=0.0,
):
    if n_cycles <= 1:
        return np.array([I0_max], dtype=float)

    shape = max(1e-6, float(shape))
    hold_fraction = clamp(float(hold_fraction), 0.0, 0.95)
    t = np.linspace(0.0, 1.0, n_cycles)

    if schedule == "linear":
        progress = t ** shape

    elif schedule == "constant":
        return np.full(n_cycles, I0_min, dtype=float)

    elif schedule == "exponential":
        k = shape
        if abs(k) < 1e-9:
            progress = t
        else:
            progress = (np.exp(k * t) - 1.0) / (np.exp(k) - 1.0)

    elif schedule == "cosine":
        progress = (1.0 - np.cos(np.pi * t)) / 2.0
        progress = progress ** shape

    elif schedule == "piecewise":
        progress = np.zeros_like(t)
        active = t > hold_fraction
        if np.any(active):
            local_t = (t[active] - hold_fraction) / (1.0 - hold_fraction)
            progress[active] = local_t ** shape

    else:
        raise ValueError(
            "schedule must be 'linear', 'constant', 'exponential', 'cosine', or 'piecewise'"
        )

    return I0_min + (I0_max - I0_min) * progress


def channel_likelihood_score(bits, channel_values):
    if channel_values is None:
        return 0.0

    bipolar_bits = 1.0 - 2.0 * np.array(bits, dtype=float)
    return float(np.sum(np.array(channel_values, dtype=float) * bipolar_bits))


def select_history_samples(history, burn_in=0, sample_window=0):
    history = np.array(history, dtype=np.uint8)

    if len(history) <= 1:
        return history[-1:]

    states = history[1:]
    burn_in = max(0, int(burn_in))

    if burn_in >= len(states):
        states = states[-1:]
    else:
        states = states[burn_in:]

    sample_window = int(sample_window)
    if sample_window > 0 and sample_window < len(states):
        states = states[-sample_window:]

    return states


def select_decoded_state(
    P,
    history,
    channel_values=None,
    decision_method="last",
    burn_in=0,
    sample_window=0,
):
    history = np.array(history, dtype=np.uint8)

    if decision_method == "last":
        return history[-1].copy()

    samples = select_history_samples(
        history,
        burn_in=burn_in,
        sample_window=sample_window,
    )

    if decision_method == "majority":
        return (np.mean(samples, axis=0) > 0.5).astype(np.uint8)

    if decision_method == "best_state":
        best_state = samples[0]
        best_syndrome_weight = np.sum(syndrome(P, best_state))
        best_channel_score = channel_likelihood_score(best_state, channel_values)

        for state in samples[1:]:
            syndrome_weight = np.sum(syndrome(P, state))
            channel_score = channel_likelihood_score(state, channel_values)

            if (
                syndrome_weight < best_syndrome_weight
                or (
                    syndrome_weight == best_syndrome_weight
                    and channel_score > best_channel_score
                )
            ):
                best_state = state
                best_syndrome_weight = syndrome_weight
                best_channel_score = channel_score

        return best_state.copy()

    raise ValueError("decision_method must be 'last', 'majority', or 'best_state'")


# ============================================================
# BP LDPC decoder
# ============================================================

def bp_decode(
    P,
    channel_llr,
    max_iter=50,
    llr_clip=50.0,
    early_stop=True,
):
    """
    Sum-product BP decoder for binary LDPC codes.

    channel_llr[j] = log(P(y_j | bit=0) / P(y_j | bit=1))
    """
    P = np.array(P, dtype=np.uint8) % 2
    channel_llr = np.array(channel_llr, dtype=float)

    M, N = P.shape

    if len(channel_llr) != N:
        raise ValueError("Length of channel_llr must match number of columns of P")

    check_to_vars = [np.where(P[m, :] == 1)[0] for m in range(M)]
    var_to_checks = [np.where(P[:, n] == 1)[0] for n in range(N)]

    v_to_c = np.zeros((M, N), dtype=float)
    c_to_v = np.zeros((M, N), dtype=float)
    clipped_llr = np.clip(channel_llr, -llr_clip, llr_clip)

    for m in range(M):
        for n in check_to_vars[m]:
            v_to_c[m, n] = clipped_llr[n]

    decoded = (clipped_llr < 0).astype(np.uint8)

    for iteration in range(1, max_iter + 1):
        for m, cols in enumerate(check_to_vars):
            for n in cols:
                other_cols = cols[cols != n]
                tanh_values = np.tanh(0.5 * v_to_c[m, other_cols])
                product = np.prod(tanh_values) if len(tanh_values) > 0 else 1.0
                product = np.clip(product, -1.0 + 1e-12, 1.0 - 1e-12)
                c_to_v[m, n] = np.clip(2.0 * np.arctanh(product), -llr_clip, llr_clip)

        posterior_llr = clipped_llr.copy()

        for n, rows in enumerate(var_to_checks):
            posterior_llr[n] += np.sum(c_to_v[rows, n])

            for m in rows:
                other_rows = rows[rows != m]
                v_to_c[m, n] = clipped_llr[n] + np.sum(c_to_v[other_rows, n])
                v_to_c[m, n] = np.clip(v_to_c[m, n], -llr_clip, llr_clip)

        decoded = (posterior_llr < 0).astype(np.uint8)

        if early_stop and np.all(syndrome(P, decoded) == 0):
            return decoded, True, iteration

    return decoded, np.all(syndrome(P, decoded) == 0), max_iter


# ============================================================
# p-bit LDPC decoder
# ============================================================



# ============================================================
# Vectorized Jacobi p-bit update
# ============================================================

PBIT_MODES = {
    "pSA",
    "lambda_pSA",
    "lambda_pSA_normalized",
    "gain_only_pSA",
    "lambda_pSA_extended",
    "tau_pSA",
    "SSA",
    "lambda_SSA",
    "dual_memory",
}


def _as_rng(seed=None, rng=None):
    if rng is not None:
        return rng
    return np.random.default_rng(seed)


def _validate_pbit_inputs(P, r, channel_values, mode, psa_p, lambda_mem, response_lambda, lambda_out):
    if mode not in PBIT_MODES:
        raise ValueError(f"unsupported p-bit mode: {mode}")
    if not 0.0 <= psa_p <= 1.0:
        raise ValueError("psa_p must be in [0, 1]")
    if mode == "lambda_pSA" and not 0.0 <= lambda_mem <= 0.95:
        raise ValueError("lambda_mem for lambda_pSA must be in [0, 0.95]")
    if mode in {"lambda_pSA_normalized", "gain_only_pSA", "lambda_pSA_extended"} and not 0.0 <= lambda_mem <= 2.0:
        raise ValueError(f"lambda_mem for {mode} must be in [0, 2]")
    if mode == "tau_pSA" and not 0.0 <= response_lambda < 1.0:
        raise ValueError("response_lambda for tau_pSA must be in [0, 1)")
    if mode in {"lambda_SSA", "dual_memory"} and not 0.0 <= lambda_mem <= 1.0:
        raise ValueError(f"lambda_mem for {mode} must be in [0, 1]")
    if mode == "dual_memory" and not 0.0 <= lambda_out <= 0.5:
        raise ValueError("lambda_out for dual_memory must be in [0, 0.5]")

    P = np.asarray(P, dtype=np.uint8) % 2
    if P.ndim != 2:
        raise ValueError("P must be a 2D parity-check matrix")

    n_bits = P.shape[1]
    if r is None and channel_values is None:
        raise ValueError("Either r or channel_values must be specified")

    if r is not None:
        r = np.asarray(r, dtype=np.uint8) % 2
        if r.shape[0] != n_bits:
            raise ValueError("Length of r must match number of columns of P")

    if channel_values is not None:
        channel_values = np.asarray(channel_values, dtype=float)
        if channel_values.shape[0] != n_bits:
            raise ValueError("Length of channel_values must match number of columns of P")

    return P, r, channel_values


@lru_cache(maxsize=32)
def _cached_sparse_struct(shape, p_bytes):
    P = np.frombuffer(p_bytes, dtype=np.uint8).reshape(shape)
    H = sparse.csr_matrix(P, dtype=np.uint8)
    coo = H.tocoo()
    edge_rows = coo.row.astype(np.int64)
    edge_cols = coo.col.astype(np.int64)
    return H, edge_rows, edge_cols


def _sparse_struct(P):
    P_c = np.ascontiguousarray(P, dtype=np.uint8)
    return _cached_sparse_struct(P_c.shape, P_c.tobytes())


def _stochastic_channel_bits_vectorized(channel_values, rng):
    rnd = rng.uniform(-1.0, 1.0, size=channel_values.shape[0])
    return (channel_values < rnd).astype(np.uint8)


def _parity_sum_jacobi_vectorized(H, edge_rows, edge_cols, w_prev):
    # check_parity[m] = XOR of all bits in check node m.
    # For edge (m, n), XOR of all other bits in that check is
    # check_parity[m] XOR w_prev[n].
    check_parity = np.asarray(H.dot(w_prev) % 2, dtype=np.uint8).ravel()
    target = np.bitwise_xor(check_parity[edge_rows], w_prev[edge_cols])
    edge_feedback = np.where(target == 1, 1.0, -1.0)
    return np.bincount(edge_cols, weights=edge_feedback, minlength=w_prev.shape[0])


def update_pbits_jacobi_vectorized(
    P,
    r=None,
    channel_values=None,
    kw=2.0,
    kr=4.0,
    n_cycles=100,
    mode="SSA",
    I0_min=0.1,
    I0_max=5.0,
    nrnd=0.0,
    psa_p=0.0,
    lambda_mem=0.0,
    response_lambda=0.0,
    lambda_out=0.0,
    I0_schedule_type="linear",
    I0_schedule_shape=1.0,
    I0_hold_fraction=0.0,
    decision_method="last",
    burn_in=0,
    sample_window=0,
    seed=None,
    rng=None,
    store_history=True,
):
    """Vectorized CPU Jacobi implementation of p-bit LDPC decoding.

    Drop-in replacement for update_pbits(...). It keeps the same return format:
    decoded, history, Itanh_history, I0_history.
    """
    P, r, channel_values = _validate_pbit_inputs(
        P, r, channel_values, mode, psa_p, lambda_mem, response_lambda, lambda_out
    )
    rng = _as_rng(seed=seed, rng=rng)

    H, edge_rows, edge_cols = _sparse_struct(P)
    n_bits = P.shape[1]

    w = np.zeros(n_bits, dtype=np.uint8)
    Itanh = np.zeros(n_bits, dtype=float)
    Itanh_hold = np.zeros(n_bits, dtype=float)

    I0_history = make_I0_schedule(
        I0_min,
        I0_max,
        n_cycles,
        I0_schedule_type,
        shape=I0_schedule_shape,
        hold_fraction=I0_hold_fraction,
    )

    need_history = store_history or decision_method in {"majority", "best_state"}
    if need_history:
        history = np.empty((n_cycles + 1, n_bits), dtype=np.uint8)
        Itanh_history = np.empty((n_cycles + 1, n_bits), dtype=float)
        history[0] = w
        Itanh_history[0] = Itanh
    else:
        history = None
        Itanh_history = None

    for cycle in range(n_cycles):
        I0 = float(I0_history[cycle])
        w_prev = w.copy()
        Itanh_prev = Itanh.copy()
        Itanh_hold_prev = Itanh_hold.copy()

        parity_sum = _parity_sum_jacobi_vectorized(H, edge_rows, edge_cols, w_prev)

        if channel_values is not None:
            r_cycle = _stochastic_channel_bits_vectorized(channel_values, rng)
        else:
            r_cycle = r
        channel_term = np.where(r_cycle == 1, 1.0, -1.0)
        I = kw * parity_sum + kr * channel_term

        if mode in {
            "pSA",
            "lambda_pSA",
            "lambda_pSA_normalized",
            "gain_only_pSA",
            "lambda_pSA_extended",
        }:
            Itanh_hold_new = np.tanh(I0 * I)
            rnd = rng.uniform(-1.0, 1.0, size=n_bits)
            if mode == "lambda_pSA":
                discriminant = lambda_mem * Itanh_hold_prev + Itanh_hold_new + rnd
            elif mode == "lambda_pSA_extended":
                discriminant = lambda_mem * Itanh_hold_prev + Itanh_hold_new + rnd
            elif mode == "lambda_pSA_normalized":
                discriminant = (lambda_mem * Itanh_hold_prev + Itanh_hold_new) / (1.0 + lambda_mem) + rnd
            elif mode == "gain_only_pSA":
                discriminant = (1.0 + lambda_mem) * Itanh_hold_new + rnd
            else:
                discriminant = Itanh_hold_new + rnd
            w_new = (discriminant >= 0.0).astype(np.uint8)
            Itanh_new = discriminant
            if psa_p > 0.0:
                hold_mask = rng.uniform(0.0, 1.0, size=n_bits) < psa_p
                w_new[hold_mask] = w_prev[hold_mask]
                Itanh_new[hold_mask] = Itanh_prev[hold_mask]
                Itanh_hold_new[hold_mask] = Itanh_hold_prev[hold_mask]
            w = w_new
            Itanh = Itanh_new
            Itanh_hold = Itanh_hold_new

        elif mode == "tau_pSA":
            # tau_pSA is implemented as a Jacobi/parallel update:
            # all inputs are computed from w_prev and Itanh_hold_prev.
            H_new = np.tanh(I0 * I)
            H_eff = response_lambda * Itanh_hold_prev + (1.0 - response_lambda) * H_new
            rnd = rng.uniform(-1.0, 1.0, size=n_bits)
            discriminant = H_eff + rnd
            w_new = (discriminant >= 0.0).astype(np.uint8)
            if psa_p > 0.0:
                hold_mask = rng.uniform(0.0, 1.0, size=n_bits) < psa_p
                w_new[hold_mask] = w_prev[hold_mask]
                discriminant[hold_mask] = Itanh_prev[hold_mask]
                H_eff[hold_mask] = Itanh_hold_prev[hold_mask]
            w = w_new
            Itanh = discriminant
            Itanh_hold = H_eff

        elif mode in {"SSA", "lambda_SSA"}:
            rnd = rng.choice(np.array([-1.0, 1.0]), size=n_bits)
            I_vector = I + nrnd * rnd
            if mode == "lambda_SSA":
                discriminant = lambda_mem * Itanh_prev + I_vector
            else:
                discriminant = Itanh_prev + I_vector
            Itanh = np.clip(discriminant, -I0, I0)
            Itanh_hold = Itanh.copy()
            w = (Itanh >= 0.0).astype(np.uint8)

        elif mode == "dual_memory":
            rnd_int = rng.choice(np.array([-1.0, 1.0]), size=n_bits)
            I_vector = I + nrnd * rnd_int
            discriminant = lambda_mem * Itanh_prev + I_vector
            Itanh = np.clip(discriminant, -I0, I0)
            Itanh_hold = np.tanh(Itanh)
            rnd_out = rng.uniform(-1.0, 1.0, size=n_bits)
            out_discriminant = Itanh_hold + lambda_out * Itanh_hold_prev + rnd_out
            w = (out_discriminant >= 0.0).astype(np.uint8)

        if need_history:
            history[cycle + 1] = w
            Itanh_history[cycle + 1] = Itanh

    if need_history:
        decoded = select_decoded_state(
            P=P,
            history=history,
            channel_values=channel_values,
            decision_method=decision_method,
            burn_in=burn_in,
            sample_window=sample_window,
        )
        return decoded, history, Itanh_history, I0_history

    decoded = w.copy()
    compact_history = np.vstack([np.zeros(n_bits, dtype=np.uint8), decoded])
    compact_Itanh_history = np.vstack([np.zeros(n_bits, dtype=float), Itanh.copy()])
    return decoded, compact_history, compact_Itanh_history, I0_history


# Public default: vectorized Jacobi implementation.
def update_pbits(*args, **kwargs):
    return update_pbits_jacobi_vectorized(*args, **kwargs)


def update_pbits_serial_jacobi(
    P,
    r=None,
    channel_values=None,
    kw=2.0,
    kr=4.0,
    n_cycles=100,
    mode="SSA",
    I0_min=0.1,
    I0_max=5.0,
    nrnd=0.0,
    psa_p=0.0,
    lambda_mem=0.0,
    response_lambda=0.0,
    lambda_out=0.0,
    I0_schedule_type="linear",
    I0_schedule_shape=1.0,
    I0_hold_fraction=0.0,
    decision_method="last",
    burn_in=0,
    sample_window=0,
    seed=None,
    rng=None,
):
    if seed is not None:
        rng = np.random.default_rng(seed)

    if not 0.0 <= psa_p <= 1.0:
        raise ValueError("psa_p must be in [0, 1]")
    if mode == "lambda_pSA" and not 0.0 <= lambda_mem <= 0.95:
        raise ValueError("lambda_mem for lambda_pSA must be in [0, 0.95]")
    if mode in {"lambda_pSA_normalized", "gain_only_pSA", "lambda_pSA_extended"} and not 0.0 <= lambda_mem <= 2.0:
        raise ValueError(f"lambda_mem for {mode} must be in [0, 2]")
    if mode == "tau_pSA" and not 0.0 <= response_lambda < 1.0:
        raise ValueError("response_lambda for tau_pSA must be in [0, 1)")
    if mode == "lambda_SSA" and not 0.0 <= lambda_mem <= 1.0:
        raise ValueError("lambda_mem for lambda_SSA must be in [0, 1]")
    if mode == "dual_memory" and not 0.0 <= lambda_mem <= 1.0:
        raise ValueError("lambda_mem for dual_memory must be in [0, 1]")
    if mode == "dual_memory" and not 0.0 <= lambda_out <= 0.5:
        raise ValueError("lambda_out for dual_memory must be in [0, 0.5]")

    P = np.array(P, dtype=np.uint8) % 2

    n_bits = P.shape[1]

    if r is None and channel_values is None:
        raise ValueError("Either r or channel_values must be specified")

    if r is not None:
        r = np.array(r, dtype=np.uint8) % 2

        if len(r) != n_bits:
            raise ValueError("Length of r must match number of columns of P")

    if channel_values is not None:
        channel_values = np.array(channel_values, dtype=float)

        if len(channel_values) != n_bits:
            raise ValueError("Length of channel_values must match number of columns of P")

    connections = build_pbit_connections(P)

    w = np.zeros(n_bits, dtype=np.uint8)
    Itanh = np.zeros(n_bits, dtype=float)
    Itanh_hold = np.zeros(n_bits, dtype=float)

    history = [w.copy()]
    Itanh_history = [Itanh.copy()]
    I0_history = make_I0_schedule(
        I0_min,
        I0_max,
        n_cycles,
        I0_schedule_type,
        shape=I0_schedule_shape,
        hold_fraction=I0_hold_fraction,
    )

    for cycle in range(n_cycles):
        I0 = I0_history[cycle]
        w_prev_cycle = w.copy()
        Itanh_prev_cycle = Itanh.copy()
        Itanh_hold_prev_cycle = Itanh_hold.copy()
        if channel_values is not None:
            r_cycle = stochastic_bits_from_values(channel_values, rng=rng)
        else:
            r_cycle = r

        for i in range(n_bits):
            parity_sum = 0

            for group in connections[i]:
                target = xor_bits(w_prev_cycle[group])

                if target == 1:
                    parity_sum += 1
                else:
                    parity_sum -= 1

            channel_term = 1 if r_cycle[i] == 1 else -1

            I = kw * parity_sum + kr * channel_term

            if mode in {
                "pSA",
                "lambda_pSA",
                "lambda_pSA_normalized",
                "gain_only_pSA",
                "lambda_pSA_extended",
            } and psa_p > 0.0:
                if rng is None:
                    hold_output = np.random.uniform(0.0, 1.0) < psa_p
                else:
                    hold_output = rng.uniform(0.0, 1.0) < psa_p

                if hold_output:
                    w[i] = w_prev_cycle[i]
                    continue

            w_i, Itanh_i, Itanh_hold_i = pbit_operation(
                I=I,
                Itanh_prev=Itanh_prev_cycle[i],
                I0=I0,
                mode=mode,
                nrnd=nrnd,
                rng=rng,
                lambda_mem=lambda_mem,
                response_lambda=response_lambda,
                lambda_out=lambda_out,
                Itanh_hold_prev=Itanh_hold_prev_cycle[i],
            )

            w[i] = w_i
            if mode == "tau_pSA" and psa_p > 0.0:
                hold_output = rng.uniform(0.0, 1.0) < psa_p if rng is not None else np.random.uniform(0.0, 1.0) < psa_p
                if hold_output:
                    w[i] = w_prev_cycle[i]
            Itanh[i] = Itanh_i
            Itanh_hold[i] = Itanh_hold_i

        history.append(w.copy())
        Itanh_history.append(Itanh.copy())

    history = np.array(history)
    decoded = select_decoded_state(
        P=P,
        history=history,
        channel_values=channel_values,
        decision_method=decision_method,
        burn_in=burn_in,
        sample_window=sample_window,
    )

    return decoded, history, np.array(Itanh_history), I0_history


# ============================================================
# Simulation
# ============================================================

def simulate_ldpc_decoder(
    P,
    fixed_bit_width=8,
    channel_input_mode="float",
    alpha_mode="fixed",
    ebno_db_list=np.arange(0, 8, 1),
    alpha=1.0,
    kw=2.0,
    kr=4.0,
    n_cycles=100,
    n_trials=100,
    mode="SSA",
    I0_min=0.1,
    I0_max=5.0,
    nrnd=0.0,
    psa_p=0.0,
    lambda_mem=0.0,
    response_lambda=0.0,
    lambda_out=0.0,
    I0_schedule_type="linear",
    I0_schedule_shape=1.0,
    I0_hold_fraction=0.0,
    decision_method="last",
    burn_in=0,
    sample_window=0,
    use_all_zero_message=False,
    bp_max_iter=50,
    bp_llr_clip=50.0,
    seed=None,
    verbose=True,
    print_matrices=False,
):
    if seed is not None:
        seed_sequence = np.random.SeedSequence(seed)
        channel_seed_sequence, pbit_seed_sequence = seed_sequence.spawn(2)
        channel_rng = np.random.default_rng(channel_seed_sequence)
        pbit_rng = np.random.default_rng(pbit_seed_sequence)
    else:
        channel_rng = np.random.default_rng()
        pbit_rng = np.random.default_rng()

    P = np.array(P, dtype=np.uint8) % 2
    G = parity_check_to_generator(P)

    N = P.shape[1]
    K = G.shape[0]
    rate = K / N

    if verbose:
        print("P shape:", P.shape)
        print("G shape:", G.shape)
        print("Code rate:", rate)
        print("Rank(P):", gf2_rank(P))
        check_product = (P @ G.T) % 2
        print("Check P @ G.T mod 2:", "OK" if np.all(check_product == 0) else "NG")

        if print_matrices:
            print("P:")
            print(P)
            print("G:")
            print(G)
            print("P @ G.T mod 2:")
            print(check_product)

    results = []

    for ebno_db in ebno_db_list:
        bit_errors = 0
        frame_errors = 0
        bp_bit_errors = 0
        bp_frame_errors = 0
        bp_converged_frames = 0
        bp_total_iterations = 0
        total_bits = 0

        for _ in range(n_trials):
            if use_all_zero_message:
                message_bits = np.zeros(K, dtype=np.uint8)
            else:
                message_bits = channel_rng.integers(0, 2, size=K, dtype=np.uint8)

            codeword_bits = encode_message(message_bits, G)

            syn = syndrome(P, codeword_bits)
            if np.any(syn != 0):
                raise RuntimeError("Generated codeword does not satisfy P c^T = 0")

            x = bpsk_modulate(codeword_bits)
            y = add_awgn(x, ebno_db, rate=rate, rng=channel_rng)

            channel_llr = awgn_channel_llr(y, ebno_db, rate=rate)
            bp_decoded, bp_converged, bp_iterations = bp_decode(
                P=P,
                channel_llr=channel_llr,
                max_iter=bp_max_iter,
                llr_clip=bp_llr_clip,
            )

            channel_values = make_stochastic_channel_values(
                y,
                alpha=alpha,
                bit_width=fixed_bit_width,
                input_mode=channel_input_mode,
                alpha_mode=alpha_mode,
                ebno_db=ebno_db,
                rate=rate,
                channel_llr=channel_llr,
            )

            decoded, history, Itanh_history, I0_history = update_pbits(
                P=P,
                channel_values=channel_values,
                kw=kw,
                kr=kr,
                n_cycles=n_cycles,
                mode=mode,
                I0_min=I0_min,
                I0_max=I0_max,
                nrnd=nrnd,
                psa_p=psa_p,
                lambda_mem=lambda_mem,
                response_lambda=response_lambda,
                lambda_out=lambda_out,
                I0_schedule_type=I0_schedule_type,
                I0_schedule_shape=I0_schedule_shape,
                I0_hold_fraction=I0_hold_fraction,
                decision_method=decision_method,
                burn_in=burn_in,
                sample_window=sample_window,
                rng=pbit_rng,
            )

            errors = np.sum(decoded != codeword_bits)
            bp_errors = np.sum(bp_decoded != codeword_bits)

            bit_errors += errors
            frame_errors += 1 if errors > 0 else 0
            bp_bit_errors += bp_errors
            bp_frame_errors += 1 if bp_errors > 0 else 0
            bp_converged_frames += 1 if bp_converged else 0
            bp_total_iterations += bp_iterations
            total_bits += N

        ber = bit_errors / total_bits
        fer = frame_errors / n_trials
        bp_ber = bp_bit_errors / total_bits
        bp_fer = bp_frame_errors / n_trials

        results.append({
            "EbNo_dB": ebno_db,
            "BER": ber,
            "FER": fer,
            "pbit_BER": ber,
            "pbit_FER": fer,
            "BP_BER": bp_ber,
            "BP_FER": bp_fer,
            "BP_convergence_rate": bp_converged_frames / n_trials,
            "BP_avg_iterations": bp_total_iterations / n_trials,
        })

    return G, results


def summarize_results(results):
    return {
        "mean_pbit_BER": float(np.mean([res["pbit_BER"] for res in results])),
        "mean_pbit_FER": float(np.mean([res["pbit_FER"] for res in results])),
        "mean_BP_BER": float(np.mean([res["BP_BER"] for res in results])),
        "mean_BP_FER": float(np.mean([res["BP_FER"] for res in results])),
        "EbNo_dB_values": format_float_list([res["EbNo_dB"] for res in results]),
        "pbit_BER_by_EbNo": format_float_list([res["pbit_BER"] for res in results]),
        "pbit_FER_by_EbNo": format_float_list([res["pbit_FER"] for res in results]),
        "BP_BER_by_EbNo": format_float_list([res["BP_BER"] for res in results]),
        "BP_FER_by_EbNo": format_float_list([res["BP_FER"] for res in results]),
    }


def format_float_list(values):
    return ";".join(f"{float(value):.12g}" for value in values)


def format_int_list(values):
    return ";".join(str(int(value)) for value in values)


def default_run_result_path(mode, P, suffix):
    n_checks, n_bits = np.array(P).shape
    if suffix == "csv":
        filename = f"{mode}_{n_checks}x{n_bits}_run_by_ebno.csv"
    elif suffix == "png":
        filename = f"{mode}_{n_checks}x{n_bits}_run_ber_fer.png"
    else:
        filename = f"{mode}_{n_checks}x{n_bits}_run.{suffix}"
    return Path("results") / filename


def default_search_ebno_result_path(search_csv):
    search_csv = Path(search_csv)
    return search_csv.with_name(f"{search_csv.stem}_by_ebno.csv")


def save_run_results_csv(results, csv_path, mode, params, n_trials, n_cycles):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "mode",
        "n_trials",
        "n_cycles",
        "EbNo_dB",
        "pbit_BER",
        "pbit_FER",
        "BP_BER",
        "BP_FER",
        "BP_convergence_rate",
        "BP_avg_iterations",
        "kw",
        "kr",
        "alpha",
        "alpha_mode",
        "nrnd",
        "psa_p",
        "lambda_mem",
        "response_lambda",
        "lambda_out",
        "I0_min",
        "I0_max",
        "I0_schedule_type",
        "I0_schedule_shape",
        "I0_hold_fraction",
        "decision_method",
        "burn_in",
        "sample_window",
    ]

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for result in results:
            writer.writerow({
                "mode": mode,
                "n_trials": n_trials,
                "n_cycles": n_cycles,
                "EbNo_dB": result["EbNo_dB"],
                "pbit_BER": result["pbit_BER"],
                "pbit_FER": result["pbit_FER"],
                "BP_BER": result["BP_BER"],
                "BP_FER": result["BP_FER"],
                "BP_convergence_rate": result["BP_convergence_rate"],
                "BP_avg_iterations": result["BP_avg_iterations"],
                **params,
            })

    return csv_path


def save_run_results_plot(results, plot_path):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    plot_path = Path(plot_path)
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    ebno = np.array([res["EbNo_dB"] for res in results], dtype=float)
    pbit_ber = np.array([res["pbit_BER"] for res in results], dtype=float)
    bp_ber = np.array([res["BP_BER"] for res in results], dtype=float)
    pbit_fer = np.array([res["pbit_FER"] for res in results], dtype=float)
    bp_fer = np.array([res["BP_FER"] for res in results], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), dpi=160)
    axes[0].plot(ebno, pbit_ber, marker="o", label="p-bit")
    axes[0].plot(ebno, bp_ber, marker="x", linestyle="--", label="BP")
    axes[0].set_title("BER")
    axes[0].set_xlabel("Eb/No [dB]")
    axes[0].set_ylabel("BER")
    axes[0].grid(True, alpha=0.35)
    axes[0].legend()

    axes[1].plot(ebno, pbit_fer, marker="o", label="p-bit")
    axes[1].plot(ebno, bp_fer, marker="x", linestyle="--", label="BP")
    axes[1].set_title("FER")
    axes[1].set_xlabel("Eb/No [dB]")
    axes[1].set_ylabel("FER")
    axes[1].grid(True, alpha=0.35)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def default_cycle_sweep_path(mode, P, suffix):
    n_checks, n_bits = np.array(P).shape
    return Path("results") / f"{mode}_{n_checks}x{n_bits}_cycle_sweep.{suffix}"


def run_cycle_sweep(
    P,
    cycle_values,
    fixed_bit_width=8,
    channel_input_mode="float",
    alpha_mode="fixed",
    ebno_db_list=np.arange(4, 5, 1),
    alpha=1.0,
    kw=2.0,
    kr=4.0,
    n_trials=100,
    mode="SSA",
    I0_min=0.1,
    I0_max=5.0,
    nrnd=0.0,
    psa_p=0.0,
    lambda_mem=0.0,
    response_lambda=0.0,
    lambda_out=0.0,
    I0_schedule_type="linear",
    I0_schedule_shape=1.0,
    I0_hold_fraction=0.0,
    decision_method="last",
    burn_in=0,
    sample_window=0,
    use_all_zero_message=False,
    bp_max_iter=50,
    bp_llr_clip=50.0,
    seed=0,
):
    rows = []
    for n_cycles in cycle_values:
        _, results = simulate_ldpc_decoder(
            P=P,
            fixed_bit_width=fixed_bit_width,
            channel_input_mode=channel_input_mode,
            alpha_mode=alpha_mode,
            ebno_db_list=ebno_db_list,
            alpha=alpha,
            kw=kw,
            kr=kr,
            n_cycles=int(n_cycles),
            n_trials=n_trials,
            mode=mode,
            I0_min=I0_min,
            I0_max=I0_max,
            nrnd=nrnd,
            psa_p=psa_p,
            lambda_mem=lambda_mem,
            response_lambda=response_lambda,
            lambda_out=lambda_out,
            I0_schedule_type=I0_schedule_type,
            I0_schedule_shape=I0_schedule_shape,
            I0_hold_fraction=I0_hold_fraction,
            decision_method=decision_method,
            burn_in=burn_in,
            sample_window=sample_window,
            use_all_zero_message=use_all_zero_message,
            bp_max_iter=bp_max_iter,
            bp_llr_clip=bp_llr_clip,
            seed=seed,
            verbose=False,
        )
        for result in results:
            rows.append({
                "mode": mode,
                "n_cycles": int(n_cycles),
                "n_trials": n_trials,
                "EbNo_dB": result["EbNo_dB"],
                "pbit_BER": result["pbit_BER"],
                "pbit_FER": result["pbit_FER"],
                "BP_BER": result["BP_BER"],
                "BP_FER": result["BP_FER"],
                "BP_convergence_rate": result["BP_convergence_rate"],
                "BP_avg_iterations": result["BP_avg_iterations"],
                "kw": kw,
                "kr": kr,
                "alpha": alpha,
                "alpha_mode": alpha_mode,
                "nrnd": nrnd,
                "psa_p": psa_p,
                "lambda_mem": lambda_mem,
                "response_lambda": response_lambda,
                "lambda_out": lambda_out,
                "I0_min": I0_min,
                "I0_max": I0_max,
                "I0_schedule_type": I0_schedule_type,
                "I0_schedule_shape": I0_schedule_shape,
                "I0_hold_fraction": I0_hold_fraction,
                "decision_method": decision_method,
                "burn_in": burn_in,
                "sample_window": sample_window,
            })
    return rows


def save_cycle_sweep_csv(rows, csv_path):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("rows must not be empty")
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def save_cycle_sweep_plot(rows, plot_path):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    plot_path = Path(plot_path)
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    ebno_values = sorted({float(row["EbNo_dB"]) for row in rows})
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), dpi=160)
    for ebno in ebno_values:
        subset = [row for row in rows if float(row["EbNo_dB"]) == ebno]
        subset.sort(key=lambda row: int(row["n_cycles"]))
        x = [int(row["n_cycles"]) for row in subset]
        axes[0].plot(x, [float(row["pbit_BER"]) for row in subset], marker="o", label=f"p-bit {ebno:g} dB")
        axes[0].plot(x, [float(row["BP_BER"]) for row in subset], marker="x", linestyle="--", label=f"BP {ebno:g} dB")
        axes[1].plot(x, [float(row["pbit_FER"]) for row in subset], marker="o", label=f"p-bit {ebno:g} dB")
        axes[1].plot(x, [float(row["BP_FER"]) for row in subset], marker="x", linestyle="--", label=f"BP {ebno:g} dB")

    axes[0].set_title("BER vs n_cycles")
    axes[0].set_xlabel("n_cycles")
    axes[0].set_ylabel("BER")
    axes[0].grid(True, alpha=0.35)
    axes[0].legend(fontsize=8)
    axes[1].set_title("FER vs n_cycles")
    axes[1].set_xlabel("n_cycles")
    axes[1].set_ylabel("FER")
    axes[1].grid(True, alpha=0.35)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)
    return plot_path


SEARCH_DECISION_METHODS = ["last", "majority", "best_state"]
SEARCH_I0_SCHEDULES = ["linear", "constant", "exponential", "cosine", "piecewise"]
SEARCH_ALPHA_MODES = ["fixed", "snr", "llr"]
SEARCH_NUMERIC_RANGES = {
    "kw": (0.25, 8.0),
    "kr": (0.25, 12.0),
    "alpha": (0.2, 5.0),
    "nrnd": (0.0, 2.0),
    "psa_p": (0.0, 1.0),
    "lambda_mem": (0.0, 1.0),
    "response_lambda": (0.0, 0.99),
    "lambda_out": (0.0, 0.2),
    "I0_min": (0.05, 2.0),
    "I0_max": (0.05, 10.0),
    "I0_schedule_shape": (0.25, 8.0),
    "I0_hold_fraction": (0.0, 0.8),
}
SEARCH_PARAM_FIELDS = [
    "kw",
    "kr",
    "alpha",
    "alpha_mode",
    "nrnd",
    "psa_p",
    "lambda_mem",
    "response_lambda",
    "lambda_out",
    "I0_min",
    "I0_max",
    "I0_schedule_type",
    "I0_schedule_shape",
    "I0_hold_fraction",
    "decision_method",
    "burn_in",
    "sample_window",
]


def clamp(value, low, high):
    return min(max(value, low), high)


def parse_float(value, default):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_int(value, default):
    try:
        if value is None or value == "":
            return default
        return int(round(float(value)))
    except (TypeError, ValueError):
        return default


def normalize_search_params(params, base_params, mode, n_cycles):
    merged = dict(base_params)
    merged.update({
        key: value
        for key, value in params.items()
        if key in SEARCH_PARAM_FIELDS
    })

    normalized = {}
    for key in ["kw", "kr", "alpha", "I0_min"]:
        low, high = SEARCH_NUMERIC_RANGES[key]
        normalized[key] = clamp(
            parse_float(merged.get(key), base_params[key]),
            low,
            high,
        )

    alpha_mode = str(merged.get("alpha_mode", base_params.get("alpha_mode", "fixed")))
    if alpha_mode not in SEARCH_ALPHA_MODES:
        alpha_mode = base_params.get("alpha_mode", "fixed")
    if alpha_mode not in SEARCH_ALPHA_MODES:
        alpha_mode = "fixed"
    normalized["alpha_mode"] = alpha_mode

    if is_ssa_mode(mode):
        low, high = SEARCH_NUMERIC_RANGES["nrnd"]
        normalized["nrnd"] = clamp(
            parse_float(merged.get("nrnd"), base_params["nrnd"]),
            low,
            high,
        )
    else:
        normalized["nrnd"] = 0.0

    if is_psa_mode(mode):
        low, high = SEARCH_NUMERIC_RANGES["psa_p"]
        normalized["psa_p"] = clamp(
            parse_float(merged.get("psa_p"), base_params.get("psa_p", 0.0)),
            low,
            high,
        )
    else:
        normalized["psa_p"] = parse_float(base_params.get("psa_p"), 0.0)

    if mode == "lambda_pSA":
        normalized["lambda_mem"] = clamp(
            parse_float(merged.get("lambda_mem"), base_params.get("lambda_mem", 0.0)),
            0.0,
            0.95,
        )
    elif mode == "lambda_SSA":
        normalized["lambda_mem"] = clamp(
            parse_float(merged.get("lambda_mem"), base_params.get("lambda_mem", 1.0)),
            0.0,
            1.0,
        )
    elif mode == "dual_memory":
        normalized["lambda_mem"] = clamp(
            parse_float(merged.get("lambda_mem"), base_params.get("lambda_mem", 0.95)),
            0.0,
            1.0,
        )
    else:
        normalized["lambda_mem"] = parse_float(base_params.get("lambda_mem"), 0.0)

    if mode == "tau_pSA":
        normalized["response_lambda"] = clamp(
            parse_float(merged.get("response_lambda"), base_params.get("response_lambda", 0.0)),
            *SEARCH_NUMERIC_RANGES["response_lambda"],
        )
    else:
        normalized["response_lambda"] = parse_float(base_params.get("response_lambda"), 0.0)

    if mode == "dual_memory":
        normalized["lambda_out"] = clamp(
            parse_float(merged.get("lambda_out"), base_params.get("lambda_out", 0.02)),
            *SEARCH_NUMERIC_RANGES["lambda_out"],
        )
    else:
        normalized["lambda_out"] = parse_float(base_params.get("lambda_out"), 0.0)

    I0_low, I0_high = SEARCH_NUMERIC_RANGES["I0_max"]
    I0_max = clamp(
        parse_float(merged.get("I0_max"), base_params["I0_max"]),
        I0_low,
        I0_high,
    )
    normalized["I0_max"] = max(normalized["I0_min"], I0_max)

    schedule_type = str(
        merged.get("I0_schedule_type", base_params.get("I0_schedule_type", "linear"))
    )
    if schedule_type not in SEARCH_I0_SCHEDULES:
        schedule_type = base_params.get("I0_schedule_type", "linear")
    if schedule_type not in SEARCH_I0_SCHEDULES:
        schedule_type = "linear"
    normalized["I0_schedule_type"] = schedule_type

    for key in ["I0_schedule_shape", "I0_hold_fraction"]:
        low, high = SEARCH_NUMERIC_RANGES[key]
        normalized[key] = clamp(
            parse_float(merged.get(key), base_params.get(key, 1.0 if key.endswith("shape") else 0.0)),
            low,
            high,
        )

    decision_method = str(merged.get("decision_method", base_params["decision_method"]))
    if decision_method not in SEARCH_DECISION_METHODS:
        decision_method = base_params["decision_method"]
    if decision_method not in SEARCH_DECISION_METHODS:
        decision_method = "best_state"
    normalized["decision_method"] = decision_method

    max_burn_in = max(0, n_cycles - 1)
    normalized["burn_in"] = clamp(
        parse_int(merged.get("burn_in"), base_params["burn_in"]),
        0,
        max_burn_in,
    )

    max_window = max(0, n_cycles - normalized["burn_in"])
    normalized["sample_window"] = clamp(
        parse_int(merged.get("sample_window"), base_params["sample_window"]),
        0,
        max_window,
    )

    return normalized


def random_search_params(rng, base_params, n_cycles, mode):
    I0_min = rng.uniform(*SEARCH_NUMERIC_RANGES["I0_min"])
    burn_in = int(rng.integers(0, max(1, n_cycles)))
    max_window = max(1, n_cycles - burn_in)
    sample_window = int(rng.integers(1, max_window + 1))

    if rng.random() < 0.25:
        sample_window = 0

    params = {
        "kw": rng.uniform(*SEARCH_NUMERIC_RANGES["kw"]),
        "kr": rng.uniform(*SEARCH_NUMERIC_RANGES["kr"]),
        "alpha": rng.uniform(*SEARCH_NUMERIC_RANGES["alpha"]),
        "alpha_mode": rng.choice(SEARCH_ALPHA_MODES),
        "nrnd": rng.uniform(*SEARCH_NUMERIC_RANGES["nrnd"]) if is_ssa_mode(mode) else 0.0,
        "psa_p": (
            rng.uniform(*SEARCH_NUMERIC_RANGES["psa_p"])
            if is_psa_mode(mode)
            else base_params["psa_p"]
        ),
        "lambda_mem": (
            rng.uniform(0.0, 0.95)
            if mode == "lambda_pSA"
            else rng.uniform(0.0, 1.0)
            if mode in {"lambda_SSA", "dual_memory"}
            else base_params.get("lambda_mem", 0.0)
        ),
        "response_lambda": (
            rng.uniform(*SEARCH_NUMERIC_RANGES["response_lambda"])
            if mode == "tau_pSA"
            else base_params.get("response_lambda", 0.0)
        ),
        "lambda_out": (
            rng.uniform(*SEARCH_NUMERIC_RANGES["lambda_out"])
            if mode == "dual_memory"
            else base_params.get("lambda_out", 0.0)
        ),
        "I0_min": I0_min,
        "I0_max": rng.uniform(I0_min, SEARCH_NUMERIC_RANGES["I0_max"][1]),
        "I0_schedule_type": rng.choice(SEARCH_I0_SCHEDULES),
        "I0_schedule_shape": rng.uniform(*SEARCH_NUMERIC_RANGES["I0_schedule_shape"]),
        "I0_hold_fraction": rng.uniform(*SEARCH_NUMERIC_RANGES["I0_hold_fraction"]),
        "decision_method": rng.choice(SEARCH_DECISION_METHODS),
        "burn_in": burn_in,
        "sample_window": sample_window,
    }
    return normalize_search_params(params, base_params, mode, n_cycles)


def load_warm_start_params(csv_path, base_params, mode, n_cycles, top_k):
    if not csv_path or top_k <= 0:
        return []

    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"warm-start CSV not found: {csv_path}")

    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        return []

    same_mode_rows = [
        row for row in rows
        if not row.get("mode") or row.get("mode") == mode
    ]
    if same_mode_rows:
        rows = same_mode_rows

    def row_sort_key(row):
        rank = parse_int(row.get("rank"), 10**9)
        ber = parse_float(row.get("mean_pbit_BER"), float("inf"))
        fer = parse_float(row.get("mean_pbit_FER"), float("inf"))
        return rank, ber, fer

    rows = sorted(rows, key=row_sort_key)
    warm_params = []

    for row in rows[:top_k]:
        warm_params.append(normalize_search_params(row, base_params, mode, n_cycles))

    return warm_params


def perturb_positive_param(rng, value, low, high, local_scale):
    if high <= low:
        return low
    if value <= 0.0 or low == 0.0:
        span = high - low
        return clamp(value + rng.normal(0.0, local_scale * span), low, high)
    return clamp(value * np.exp(rng.normal(0.0, local_scale)), low, high)


def perturb_search_params(rng, params, base_params, mode, n_cycles, local_scale):
    perturbed = dict(params)

    for key in ["kw", "kr", "alpha", "I0_min"]:
        low, high = SEARCH_NUMERIC_RANGES[key]
        perturbed[key] = perturb_positive_param(
            rng,
            float(params[key]),
            low,
            high,
            local_scale,
        )

    if rng.random() < 0.15:
        perturbed["alpha_mode"] = rng.choice(SEARCH_ALPHA_MODES)

    if is_ssa_mode(mode):
        low, high = SEARCH_NUMERIC_RANGES["nrnd"]
        perturbed["nrnd"] = perturb_positive_param(
            rng,
            float(params["nrnd"]),
            low,
            high,
            local_scale,
        )
    else:
        perturbed["nrnd"] = 0.0

    if is_psa_mode(mode):
        low, high = SEARCH_NUMERIC_RANGES["psa_p"]
        if rng.random() < 0.35:
            perturbed["psa_p"] = rng.uniform(low, high)
        else:
            perturbed["psa_p"] = clamp(
                float(params["psa_p"]) + rng.normal(0.0, local_scale),
                low,
                high,
            )
    else:
        perturbed["psa_p"] = base_params["psa_p"]

    if mode in {"lambda_pSA", "lambda_SSA", "dual_memory"}:
        low = 0.0
        high = 0.95 if mode == "lambda_pSA" else 1.0
        if rng.random() < 0.35:
            perturbed["lambda_mem"] = rng.uniform(low, high)
        else:
            perturbed["lambda_mem"] = clamp(
                float(params["lambda_mem"]) + rng.normal(0.0, local_scale),
                low,
                high,
            )
    else:
        perturbed["lambda_mem"] = base_params.get("lambda_mem", 0.0)

    if mode == "tau_pSA":
        low, high = SEARCH_NUMERIC_RANGES["response_lambda"]
        if rng.random() < 0.35:
            perturbed["response_lambda"] = rng.uniform(low, high)
        else:
            perturbed["response_lambda"] = clamp(
                float(params.get("response_lambda", 0.0)) + rng.normal(0.0, local_scale),
                low,
                high,
            )
    else:
        perturbed["response_lambda"] = base_params.get("response_lambda", 0.0)

    if mode == "dual_memory":
        low, high = SEARCH_NUMERIC_RANGES["lambda_out"]
        if rng.random() < 0.35:
            perturbed["lambda_out"] = rng.uniform(low, high)
        else:
            perturbed["lambda_out"] = clamp(
                float(params["lambda_out"]) + rng.normal(0.0, local_scale * (high - low)),
                low,
                high,
            )
    else:
        perturbed["lambda_out"] = base_params.get("lambda_out", 0.0)

    i0_min = perturbed["I0_min"]
    perturbed["I0_max"] = perturb_positive_param(
        rng,
        float(params["I0_max"]),
        max(i0_min, SEARCH_NUMERIC_RANGES["I0_max"][0]),
        SEARCH_NUMERIC_RANGES["I0_max"][1],
        local_scale,
    )

    if rng.random() < 0.25:
        perturbed["I0_schedule_type"] = rng.choice(SEARCH_I0_SCHEDULES)

    for key in ["I0_schedule_shape", "I0_hold_fraction"]:
        low, high = SEARCH_NUMERIC_RANGES[key]
        perturbed[key] = perturb_positive_param(
            rng,
            float(params[key]),
            low,
            high,
            local_scale,
        )

    if rng.random() < 0.15:
        perturbed["decision_method"] = rng.choice(SEARCH_DECISION_METHODS)

    burn_jitter = int(round(rng.normal(0.0, max(1.0, local_scale * n_cycles))))
    perturbed["burn_in"] = int(params["burn_in"]) + burn_jitter

    max_window = max(0, n_cycles - int(params["burn_in"]))
    window_scale = max(1.0, local_scale * max(1, max_window))
    if int(params["sample_window"]) == 0 and rng.random() < 0.5:
        perturbed["sample_window"] = 0
    else:
        window_jitter = int(round(rng.normal(0.0, window_scale)))
        perturbed["sample_window"] = int(params["sample_window"]) + window_jitter

    return normalize_search_params(perturbed, base_params, mode, n_cycles)


def build_search_candidates(
    base_params,
    n_candidates,
    search_seed,
    n_cycles,
    mode,
    warm_start_csv=None,
    warm_start_top_k=0,
    local_samples_per_seed=0,
    local_scale=0.25,
):
    rng = np.random.default_rng(search_seed)
    candidates = [normalize_search_params(base_params, base_params, mode, n_cycles)]

    warm_params = load_warm_start_params(
        warm_start_csv,
        base_params,
        mode,
        n_cycles,
        warm_start_top_k,
    )
    candidates.extend(warm_params)

    local_scale = max(0.0, float(local_scale))
    local_samples_per_seed = max(0, int(local_samples_per_seed))
    for warm_param in warm_params:
        for _ in range(local_samples_per_seed):
            candidates.append(
                perturb_search_params(
                    rng,
                    warm_param,
                    base_params,
                    mode,
                    n_cycles,
                    local_scale,
                )
            )

    while len(candidates) < n_candidates:
        candidates.append(random_search_params(rng, base_params, n_cycles, mode))

    candidates = candidates[:n_candidates]
    return [
        {"candidate_id": idx, **params}
        for idx, params in enumerate(candidates, start=1)
    ]


def average_result_sets(result_sets):
    if not result_sets:
        return []

    n_ebno = len(result_sets[0])
    averaged = []
    numeric_fields = [
        "BER",
        "FER",
        "pbit_BER",
        "pbit_FER",
        "BP_BER",
        "BP_FER",
        "BP_convergence_rate",
        "BP_avg_iterations",
    ]

    for idx in range(n_ebno):
        ebno = result_sets[0][idx]["EbNo_dB"]
        row = {"EbNo_dB": ebno}
        for field in numeric_fields:
            row[field] = float(np.mean([results[idx][field] for results in result_sets]))
        averaged.append(row)

    return averaged


def evaluate_search_candidate(task):
    params = task["params"]
    seeds = task.get("seeds")
    if seeds is None:
        seeds = [task["seed"]]

    result_sets = []
    for seed in seeds:
        _, results = simulate_ldpc_decoder(
            P=task["P"],
            fixed_bit_width=task["fixed_bit_width"],
            channel_input_mode=task["channel_input_mode"],
            alpha_mode=params["alpha_mode"],
            ebno_db_list=task["ebno_db_list"],
            alpha=params["alpha"],
            kw=params["kw"],
            kr=params["kr"],
            n_cycles=task["n_cycles"],
            n_trials=task["n_trials"],
            mode=task["mode"],
            I0_min=params["I0_min"],
            I0_max=params["I0_max"],
            nrnd=params["nrnd"],
            psa_p=params["psa_p"],
            lambda_mem=params["lambda_mem"],
            response_lambda=params.get("response_lambda", 0.0),
            lambda_out=params.get("lambda_out", 0.0),
            I0_schedule_type=params["I0_schedule_type"],
            I0_schedule_shape=params["I0_schedule_shape"],
            I0_hold_fraction=params["I0_hold_fraction"],
            decision_method=params["decision_method"],
            burn_in=params["burn_in"],
            sample_window=params["sample_window"],
            use_all_zero_message=task["use_all_zero_message"],
            bp_max_iter=task["bp_max_iter"],
            bp_llr_clip=task["bp_llr_clip"],
            seed=seed,
            verbose=False,
        )
        result_sets.append(results)

    results = average_result_sets(result_sets)
    return {
        "mode": task["mode"],
        "search_algorithm": task.get("search_algorithm", "random"),
        "evaluation_trials": task["n_trials"] * len(seeds),
        "evaluation_seeds": format_int_list(seeds),
        **params,
        **summarize_results(results),
        "by_ebno_results": results,
    }


def rank_search_records(records):
    for rank, record in enumerate(records, start=1):
        record["rank"] = rank
    return records


def sort_search_records(records):
    sorted_records = sorted(
        records,
        key=lambda record: (record["mean_pbit_BER"], record["mean_pbit_FER"]),
    )
    return rank_search_records(sorted_records)


def random_parameter_search(
    P,
    ebno_db_list,
    base_params,
    mode="SSA",
    fixed_bit_width=8,
    channel_input_mode="float",
    n_cycles=100,
    n_trials=100,
    n_candidates=50,
    seed=0,
    search_seed=1,
    I0_schedule_type="linear",
    use_all_zero_message=False,
    bp_max_iter=50,
    bp_llr_clip=50.0,
    n_workers=1,
    warm_start_csv=None,
    warm_start_top_k=0,
    local_samples_per_seed=0,
    local_scale=0.25,
    refine_top_k=0,
    refine_trials=0,
    refine_seeds=None,
    verbose=True,
):
    """
    pSA/SSA用の簡易ランダム探索。
    同じseedで全候補を評価するので、候補間でチャネル条件をそろえて比較できる。
    """
    if n_workers < 1:
        raise ValueError("n_workers must be >= 1")
    if n_candidates < 1:
        raise ValueError("n_candidates must be >= 1")

    candidates = build_search_candidates(
        base_params,
        n_candidates,
        search_seed,
        n_cycles,
        mode,
        warm_start_csv=warm_start_csv,
        warm_start_top_k=warm_start_top_k,
        local_samples_per_seed=local_samples_per_seed,
        local_scale=local_scale,
    )
    tasks = [
        {
            "P": P,
            "mode": mode,
            "ebno_db_list": ebno_db_list,
            "params": candidate,
            "fixed_bit_width": fixed_bit_width,
            "channel_input_mode": channel_input_mode,
            "n_cycles": n_cycles,
            "n_trials": n_trials,
            "seed": seed,
            "I0_schedule_type": I0_schedule_type,
            "use_all_zero_message": use_all_zero_message,
            "bp_max_iter": bp_max_iter,
            "bp_llr_clip": bp_llr_clip,
        }
        for candidate in candidates
    ]

    records = run_search_tasks(tasks, n_workers, verbose=verbose)
    sorted_records = sort_search_records(records)

    if refine_seeds is None:
        refine_seeds = [seed]
    refine_seeds = [int(value) for value in refine_seeds]
    if refine_trials <= 0:
        refine_trials = n_trials
    do_refine = (
        refine_top_k > 0
        and (
            refine_trials > n_trials
            or len(refine_seeds) > 1
            or (len(refine_seeds) == 1 and refine_seeds[0] != seed)
        )
    )

    if not do_refine:
        return sorted_records

    refine_top_k = min(int(refine_top_k), len(sorted_records))
    selected_ids = {
        record["candidate_id"]
        for record in sorted_records[:refine_top_k]
    }
    candidate_by_id = {
        candidate["candidate_id"]: candidate
        for candidate in candidates
    }
    refine_tasks = [
        {
            **tasks[0],
            "params": candidate_by_id[candidate_id],
            "n_trials": refine_trials,
            "seeds": refine_seeds,
        }
        for candidate_id in selected_ids
    ]

    if verbose:
        print(
            f"\nRefining top {refine_top_k} candidates "
            f"with {refine_trials} trials x {len(refine_seeds)} seed(s)."
        )

    refined_records = run_search_tasks(
        refine_tasks,
        n_workers,
        verbose=verbose,
    )
    sorted_refined_records = sort_search_records(refined_records)
    coarse_tail = [
        record for record in sorted_records
        if record["candidate_id"] not in selected_ids
    ]

    return rank_search_records(sorted_refined_records + coarse_tail)


def suggest_optuna_params(trial, base_params, mode, n_cycles):
    i0_min = trial.suggest_float(
        "I0_min",
        *SEARCH_NUMERIC_RANGES["I0_min"],
        log=True,
    )
    burn_in = trial.suggest_int("burn_in", 0, max(0, n_cycles - 1))
    max_window = max(0, n_cycles - burn_in)

    params = {
        "kw": trial.suggest_float("kw", *SEARCH_NUMERIC_RANGES["kw"], log=True),
        "kr": trial.suggest_float("kr", *SEARCH_NUMERIC_RANGES["kr"], log=True),
        "alpha": trial.suggest_float("alpha", *SEARCH_NUMERIC_RANGES["alpha"], log=True),
        "alpha_mode": trial.suggest_categorical("alpha_mode", SEARCH_ALPHA_MODES),
        "nrnd": (
            trial.suggest_float("nrnd", *SEARCH_NUMERIC_RANGES["nrnd"])
            if is_ssa_mode(mode)
            else 0.0
        ),
        "psa_p": (
            trial.suggest_float("psa_p", *SEARCH_NUMERIC_RANGES["psa_p"])
            if is_psa_mode(mode)
            else base_params["psa_p"]
        ),
        "lambda_mem": (
            trial.suggest_float("lambda_mem", 0.0, 0.95)
            if mode == "lambda_pSA"
            else trial.suggest_float("lambda_mem", 0.0, 1.0)
            if mode in {"lambda_SSA", "dual_memory"}
            else base_params.get("lambda_mem", 0.0)
        ),
        "response_lambda": (
            trial.suggest_float("response_lambda", *SEARCH_NUMERIC_RANGES["response_lambda"])
            if mode == "tau_pSA"
            else base_params.get("response_lambda", 0.0)
        ),
        "lambda_out": (
            trial.suggest_float("lambda_out", *SEARCH_NUMERIC_RANGES["lambda_out"])
            if mode == "dual_memory"
            else base_params.get("lambda_out", 0.0)
        ),
        "I0_min": i0_min,
        "I0_max": trial.suggest_float(
            "I0_max",
            max(i0_min, SEARCH_NUMERIC_RANGES["I0_max"][0]),
            SEARCH_NUMERIC_RANGES["I0_max"][1],
            log=True,
        ),
        "I0_schedule_type": trial.suggest_categorical(
            "I0_schedule_type",
            SEARCH_I0_SCHEDULES,
        ),
        "I0_schedule_shape": trial.suggest_float(
            "I0_schedule_shape",
            *SEARCH_NUMERIC_RANGES["I0_schedule_shape"],
            log=True,
        ),
        "I0_hold_fraction": trial.suggest_float(
            "I0_hold_fraction",
            *SEARCH_NUMERIC_RANGES["I0_hold_fraction"],
        ),
        "decision_method": trial.suggest_categorical(
            "decision_method",
            SEARCH_DECISION_METHODS,
        ),
        "burn_in": burn_in,
        "sample_window": trial.suggest_int("sample_window", 0, max_window),
    }
    return normalize_search_params(params, base_params, mode, n_cycles)


def optuna_parameter_search(
    P,
    ebno_db_list,
    base_params,
    mode="SSA",
    fixed_bit_width=8,
    channel_input_mode="float",
    n_cycles=100,
    n_trials=100,
    n_candidates=50,
    seed=0,
    search_seed=1,
    use_all_zero_message=False,
    bp_max_iter=50,
    bp_llr_clip=50.0,
    n_workers=1,
    warm_start_csv=None,
    warm_start_top_k=0,
    refine_top_k=0,
    refine_trials=0,
    refine_seeds=None,
    storage=None,
    study_name=None,
    verbose=True,
):
    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError(
            "Optuna is not installed. Install it with './m2max/bin/python -m pip install optuna'."
        ) from exc

    sampler = optuna.samplers.TPESampler(seed=search_seed)
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        storage=storage,
        study_name=study_name,
        load_if_exists=bool(storage and study_name),
    )

    warm_params = [normalize_search_params(base_params, base_params, mode, n_cycles)]
    warm_params.extend(
        load_warm_start_params(
            warm_start_csv,
            base_params,
            mode,
            n_cycles,
            warm_start_top_k,
        )
    )
    for params in warm_params:
        study.enqueue_trial({
            key: value
            for key, value in params.items()
            if key in SEARCH_PARAM_FIELDS
        })

    def objective(trial):
        params = suggest_optuna_params(trial, base_params, mode, n_cycles)
        params["candidate_id"] = trial.number + 1
        task = {
            "P": P,
            "mode": mode,
            "ebno_db_list": ebno_db_list,
            "params": params,
            "fixed_bit_width": fixed_bit_width,
            "channel_input_mode": channel_input_mode,
            "n_cycles": n_cycles,
            "n_trials": n_trials,
            "seed": seed,
            "use_all_zero_message": use_all_zero_message,
            "bp_max_iter": bp_max_iter,
            "bp_llr_clip": bp_llr_clip,
        }
        record = evaluate_search_candidate(task)
        record["search_algorithm"] = "optuna_tpe"
        trial.set_user_attr("record", record)
        if verbose:
            print_search_progress(trial.number + 1, n_candidates, record)
        return record["mean_pbit_BER"]

    study.optimize(
        objective,
        n_trials=n_candidates,
        n_jobs=max(1, int(n_workers)),
        show_progress_bar=False,
    )

    records = []
    for trial in study.trials:
        record = trial.user_attrs.get("record")
        if record is not None:
            records.append(record)

    sorted_records = sort_search_records(records)
    if refine_seeds is None:
        refine_seeds = [seed]
    refine_seeds = [int(value) for value in refine_seeds]
    if refine_trials <= 0:
        refine_trials = n_trials

    do_refine = (
        refine_top_k > 0
        and (
            refine_trials > n_trials
            or len(refine_seeds) > 1
            or (len(refine_seeds) == 1 and refine_seeds[0] != seed)
        )
    )
    if not do_refine:
        return sorted_records

    refine_top_k = min(int(refine_top_k), len(sorted_records))
    if verbose:
        print(
            f"\nRefining Optuna top {refine_top_k} candidates "
            f"with {refine_trials} trials x {len(refine_seeds)} seed(s)."
        )

    refine_tasks = []
    for record in sorted_records[:refine_top_k]:
        params = {
            key: record[key]
            for key in SEARCH_PARAM_FIELDS
        }
        params["candidate_id"] = record["candidate_id"]
        refine_tasks.append({
            "P": P,
            "mode": mode,
            "ebno_db_list": ebno_db_list,
            "params": params,
            "fixed_bit_width": fixed_bit_width,
            "channel_input_mode": channel_input_mode,
            "n_cycles": n_cycles,
            "n_trials": refine_trials,
            "seed": seed,
            "seeds": refine_seeds,
            "use_all_zero_message": use_all_zero_message,
            "bp_max_iter": bp_max_iter,
            "bp_llr_clip": bp_llr_clip,
        })

    refined_records = run_search_tasks(refine_tasks, n_workers, verbose=verbose)
    for record in refined_records:
        record["search_algorithm"] = "optuna_tpe_refined"

    selected_ids = {record["candidate_id"] for record in refined_records}
    coarse_tail = [
        record for record in sorted_records
        if record["candidate_id"] not in selected_ids
    ]
    return rank_search_records(sort_search_records(refined_records) + coarse_tail)


def run_search_tasks(tasks, n_workers, verbose=True):
    records = []

    if n_workers == 1:
        for done, task in enumerate(tasks, start=1):
            record = evaluate_search_candidate(task)
            records.append(record)

            if verbose:
                print_search_progress(done, len(tasks), record)
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(evaluate_search_candidate, task) for task in tasks]

            for done, future in enumerate(as_completed(futures), start=1):
                record = future.result()
                records.append(record)

                if verbose:
                    print_search_progress(done, len(tasks), record)

    return records


def print_search_progress(done, total, record):
    print(
        f"[{done:03d}/{total:03d}] "
        f"id={record['candidate_id']:03d}, "
        f"trials={record['evaluation_trials']}, "
        f"BER={record['mean_pbit_BER']:.6f}, "
        f"FER={record['mean_pbit_FER']:.6f}, "
        f"kw={record['kw']:.4g}, kr={record['kr']:.4g}, "
        f"alpha={record['alpha']:.4g}, alpha_mode={record['alpha_mode']}, "
        f"nrnd={record['nrnd']:.4g}, "
        f"psa_p={record['psa_p']:.4g}, "
        f"lambda_mem={record['lambda_mem']:.4g}, "
        f"response_lambda={record.get('response_lambda', 0.0):.4g}, "
        f"lambda_out={record.get('lambda_out', 0.0):.4g}, "
        f"I0_min={record['I0_min']:.4g}, I0_max={record['I0_max']:.4g}, "
        f"I0_schedule={record['I0_schedule_type']}, "
        f"decision={record['decision_method']}, "
        f"burn_in={record['burn_in']}, window={record['sample_window']}",
        flush=True,
    )


def save_search_records_csv(records, csv_path):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "rank",
        "candidate_id",
        "mode",
        "search_algorithm",
        "evaluation_trials",
        "evaluation_seeds",
        "mean_pbit_BER",
        "mean_pbit_FER",
        "mean_BP_BER",
        "mean_BP_FER",
        "EbNo_dB_values",
        "pbit_BER_by_EbNo",
        "pbit_FER_by_EbNo",
        "BP_BER_by_EbNo",
        "BP_FER_by_EbNo",
        "kw",
        "kr",
        "alpha",
        "alpha_mode",
        "nrnd",
        "psa_p",
        "lambda_mem",
        "response_lambda",
        "lambda_out",
        "I0_min",
        "I0_max",
        "I0_schedule_type",
        "I0_schedule_shape",
        "I0_hold_fraction",
        "decision_method",
        "burn_in",
        "sample_window",
    ]

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for record in records:
            writer.writerow({key: record.get(key, "") for key in fieldnames})

    return csv_path


def save_search_ebno_records_csv(records, csv_path):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "rank",
        "candidate_id",
        "mode",
        "search_algorithm",
        "evaluation_trials",
        "evaluation_seeds",
        "EbNo_dB",
        "pbit_BER",
        "pbit_FER",
        "BP_BER",
        "BP_FER",
        "BP_convergence_rate",
        "BP_avg_iterations",
        "kw",
        "kr",
        "alpha",
        "alpha_mode",
        "nrnd",
        "psa_p",
        "lambda_mem",
        "response_lambda",
        "lambda_out",
        "I0_min",
        "I0_max",
        "I0_schedule_type",
        "I0_schedule_shape",
        "I0_hold_fraction",
        "decision_method",
        "burn_in",
        "sample_window",
    ]

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for record in records:
            for result in record.get("by_ebno_results", []):
                writer.writerow({
                    "rank": record["rank"],
                    "candidate_id": record["candidate_id"],
                    "mode": record["mode"],
                    "search_algorithm": record.get("search_algorithm", "random"),
                    "evaluation_trials": record["evaluation_trials"],
                    "evaluation_seeds": record.get("evaluation_seeds", ""),
                    "EbNo_dB": result["EbNo_dB"],
                    "pbit_BER": result["pbit_BER"],
                    "pbit_FER": result["pbit_FER"],
                    "BP_BER": result["BP_BER"],
                    "BP_FER": result["BP_FER"],
                    "BP_convergence_rate": result["BP_convergence_rate"],
                    "BP_avg_iterations": result["BP_avg_iterations"],
                    "kw": record["kw"],
                    "kr": record["kr"],
                    "alpha": record["alpha"],
                    "alpha_mode": record["alpha_mode"],
                    "nrnd": record["nrnd"],
                    "psa_p": record["psa_p"],
                    "lambda_mem": record["lambda_mem"],
                    "response_lambda": record.get("response_lambda", 0.0),
                    "lambda_out": record.get("lambda_out", 0.0),
                    "I0_min": record["I0_min"],
                    "I0_max": record["I0_max"],
                    "I0_schedule_type": record["I0_schedule_type"],
                    "I0_schedule_shape": record["I0_schedule_shape"],
                    "I0_hold_fraction": record["I0_hold_fraction"],
                    "decision_method": record["decision_method"],
                    "burn_in": record["burn_in"],
                    "sample_window": record["sample_window"],
                })

    return csv_path


def save_search_plot(records, plot_path):
    try:
        return save_search_plot_matplotlib(records, plot_path)
    except ImportError:
        fallback_path = Path(plot_path).with_suffix(".svg")
        return save_search_plot_svg(records, fallback_path)


def memory_text_for_record(record):
    if record.get("mode") == "tau_pSA":
        return f"response_lambda={record.get('response_lambda', 0.0):.3g}"
    return f"lambda_mem={record.get('lambda_mem', 0.0):.3g}"


def save_search_plot_matplotlib(records, plot_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_path = Path(plot_path)
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    if not records:
        raise ValueError("records must not be empty")

    ranks = np.array([record["rank"] for record in records], dtype=int)
    ber_values = np.array([record["mean_pbit_BER"] for record in records], dtype=float)
    bp_mean_ber = float(np.mean([record["mean_BP_BER"] for record in records]))
    best = records[0]
    mode_label = best.get("mode", "p-bit")
    memory_text = memory_text_for_record(best)

    fig, ax = plt.subplots(figsize=(10, 5.8), dpi=160)
    ax.plot(ranks, ber_values, marker="o", markersize=3.5, linewidth=1.4, label=f"{mode_label} mean BER")
    ax.scatter([1], [best["mean_pbit_BER"]], color="tab:red", s=48, zorder=4, label="Best")
    ax.axhline(
        bp_mean_ber,
        color="tab:red",
        linestyle="--",
        linewidth=1.2,
        label=f"BP mean BER = {bp_mean_ber:.3g}",
    )

    ax.set_title(f"{mode_label} Parameter Search")
    ax.set_xlabel("Candidate rank (sorted by mean BER)")
    ax.set_ylabel("Mean BER")
    ax.grid(True, which="both", linewidth=0.5, alpha=0.35)
    ax.set_xlim(0.5, max(1.5, len(records) + 0.5))
    ax.set_ylim(bottom=0.0)
    ax.legend(loc="best")

    best_text = (
        f"Best: BER={best['mean_pbit_BER']:.3g}, FER={best['mean_pbit_FER']:.3g}, "
        f"kw={best['kw']:.3g}, kr={best['kr']:.3g}, alpha={best['alpha']:.3g}, "
        f"alpha_mode={best['alpha_mode']}, nrnd={best['nrnd']:.3g}, "
        f"psa_p={best['psa_p']:.3g}, {memory_text}, "
        f"I0=({best['I0_min']:.3g}, {best['I0_max']:.3g}), "
        f"sched={best['I0_schedule_type']}, "
        f"{best['decision_method']}, burn-in={best['burn_in']}, window={best['sample_window']}"
    )
    fig.text(0.5, 0.01, best_text, ha="center", va="bottom", fontsize=8.5)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)

    return plot_path


def xml_escape(value):
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def save_search_plot_svg(records, svg_path):
    svg_path = Path(svg_path)
    svg_path.parent.mkdir(parents=True, exist_ok=True)

    width = 960
    height = 540
    margin_left = 80
    margin_right = 40
    margin_top = 70
    margin_bottom = 90
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    if not records:
        raise ValueError("records must not be empty")

    ranks = np.array([record["rank"] for record in records], dtype=float)
    ber_values = np.array([record["mean_pbit_BER"] for record in records], dtype=float)
    bp_values = np.array([record["mean_BP_BER"] for record in records], dtype=float)

    y_max = float(max(np.max(ber_values), np.max(bp_values), 1e-12) * 1.1)
    y_min = 0.0
    x_min = 1.0
    x_max = float(max(len(records), 1))

    def x_map(x):
        if x_max == x_min:
            return margin_left + plot_width / 2
        return margin_left + (x - x_min) / (x_max - x_min) * plot_width

    def y_map(y):
        return margin_top + (y_max - y) / (y_max - y_min) * plot_height

    points = " ".join(
        f"{x_map(rank):.2f},{y_map(ber):.2f}"
        for rank, ber in zip(ranks, ber_values)
    )
    best = records[0]
    bp_mean_ber = float(np.mean(bp_values))
    mode_label = best.get("mode", "p-bit")
    memory_text = memory_text_for_record(best)

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2}" y="28" text-anchor="middle" font-family="Arial, sans-serif" font-size="22" font-weight="700" fill="#111827">{xml_escape(mode_label)} Parameter Search</text>',
        f'<text x="{width / 2}" y="52" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" fill="#4b5563">x-axis: candidate rank sorted by mean BER</text>',
        f'<rect x="{margin_left}" y="{margin_top}" width="{plot_width}" height="{plot_height}" fill="#f9fafb" stroke="#d1d5db"/>',
    ]

    for tick in range(6):
        y_value = y_min + (y_max - y_min) * tick / 5
        y = y_map(y_value)
        elements.append(
            f'<line x1="{margin_left}" y1="{y:.2f}" x2="{margin_left + plot_width}" y2="{y:.2f}" stroke="#e5e7eb"/>'
        )
        elements.append(
            f'<text x="{margin_left - 12}" y="{y + 4:.2f}" text-anchor="end" font-family="Arial, sans-serif" font-size="12" fill="#374151">{y_value:.3g}</text>'
        )

    x_tick_count = min(10, len(records))
    if x_tick_count == 1:
        x_ticks = [1]
    else:
        x_ticks = np.linspace(1, len(records), x_tick_count)

    for x_value in x_ticks:
        x = x_map(float(x_value))
        elements.append(
            f'<line x1="{x:.2f}" y1="{margin_top + plot_height}" x2="{x:.2f}" y2="{margin_top + plot_height + 6}" stroke="#6b7280"/>'
        )
        elements.append(
            f'<text x="{x:.2f}" y="{margin_top + plot_height + 24}" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#374151">{int(round(x_value))}</text>'
        )

    bp_y = y_map(bp_mean_ber)
    elements.extend([
        f'<line x1="{margin_left}" y1="{bp_y:.2f}" x2="{margin_left + plot_width}" y2="{bp_y:.2f}" stroke="#dc2626" stroke-width="2" stroke-dasharray="6 5"/>',
        f'<text x="{margin_left + plot_width - 4}" y="{bp_y - 8:.2f}" text-anchor="end" font-family="Arial, sans-serif" font-size="12" fill="#dc2626">BP mean BER = {bp_mean_ber:.3g}</text>',
        f'<polyline points="{points}" fill="none" stroke="#2563eb" stroke-width="2"/>',
    ])

    for rank, ber in zip(ranks, ber_values):
        fill = "#ef4444" if rank == 1 else "#2563eb"
        radius = 5 if rank == 1 else 3
        elements.append(
            f'<circle cx="{x_map(rank):.2f}" cy="{y_map(ber):.2f}" r="{radius}" fill="{fill}" opacity="0.88"/>'
        )

    best_text = (
        f"Best rank 1: BER={best['mean_pbit_BER']:.3g}, FER={best['mean_pbit_FER']:.3g}, "
        f"kw={best['kw']:.3g}, kr={best['kr']:.3g}, alpha={best['alpha']:.3g}, "
        f"alpha_mode={best['alpha_mode']}, nrnd={best['nrnd']:.3g}, "
        f"psa_p={best['psa_p']:.3g}, {memory_text}, "
        f"I0=({best['I0_min']:.3g}, {best['I0_max']:.3g}), "
        f"sched={best['I0_schedule_type']}, "
        f"{best['decision_method']}, burn-in={best['burn_in']}, window={best['sample_window']}"
    )
    elements.extend([
        f'<text x="{margin_left + plot_width / 2}" y="{height - 24}" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" fill="#111827">{xml_escape(best_text)}</text>',
        f'<text x="{margin_left + plot_width / 2}" y="{height - 54}" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" fill="#111827">Candidate rank</text>',
        f'<text transform="translate(24 {margin_top + plot_height / 2}) rotate(-90)" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" fill="#111827">Mean BER</text>',
        "</svg>",
    ])

    svg_path.write_text("\n".join(elements))
    return svg_path


def print_results(results):
    for res in results:
        print(
            f"Eb/No = {res['EbNo_dB']} dB, "
            f"p-bit BER = {res['pbit_BER']:.6f}, "
            f"p-bit FER = {res['pbit_FER']:.6f}, "
            f"BP BER = {res['BP_BER']:.6f}, "
            f"BP FER = {res['BP_FER']:.6f}, "
            f"BP conv = {res['BP_convergence_rate']:.3f}, "
            f"BP avg iter = {res['BP_avg_iterations']:.2f}"
        )


def make_ebno_list(start, stop, step):
    if step <= 0:
        raise ValueError("Eb/No step must be positive")
    return np.arange(start, stop, step)


def parse_int_list_arg(value):
    if value is None or str(value).strip() == "":
        return []
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def make_refine_seeds(seed, seed_count=1, seed_list=None):
    seeds = parse_int_list_arg(seed_list)
    if seeds:
        return seeds
    return [int(seed) + idx for idx in range(max(1, int(seed_count)))]


def default_search_result_path(mode, P, suffix):
    n_checks, n_bits = np.array(P).shape
    return Path("results") / f"{mode}_{n_checks}x{n_bits}_search.{suffix}"


def parse_args():
    parser = argparse.ArgumentParser(
        description="LDPC p-bit decoder simulation with BP baseline."
    )
    parser.add_argument("--mode", choices=PBIT_MODES, default="SSA")
    parser.add_argument(
        "--p-matrix",
        choices=["toy", "hamming74", "hamming1511", "random_regular"],
        default="toy",
        help="Parity-check matrix source. Matrix shape is (n_checks, n_bits).",
    )
    parser.add_argument(
        "--p-matrix-file",
        default=None,
        help="Load parity-check matrix from a whitespace-separated text file.",
    )
    parser.add_argument("--n-bits", type=int, default=96)
    parser.add_argument("--n-checks", type=int, default=48)
    parser.add_argument("--variable-degree", type=int, default=3)
    parser.add_argument(
        "--check-degree",
        type=int,
        default=None,
        help="If omitted for random_regular, computed from n_bits*variable_degree/n_checks.",
    )
    parser.add_argument("--matrix-seed", type=int, default=0)
    parser.add_argument("--print-matrices", action="store_true")
    parser.add_argument("--ebno-start", type=float, default=0.0)
    parser.add_argument("--ebno-stop", type=float, default=8.0)
    parser.add_argument("--ebno-step", type=float, default=1.0)
    parser.add_argument("--n-trials", type=int, default=1000)
    parser.add_argument("--n-cycles", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fixed-bit-width", type=int, default=8)
    parser.add_argument(
        "--channel-input-mode",
        choices=["float", "fixed"],
        default="float",
        help="Use tanh(alpha*y) directly or quantize it before stochastic sampling.",
    )
    parser.add_argument(
        "--alpha-mode",
        choices=SEARCH_ALPHA_MODES,
        default="fixed",
        help="fixed: tanh(alpha*y), snr: tanh(alpha*y/sigma^2), llr: tanh(alpha*LLR).",
    )
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--kw", type=float, default=2.0)
    parser.add_argument("--kr", type=float, default=4.0)
    parser.add_argument(
        "--nrnd",
        type=float,
        default=0.5,
        help="SSA noise strength. Ignored in pSA.",
    )
    parser.add_argument(
        "--psa-p",
        type=float,
        default=0.0,
        help="pSA hold probability: p=0.1 keeps 10% of p-bit outputs unchanged.",
    )
    parser.add_argument(
        "--lambda-mem",
        type=float,
        default=0.0,
        help="Memory coefficient for lambda_pSA/lambda_SSA/dual_memory.",
    )
    parser.add_argument(
        "--response-lambda",
        type=float,
        default=0.0,
        help="Finite-response coefficient for tau_pSA. Must satisfy 0 <= response_lambda < 1.",
    )
    parser.add_argument(
        "--lambda-out",
        type=float,
        default=0.0,
        help="Weak output-memory coefficient for dual_memory.",
    )
    parser.add_argument("--i0-min", type=float, default=0.1)
    parser.add_argument("--i0-max", type=float, default=5.0)
    parser.add_argument(
        "--i0-schedule",
        choices=SEARCH_I0_SCHEDULES,
        default="linear",
    )
    parser.add_argument("--i0-shape", type=float, default=1.0)
    parser.add_argument("--i0-hold-fraction", type=float, default=0.0)
    parser.add_argument("--bp-max-iter", type=int, default=50)
    parser.add_argument("--bp-llr-clip", type=float, default=50.0)
    parser.add_argument(
        "--decision-method",
        choices=["last", "majority", "best_state"],
        default="last",
    )
    parser.add_argument("--burn-in", type=int, default=0)
    parser.add_argument(
        "--sample-window",
        type=int,
        default=0,
        help="0 means all samples after burn-in for majority/best_state.",
    )
    parser.add_argument("--use-all-zero-message", action="store_true")
    parser.add_argument("--search", action="store_true", help="Run pSA/SSA parameter search.")
    parser.add_argument(
        "--search-algorithm",
        choices=["random", "optuna"],
        default="random",
        help="Use random search or Optuna TPE search.",
    )
    parser.add_argument(
        "--search-only",
        action="store_true",
        help="Stop after writing search CSV/plot instead of running full simulation with the best row.",
    )
    parser.add_argument("--search-candidates", type=int, default=50)
    parser.add_argument("--search-trials", type=int, default=200)
    parser.add_argument(
        "--refine-top-k",
        type=int,
        default=0,
        help="Re-evaluate the coarse-search top K candidates with --refine-trials.",
    )
    parser.add_argument(
        "--refine-trials",
        type=int,
        default=0,
        help="Trial count for the second-stage refinement search.",
    )
    parser.add_argument(
        "--refine-seed-count",
        type=int,
        default=1,
        help="Evaluate refined top candidates across seed, seed+1, ...",
    )
    parser.add_argument(
        "--refine-seeds",
        default=None,
        help="Comma-separated seed list for refined top candidates. Overrides --refine-seed-count.",
    )
    parser.add_argument("--search-seed", type=int, default=1)
    parser.add_argument("--n-workers", type=int, default=1)
    parser.add_argument(
        "--warm-start-csv",
        default=None,
        help="Use top results from an existing search CSV before random candidates.",
    )
    parser.add_argument(
        "--warm-start-top-k",
        type=int,
        default=0,
        help="Number of previous best rows to reuse from --warm-start-csv.",
    )
    parser.add_argument(
        "--local-samples-per-seed",
        type=int,
        default=0,
        help="Number of local perturbation candidates generated around each warm-start row.",
    )
    parser.add_argument(
        "--local-scale",
        type=float,
        default=0.25,
        help="Perturbation strength for warm-start local search.",
    )
    parser.add_argument(
        "--search-csv",
        default=None,
        help="If omitted, use results/<mode>_<n_checks>x<n_bits>_search.csv.",
    )
    parser.add_argument(
        "--search-plot",
        default=None,
        help="If omitted, use results/<mode>_<n_checks>x<n_bits>_search.png.",
    )
    parser.add_argument("--optuna-storage", default=None)
    parser.add_argument("--optuna-study-name", default=None)
    parser.add_argument(
        "--results-csv",
        default=None,
        help="Run result CSV. If omitted, use results/<mode>_<n_checks>x<n_bits>_run_by_ebno.csv.",
    )
    parser.add_argument(
        "--results-plot",
        default=None,
        help="Run result plot. If omitted, use results/<mode>_<n_checks>x<n_bits>_run_ber_fer.png.",
    )
    parser.add_argument(
        "--cycle-sweep",
        default=None,
        help="Comma-separated n_cycles list. Runs a cycle sweep instead of one normal run.",
    )
    parser.add_argument("--cycle-sweep-csv", default=None)
    parser.add_argument("--cycle-sweep-plot", default=None)
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    args = parse_args()

    ebno_args_set = any(
        arg in sys.argv[1:]
        for arg in ("--ebno-start", "--ebno-stop", "--ebno-step")
    )
    if args.search and not ebno_args_set:
        args.ebno_start = 4.0
        args.ebno_stop = 5.0
        args.ebno_step = 1.0

    P = make_parity_check_matrix(args)

    ebno_db_list = make_ebno_list(args.ebno_start, args.ebno_stop, args.ebno_step)
    params = {
        "kw": args.kw,
        "kr": args.kr,
        "alpha": args.alpha,
        "alpha_mode": args.alpha_mode,
        "nrnd": args.nrnd,
        "psa_p": args.psa_p,
        "lambda_mem": args.lambda_mem,
        "response_lambda": args.response_lambda,
        "lambda_out": args.lambda_out,
        "I0_min": args.i0_min,
        "I0_max": args.i0_max,
        "I0_schedule_type": args.i0_schedule,
        "I0_schedule_shape": args.i0_shape,
        "I0_hold_fraction": args.i0_hold_fraction,
        "decision_method": args.decision_method,
        "burn_in": args.burn_in,
        "sample_window": args.sample_window,
    }
    if is_psa_mode(args.mode):
        params["nrnd"] = 0.0
    if not is_psa_mode(args.mode):
        params["psa_p"] = 0.0
    if not is_lambda_mode(args.mode):
        params["lambda_mem"] = 0.0
        params["lambda_out"] = 0.0
    if args.mode != "tau_pSA":
        params["response_lambda"] = 0.0
    if args.mode != "dual_memory":
        params["lambda_out"] = 0.0

    if args.cycle_sweep:
        cycle_values = parse_int_list_arg(args.cycle_sweep)
        if not cycle_values:
            raise ValueError("--cycle-sweep must contain at least one cycle count")

        print("\n==============================")
        print(f"Mode: {args.mode} cycle sweep")
        print("==============================")
        print(f"cycles: {cycle_values}")
        rows = run_cycle_sweep(
            P=P,
            cycle_values=cycle_values,
            fixed_bit_width=args.fixed_bit_width,
            channel_input_mode=args.channel_input_mode,
            alpha_mode=params["alpha_mode"],
            ebno_db_list=ebno_db_list,
            alpha=params["alpha"],
            kw=params["kw"],
            kr=params["kr"],
            n_trials=args.n_trials,
            mode=args.mode,
            I0_min=params["I0_min"],
            I0_max=params["I0_max"],
            nrnd=params["nrnd"],
            psa_p=params["psa_p"],
            lambda_mem=params["lambda_mem"],
            response_lambda=params["response_lambda"],
            lambda_out=params["lambda_out"],
            I0_schedule_type=params["I0_schedule_type"],
            I0_schedule_shape=params["I0_schedule_shape"],
            I0_hold_fraction=params["I0_hold_fraction"],
            decision_method=params["decision_method"],
            burn_in=params["burn_in"],
            sample_window=params["sample_window"],
            use_all_zero_message=args.use_all_zero_message,
            bp_max_iter=args.bp_max_iter,
            bp_llr_clip=args.bp_llr_clip,
            seed=args.seed,
        )
        cycle_csv = args.cycle_sweep_csv or default_cycle_sweep_path(args.mode, P, "csv")
        cycle_plot = args.cycle_sweep_plot or default_cycle_sweep_path(args.mode, P, "png")
        saved_cycle_csv = save_cycle_sweep_csv(rows, cycle_csv)
        saved_cycle_plot = save_cycle_sweep_plot(rows, cycle_plot)
        print(f"Saved cycle sweep CSV: {saved_cycle_csv}")
        if saved_cycle_plot is not None:
            print(f"Saved cycle sweep plot: {saved_cycle_plot}")
        raise SystemExit(0)

    if args.search:
        search_csv = args.search_csv or default_search_result_path(args.mode, P, "csv")
        search_plot = args.search_plot or default_search_result_path(args.mode, P, "png")

        print("\n==============================")
        print(f"Mode: {args.mode} parameter search")
        print("==============================")
        print(f"Search CSV: {search_csv}")
        print(f"Search plot: {search_plot}")
        print(f"Search algorithm: {args.search_algorithm}")
        print(f"Channel input mode: {args.channel_input_mode}")
        print(f"Alpha mode: {args.alpha_mode}")
        if args.warm_start_csv:
            print(
                "Warm start: "
                f"csv={args.warm_start_csv}, "
                f"top_k={args.warm_start_top_k}, "
                f"local_samples_per_seed={args.local_samples_per_seed}, "
                f"local_scale={args.local_scale}"
            )
        if args.refine_top_k > 0 and args.refine_trials > args.search_trials:
            print(
                "Refinement: "
                f"top_k={args.refine_top_k}, "
                f"trials={args.refine_trials}"
            )

        refine_seeds = make_refine_seeds(
            args.seed,
            seed_count=args.refine_seed_count,
            seed_list=args.refine_seeds,
        )
        if args.search_algorithm == "optuna":
            records = optuna_parameter_search(
                P=P,
                ebno_db_list=ebno_db_list,
                base_params=params,
                mode=args.mode,
                fixed_bit_width=args.fixed_bit_width,
                channel_input_mode=args.channel_input_mode,
                n_cycles=args.n_cycles,
                n_trials=args.search_trials,
                n_candidates=args.search_candidates,
                seed=args.seed,
                search_seed=args.search_seed,
                use_all_zero_message=args.use_all_zero_message,
                bp_max_iter=args.bp_max_iter,
                bp_llr_clip=args.bp_llr_clip,
                n_workers=args.n_workers,
                warm_start_csv=args.warm_start_csv,
                warm_start_top_k=args.warm_start_top_k,
                refine_top_k=args.refine_top_k,
                refine_trials=args.refine_trials,
                refine_seeds=refine_seeds,
                storage=args.optuna_storage,
                study_name=args.optuna_study_name,
            )
        else:
            records = random_parameter_search(
                P=P,
                ebno_db_list=ebno_db_list,
                base_params=params,
                mode=args.mode,
                fixed_bit_width=args.fixed_bit_width,
                channel_input_mode=args.channel_input_mode,
                n_cycles=args.n_cycles,
                n_trials=args.search_trials,
                n_candidates=args.search_candidates,
                seed=args.seed,
                search_seed=args.search_seed,
                use_all_zero_message=args.use_all_zero_message,
                bp_max_iter=args.bp_max_iter,
                bp_llr_clip=args.bp_llr_clip,
                n_workers=args.n_workers,
                warm_start_csv=args.warm_start_csv,
                warm_start_top_k=args.warm_start_top_k,
                local_samples_per_seed=args.local_samples_per_seed,
                local_scale=args.local_scale,
                refine_top_k=args.refine_top_k,
                refine_trials=args.refine_trials,
                refine_seeds=refine_seeds,
            )

        search_ebno_csv = default_search_ebno_result_path(search_csv)

        csv_path = save_search_records_csv(records, search_csv)
        ebno_csv_path = save_search_ebno_records_csv(records, search_ebno_csv)
        plot_path = save_search_plot(records, search_plot)
        print(f"\nSaved search CSV: {csv_path}")
        print(f"Saved search Eb/No CSV: {ebno_csv_path}")
        print(f"Saved search plot: {plot_path}")

        print("\nTop parameter sets:")
        for idx, record in enumerate(records[:args.top_k], start=1):
            print(
                f"{idx}: trials={record['evaluation_trials']}, "
                f"FER={record['mean_pbit_FER']:.6f}, "
                f"BER={record['mean_pbit_BER']:.6f}, "
                f"kw={record['kw']:.6g}, kr={record['kr']:.6g}, "
                f"alpha={record['alpha']:.6g}, alpha_mode={record['alpha_mode']}, "
                f"nrnd={record['nrnd']:.6g}, "
                f"psa_p={record['psa_p']:.6g}, "
                f"lambda_mem={record['lambda_mem']:.6g}, "
                f"response_lambda={record.get('response_lambda', 0.0):.6g}, "
                f"lambda_out={record.get('lambda_out', 0.0):.6g}, "
                f"I0_min={record['I0_min']:.6g}, I0_max={record['I0_max']:.6g}, "
                f"I0_schedule={record['I0_schedule_type']}, "
                f"I0_shape={record['I0_schedule_shape']:.6g}, "
                f"I0_hold={record['I0_hold_fraction']:.6g}, "
                f"decision={record['decision_method']}, "
                f"burn_in={record['burn_in']}, window={record['sample_window']}"
            )

        if args.search_only:
            raise SystemExit(0)

        params = {key: records[0][key] for key in params}
        print("\nRun full simulation with best parameters.")
        mode = args.mode
    else:
        mode = args.mode

    print("\n==============================")
    print("Mode:", mode)
    print("==============================")

    G, results = simulate_ldpc_decoder(
        P=P,
        fixed_bit_width=args.fixed_bit_width,
        channel_input_mode=args.channel_input_mode,
        alpha_mode=params["alpha_mode"],
        ebno_db_list=ebno_db_list,
        alpha=params["alpha"],
        kw=params["kw"],
        kr=params["kr"],
        n_cycles=args.n_cycles,
        n_trials=args.n_trials,
        mode=mode,
        I0_min=params["I0_min"],
        I0_max=params["I0_max"],
        nrnd=params["nrnd"],
        psa_p=params["psa_p"],
        lambda_mem=params["lambda_mem"],
        response_lambda=params["response_lambda"],
        lambda_out=params["lambda_out"],
        I0_schedule_type=params["I0_schedule_type"],
        I0_schedule_shape=params["I0_schedule_shape"],
        I0_hold_fraction=params["I0_hold_fraction"],
        decision_method=params["decision_method"],
        burn_in=params["burn_in"],
        sample_window=params["sample_window"],
        use_all_zero_message=args.use_all_zero_message,
        bp_max_iter=args.bp_max_iter,
        bp_llr_clip=args.bp_llr_clip,
        seed=args.seed,
        print_matrices=args.print_matrices,
    )

    print("\nParameters:")
    print(
        f"kw={params['kw']:.6g}, kr={params['kr']:.6g}, "
        f"alpha={params['alpha']:.6g}, alpha_mode={params['alpha_mode']}, "
        f"nrnd={params['nrnd']:.6g}, "
        f"psa_p={params['psa_p']:.6g}, "
        f"lambda_mem={params['lambda_mem']:.6g}, "
        f"response_lambda={params['response_lambda']:.6g}, "
        f"lambda_out={params.get('lambda_out', 0.0):.6g}, "
        f"I0_min={params['I0_min']:.6g}, I0_max={params['I0_max']:.6g}, "
        f"I0_schedule={params['I0_schedule_type']}, "
        f"I0_shape={params['I0_schedule_shape']:.6g}, "
        f"I0_hold={params['I0_hold_fraction']:.6g}, "
        f"channel_input_mode={args.channel_input_mode}, "
        f"decision={params['decision_method']}, "
        f"burn_in={params['burn_in']}, window={params['sample_window']}"
    )

    print("\nGenerated G:")
    if args.print_matrices:
        print(G)
    else:
        print(f"shape={G.shape}")

    print("\nResults:")
    print_results(results)

    result_params = {
        "kw": params["kw"],
        "kr": params["kr"],
        "alpha": params["alpha"],
        "alpha_mode": params["alpha_mode"],
        "nrnd": params["nrnd"],
        "psa_p": params["psa_p"],
        "lambda_mem": params["lambda_mem"],
        "response_lambda": params["response_lambda"],
        "lambda_out": params.get("lambda_out", 0.0),
        "I0_min": params["I0_min"],
        "I0_max": params["I0_max"],
        "I0_schedule_type": params["I0_schedule_type"],
        "I0_schedule_shape": params["I0_schedule_shape"],
        "I0_hold_fraction": params["I0_hold_fraction"],
        "decision_method": params["decision_method"],
        "burn_in": params["burn_in"],
        "sample_window": params["sample_window"],
    }
    results_csv = args.results_csv or default_run_result_path(mode, P, "csv")
    results_plot = args.results_plot or default_run_result_path(mode, P, "png")
    saved_results_csv = save_run_results_csv(
        results,
        results_csv,
        mode,
        result_params,
        args.n_trials,
        args.n_cycles,
    )
    saved_results_plot = save_run_results_plot(results, results_plot)

    print(f"\nSaved run Eb/No CSV: {saved_results_csv}")
    if saved_results_plot is not None:
        print(f"Saved run BER/FER plot: {saved_results_plot}")

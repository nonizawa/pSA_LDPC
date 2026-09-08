"""Deterministic random 2-SAT generation and identity checks."""

from __future__ import annotations

import hashlib

import numpy as np


def formula_sha256(clauses: np.ndarray) -> str:
    values = np.asarray(clauses, dtype=np.int64, order="C")
    digest = hashlib.sha256()
    digest.update(str(values.shape).encode("ascii"))
    digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def _literal_node(variable: int, sign: int) -> int:
    return 2 * int(variable) + (0 if int(sign) == 1 else 1)


def is_satisfiable(clauses: np.ndarray, n_variables: int) -> bool:
    """Solve 2-SAT by strongly connected components of the implication graph."""
    n_nodes = 2 * int(n_variables)
    graph = [[] for _ in range(n_nodes)]
    reverse = [[] for _ in range(n_nodes)]
    for i, a, j, b in np.asarray(clauses, dtype=int):
        left = _literal_node(i, a)
        right = _literal_node(j, b)
        graph[left ^ 1].append(right)
        reverse[right].append(left ^ 1)
        graph[right ^ 1].append(left)
        reverse[left].append(right ^ 1)

    seen = [False] * n_nodes
    order: list[int] = []
    for start in range(n_nodes):
        if seen[start]:
            continue
        stack = [(start, 0)]
        seen[start] = True
        while stack:
            node, index = stack[-1]
            if index < len(graph[node]):
                neighbor = graph[node][index]
                stack[-1] = (node, index + 1)
                if not seen[neighbor]:
                    seen[neighbor] = True
                    stack.append((neighbor, 0))
            else:
                order.append(node)
                stack.pop()

    component = [-1] * n_nodes
    component_id = 0
    for start in reversed(order):
        if component[start] != -1:
            continue
        stack = [start]
        component[start] = component_id
        while stack:
            node = stack.pop()
            for neighbor in reverse[node]:
                if component[neighbor] == -1:
                    component[neighbor] = component_id
                    stack.append(neighbor)
        component_id += 1
    return all(component[2 * index] != component[2 * index + 1] for index in range(n_variables))


def generate_random_2sat(n_variables: int, alpha: float, seed: int) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    n_clauses = max(1, int(round(float(alpha) * int(n_variables))))
    clauses = []
    for _ in range(n_clauses):
        i, j = rng.choice(n_variables, size=2, replace=False)
        a = int(rng.choice(np.array([-1, 1], dtype=int)))
        b = int(rng.choice(np.array([-1, 1], dtype=int)))
        clauses.append((int(i), a, int(j), b))
    clause_array = np.asarray(clauses, dtype=np.int64)
    return {
        "n_variables": int(n_variables),
        "n_clauses": int(n_clauses),
        "alpha": float(alpha),
        "seed": int(seed),
        "clauses": clause_array,
        "sha256": formula_sha256(clause_array),
        "is_satisfiable": is_satisfiable(clause_array, int(n_variables)),
    }

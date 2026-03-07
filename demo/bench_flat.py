#!/usr/bin/env python3
"""
Benchmark: SqliteFlatBackend — BQ prescore + exact rerank.

Generates 50 000 fake vectors (dim=1024), loads them into the flat backend,
and benchmarks search() latency.

Usage:
    python demo/bench_flat.py
    python demo/bench_flat.py --n 100000 --dim 256 --runs 20
"""

import argparse
import array
import os
import random
import struct
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sqfox.backends.flat import SqliteFlatBackend, blas_loaded


def generate_vectors(n: int, dim: int, seed: int = 42) -> list[list[float]]:
    """Generate n random float32 vectors of given dimensionality."""
    rng = random.Random(seed)
    vecs = []
    for _ in range(n):
        v = [rng.gauss(0, 1) for _ in range(dim)]
        # Normalize
        norm = sum(x * x for x in v) ** 0.5
        if norm > 0:
            v = [x / norm for x in v]
        vecs.append(v)
    return vecs


def main():
    parser = argparse.ArgumentParser(description="Benchmark SqliteFlatBackend")
    parser.add_argument("--n", type=int, default=50_000, help="Number of vectors")
    parser.add_argument("--dim", type=int, default=1024, help="Vector dimensionality")
    parser.add_argument("--k", type=int, default=10, help="Top-K results")
    parser.add_argument("--runs", type=int, default=10, help="Number of search runs")
    parser.add_argument("--oversample", type=int, default=20, help="BQ oversample factor")
    args = parser.parse_args()

    N = args.n
    DIM = args.dim
    K = args.k
    RUNS = args.runs

    print("=" * 60)
    print(f"  SqliteFlatBackend Benchmark")
    print(f"  Vectors: {N:,}  Dim: {DIM}  K: {K}  Runs: {RUNS}")
    print(f"  Oversample: {args.oversample}x")
    print(f"  BLAS: {'YES' if blas_loaded() else 'NO (math.dist fallback)'}")
    print(f"  bit_count: {'int.bit_count (C)' if hasattr(int, 'bit_count') else 'bin().count (fallback)'}")
    print("=" * 60)

    # -- Generate vectors --------------------------------------
    print(f"\nGenerating {N:,} vectors (dim={DIM})...", end=" ", flush=True)
    t0 = time.perf_counter()
    vecs = generate_vectors(N, DIM)
    gen_ms = (time.perf_counter() - t0) * 1000
    print(f"{gen_ms:.0f}ms")

    mem_vecs_mb = N * DIM * 4 / 1024 / 1024
    mem_bq_mb = N * (DIM // 8) / 1024 / 1024
    print(f"  Float32 memory: ~{mem_vecs_mb:.1f} MB")
    print(f"  BQ memory:      ~{mem_bq_mb:.1f} MB")

    # -- Load into backend -------------------------------------
    backend = SqliteFlatBackend(oversample=args.oversample)
    # Initialize sets up BLAS for this dimensionality (no DB needed)
    backend.initialize(":memory:", DIM)

    print(f"\nLoading {N:,} vectors into flat backend...", end=" ", flush=True)
    t0 = time.perf_counter()
    keys = list(range(1, N + 1))
    backend.add(keys, vecs)
    load_ms = (time.perf_counter() - t0) * 1000
    print(f"{load_ms:.0f}ms")
    print(f"  Count: {backend.count():,}")

    # -- Warmup ------------------------------------------------
    query = vecs[0]
    print(f"\nWarmup search...", end=" ", flush=True)
    result = backend.search(query, K)
    print(f"top-1 = doc_id={result[0][0]}, dist={result[0][1]:.6f}")

    # -- Benchmark ---------------------------------------------
    print(f"\nBenchmarking {RUNS} searches...")
    times = []
    rng = random.Random(123)
    queries = [vecs[rng.randint(0, N - 1)] for _ in range(RUNS)]

    for i, q in enumerate(queries):
        t0 = time.perf_counter()
        result = backend.search(q, K)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        times.append(elapsed_ms)

    times.sort()
    avg = sum(times) / len(times)
    p50 = times[len(times) // 2]
    p95 = times[int(len(times) * 0.95)]
    p99 = times[int(len(times) * 0.99)] if len(times) >= 100 else times[-1]

    print(f"\n{'=' * 60}")
    print(f"  RESULTS ({N:,} vectors, dim={DIM}, K={K})")
    print(f"  {'-' * 56}")
    print(f"  Average:  {avg:8.2f} ms")
    print(f"  P50:      {p50:8.2f} ms")
    print(f"  P95:      {p95:8.2f} ms")
    print(f"  Best:     {times[0]:8.2f} ms")
    print(f"  Worst:    {times[-1]:8.2f} ms")
    print(f"  {'-' * 56}")
    print(f"  BLAS:     {'YES' if blas_loaded() else 'NO'}")
    print(f"  Rerank:   {backend.rerank_method}")
    print(f"  Pipeline: map/zip/heapq (all C-level)")
    print(f"{'=' * 60}")

    # -- Recall check ------------------------------------------
    import math
    print(f"\nRecall check (comparing BQ+rerank vs exact brute-force)...")
    # Exact brute-force on first query
    q_arr = array.array("f", queries[0])
    all_dists = [(math.dist(q_arr, array.array("f", vecs[i])), i + 1)
                 for i in range(N)]
    all_dists.sort()
    exact_topk = set(did for _, did in all_dists[:K])

    approx_result = backend.search(queries[0], K)
    approx_topk = set(did for did, _ in approx_result)

    recall = len(exact_topk & approx_topk) / K
    print(f"  Recall@{K}: {recall:.1%}")
    if recall < 1.0:
        missed = exact_topk - approx_topk
        print(f"  Missed: {missed}")

    backend.close()


if __name__ == "__main__":
    main()

import os
import sys
import time
import numpy as np

# Ensure root folder is accessible for module imports
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

try:
    from app.retriever import search
except ImportError:
    from retriever import search

# Pool of benchmark test queries
BENCHMARK_QUERIES = [
    "What is a corporation?",
    "निगम क्या है?",
    "کارپوریشن کیا ہے؟",
    "What is the Manhattan Project?",
    "मैनहट्टन परियोजना क्या है?",
    "Constitution of India summary",
    "High performance neural search"
]

def run_benchmark(num_queries: int = 50):
    print("Warming up (model load + first inference)...")
    
    # 1. Warm-up Phase: Pre-load cache and ONNX runtime execution threads
    for q in BENCHMARK_QUERIES:
        try:
            search(q, top_k=3)
        except Exception:
            pass

    print(f"\nRan {num_queries} queries\n")

    embed_times = []
    search_times = []
    total_times = []

    # 2. Timed Benchmark Execution Loop
    for i in range(num_queries):
        query = BENCHMARK_QUERIES[i % len(BENCHMARK_QUERIES)]

        t_start = time.perf_counter()
        results, embed_ms, search_ms = search(query, top_k=3)
        t_total = (time.perf_counter() - t_start) * 1000.0

        embed_times.append(embed_ms)
        search_times.append(search_ms)
        total_times.append(t_total)

    # 3. Calculate Percentile Metrics
    def calc_stats(arr):
        return {
            "avg": np.mean(arr),
            "p50": np.percentile(arr, 50),
            "p75": np.percentile(arr, 75),
            "p100": np.percentile(arr, 100),
        }

    e_stat = calc_stats(embed_times)
    s_stat = calc_stats(search_times)
    t_stat = calc_stats(total_times)

    # 4. Display Results Table
    print(f"{'stage':<10} {'avg':>8} {'p50':>8} {'p75':>8} {'p100':>8}   (ms)")
    print(f"{'embed':<10} {e_stat['avg']:>8.2f} {e_stat['p50']:>8.2f} {e_stat['p75']:>8.2f} {e_stat['p100']:>8.2f}")
    print(f"{'search':<10} {s_stat['avg']:>8.2f} {s_stat['p50']:>8.2f} {s_stat['p75']:>8.2f} {s_stat['p100']:>8.2f}")
    print(f"{'total':<10} {t_stat['avg']:>8.2f} {t_stat['p50']:>8.2f} {t_stat['p75']:>8.2f} {t_stat['p100']:>8.2f}")

    print(f"\nLatency budget: 50.0ms | p100 total: {t_stat['p100']:.2f}ms")
    if t_stat['p100'] <= 50.0:
        print("PASS: within budget")
    else:
        print("FAIL: exceeds budget")

if __name__ == "__main__":
    run_benchmark(50)
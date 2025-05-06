#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from hopfield import simulation
"""
from hopfield import extract_pattern, compute_couplings, init_neurons, dynamic

def run_trial_first_pattern(N, p, sweep_max, T, delta=False):
    patterns = extract_pattern(N, p, 0.0, delta=delta)
    J = compute_couplings(N, patterns)
    neurons = init_neurons(patterns, "pattern")  # starts from a stored pattern
    evolution = np.fromiter(
        dynamic(neurons, J, patterns, sweep_max, T),
        dtype=np.dtype((float, p + 1))  # (p patterns + 1 mixture column)
    )
    final_overlap_first = evolution[-1, 0]  # overlap with pattern 0 at final sweep
    return final_overlap_first
"""
def compute_first_pattern_capacity(N_values, load_values, sweep_max, T, s, delta):
    results = {}

    for N in N_values:
        means = []
        stds = []

        print(f"# Running for N = {N}")
        for load in tqdm(load_values):
            p = int(load * N)
            m_values = [simulation(N, p, sweep_max, 0.0, T, init_type="pattern", delta=delta)[-1,0] for _ in range(s)]
            means.append(np.mean(m_values))
            stds.append(np.std(m_values))

        results[N] = {
            "load": load_values,
            "mean": np.array(means),
            "std": np.array(stds)
        }

    return results

def print_csv(results):
    print("# N,load,magnetization_mean,magnetization_std")
    for N, data in results.items():
        for load, mean, std in zip(data["load"], data["mean"], data["std"]):
            print(f"{N},{load:.5f},{mean:.6f},{std:.6f}")

def plot_first_pattern_capacity(results):
    plt.figure(figsize=(10, 6))
    for N, stats in results.items():
        plt.errorbar(
            stats["load"], stats["mean"], yerr=stats["std"],
            label=f"N = {N}", capsize=4, fmt='-o'
        )

    plt.xlabel("Load (α = p/N)")
    plt.ylabel("Final magnetization w.r.t. pattern 1")
    plt.title("Critical Capacity: Final Magnetization (Pattern 1)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Parameters
N_values = [1000, 2000, 4000]
load_values = np.arange(0.09, 0.201, 0.005)
sweep_max = 100
samples = 20
temperature = 0.0  # deterministic update

# Run and plot
results = compute_first_pattern_capacity(
    N_values, load_values, sweep_max, T=temperature, s=samples, delta=False
)

print_csv(results)
plot_first_pattern_capacity(results)

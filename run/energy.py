#!/usr/bin/env python3

import argparse
import sys
import os

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from hopfield import varying_a_energy, varying_a_energy_stat, energy_stat_vary_mixs, varying_a_energy_stat_until_n_twenty

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-N", type=int, default=1000, help="number of neurons")
parser.add_argument("-p", type=int, default=9, help="number of patterns")
parser.add_argument("-a", type=float, default=0, help="parameter of the distribution of probability")
parser.add_argument("-s", type=int, default=20, help="number of samples")
parser.add_argument('--d', action='store_true', help="if delta is true it do the delta simulation")

arguments = parser.parse_args()

"""
stats_5 = energy_stat_vary_mixs(
    N=arguments.N,
    p=arguments.p,
    delta=arguments.d
)

stats = varying_a_energy_stat(
    N=arguments.N,
    p=arguments.p,
    delta=arguments.d
)
"""
stats_7 = varying_a_energy_stat_until_n_twenty(
    N=arguments.N,
    p=arguments.p,
    max_n=19,
    delta=arguments.d
)

# print(stats_7)

a_values = np.linspace(0, 1, 25)

def plot_energy_stats(stats, a_values):
    """Visualize the mean and std of energy for patterns and mixtures over varying a."""
    plt.figure(figsize=(10, 6))

    # Plot means with error bars
    plt.errorbar(a_values, stats["patterns_mean"], yerr=stats["patterns_std"],
                 label="Patterns", fmt='-o', capsize=4)
    plt.errorbar(a_values, stats["mixtures_mean"], yerr=stats["mixtures_std"],
                 label="Mixtures", fmt='-s', capsize=4)

    plt.xlabel("a (distribution parameter)")
    plt.ylabel("Energy")
    plt.title("Energy Statistics for Patterns and Mixtures")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_energy_stats_5(stats, a_values):
    plt.figure(figsize=(10, 6))

    plt.errorbar(a_values, stats["patterns_mean"], yerr=stats["patterns_std"],
                 label="Patterns", fmt='-o', capsize=4)
    plt.errorbar(a_values, stats["3mix_mean"], yerr=stats["3mix_std"],
                 label="3-Mixtures", fmt='-s', capsize=4)
    plt.errorbar(a_values, stats["5mix_mean"], yerr=stats["5mix_std"],
                 label="5-Mixtures", fmt='-^', capsize=4)

    plt.xlabel("a (distribution parameter)")
    plt.ylabel("Energy")
    plt.title("Energy Statistics for Patterns, 3-Mixtures, and 5-Mixtures")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_energy_stats_until_n(stats):
    plt.figure(figsize=(10, 6))
    a_values = stats["a_values"]

    plt.errorbar(a_values, stats["patterns_mean"], yerr=stats["patterns_std"],
                 label="Patterns", fmt='-o', capsize=4)

    for key in stats:
        if "mix_mean" in key:
            k = key.split("mix")[0]
            mean = stats[key]
            std = stats[f"{k}mix_std"]
            plt.errorbar(a_values, mean, yerr=std,
                         label=f"{k}-Mixtures", fmt='-s', capsize=4)

    plt.xlabel("a (distribution parameter)")
    plt.ylabel("Energy")
    plt.title("Energy Statistics for Patterns and Mixtures (Odd n ≤ max_n)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


#plot_energy_stats(stats, a_values)

#plot_energy_stats_5(stats_5, a_values)


plot_energy_stats_until_n(stats_7)
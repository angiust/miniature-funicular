#!/usr/bin/env python3

import sys
import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-i", "--input", type=str, default="-", help="name of the input file ('-' for stdin)")
parser.add_argument("--title", type=str, default="", help="custom plot title")
parser.add_argument("--output", type=str, help="output filename if saving the plot")

arguments = parser.parse_args()

filename = arguments.input if arguments.input != '-' else sys.stdin
data = np.loadtxt(filename, delimiter=",")

time = np.arange(data.shape[0])
num_columns = data.shape[1]

# each magnetization and std pair: m1, std1, m2, std2, ..., mp, stdp
num_pairs = num_columns // 2

plt.figure(figsize=(10, 6))

colors = plt.cm.viridis(np.linspace(0, 1, num_pairs))

for i in range(num_pairs):
    m = data[:, i]
    std = data[:, num_pairs + i]
    plt.plot(time, m, label=f"m_{i+1}", color=colors[i])
    plt.fill_between(time, m - std, m + std, color=colors[i], alpha=0.2)

plt.title(f"Magnetization vs Time {arguments.title}")
plt.xlabel("Time Step")
plt.ylabel("Magnetization")
plt.grid(True)
plt.legend()
plt.tight_layout()

if arguments.output:
    plt.savefig(arguments.output)
else:
    plt.show()

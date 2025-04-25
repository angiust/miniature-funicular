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

# Infer number of magnetization components and energy
# Format: m_1, ..., m_p, e, std_m_1, ..., std_m_p, std_e
half = num_columns // 2
p = half - 1  # there is 1 energy column

magnetizations = data[:, :p]
energy = data[:, p]
std_magnetizations = data[:, half:half + p]
std_energy = data[:, half + p]

colors = plt.cm.viridis(np.linspace(0, 1, p))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8), sharex=True)
#fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# --- Magnetizations ---
for i in range(p):
    m = magnetizations[:, i]
    std = std_magnetizations[:, i]
    ax1.plot(time, m, label=f"$m_{{{i+1}}}$", color=colors[i])
    ax1.fill_between(time, m - std, m + std, color=colors[i], alpha=0.2)

ax1.set_ylabel("Magnetization")
ax1.set_title(f"Magnetization and Energy vs Time {arguments.title}")
ax1.grid(True)
ax1.legend(loc='upper right')

# --- Energy ---
ax2.plot(time, energy, label="Energy", color="tab:red")
ax2.fill_between(time, energy - std_energy, energy + std_energy, color="tab:red", alpha=0.3)
ax2.set_ylabel("Energy")
ax2.set_xlabel("Time Step")
ax2.grid(True)
ax2.legend(loc='upper right')

plt.tight_layout()

# Save or show
if arguments.output:
    plt.savefig(arguments.output)
else:
    plt.show()

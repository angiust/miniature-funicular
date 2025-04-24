#!/usr/bin/env python3

import sys
import numpy as np
import matplotlib.pyplot as plt
import argparse
import math

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-i", "--input", type=str, default="-", help="Input CSV file ('-' for stdin)")
parser.add_argument("--title", type=str, default="Magnetization and Energy Evolution", help="Plot title base")
parser.add_argument("--output", type=str, help="Base filename for saving plots (appends '_start_i.png')")
parser.add_argument("--max-plots", type=int, default=9, help="Maximum number of pattern initializations to plot")

arguments = parser.parse_args()

# Load data
filename = arguments.input if arguments.input != '-' else sys.stdin
try:
    data = np.loadtxt(filename, delimiter=",", skiprows=1)
except Exception as e:
    print(f"Error loading data from {filename}: {e}", file=sys.stderr)
    sys.exit(1)

# Infer shape
num_sweeps = data.shape[0]
total_columns = data.shape[1]
# Infer p from shape: total_columns = p * (p + 1)
a = 1
b = 1
c = -data.shape[1]

discriminant = b**2 - 4 * a * c
if discriminant < 0:
    print("Error: Cannot infer 'p'. Discriminant < 0", file=sys.stderr)
    sys.exit(1)

sqrt_disc = math.isqrt(discriminant)
if sqrt_disc * sqrt_disc != discriminant:
    print("Error: Discriminant not a perfect square", file=sys.stderr)
    sys.exit(1)

p = (-b + sqrt_disc) // 2
if p * (p + 1) != data.shape[1]:
    print(f"Error: Column count ({data.shape[1]}) doesn't match p*(p+1) format (p inferred: {p})", file=sys.stderr)
    sys.exit(1)


if p * (p + 1) != total_columns:
    print(f"Error: Column count ({total_columns}) doesn't match p*(p+1) format.", file=sys.stderr)
    sys.exit(1)

print(f"Data loaded: {num_sweeps} sweeps, p={p} patterns inferred.")

plots_to_show = min(p, arguments.max_plots)

sweeps_axis = np.arange(1, num_sweeps + 1)

# Reshape the data: from (sweeps, p * (p+1)) → (p, sweeps, p+1)
reshaped = data.reshape(num_sweeps, p, p + 1).transpose(1, 0, 2)

plt.style.use('seaborn-v0_8-darkgrid')

for start_pattern_idx in range(plots_to_show):
    print(f"Plotting simulation starting from pattern {start_pattern_idx + 1}...")

    fig, (ax_mag, ax_energy) = plt.subplots(1, 2, figsize=(14, 5))

    simulation_data = reshaped[start_pattern_idx]  # shape (sweeps, p + 1)
    magnetizations = simulation_data[:, :p]
    energy = simulation_data[:, -1]

    # --- Magnetization plot ---
    colors = plt.cm.viridis(np.linspace(0, 0.9, p))
    for j in range(p):
        label = f"$m_{{{j+1}}}$"
        linewidth = 2.5 if j == start_pattern_idx else 1.5
        alpha = 1.0 if j == start_pattern_idx else 0.7
        ax_mag.plot(sweeps_axis, magnetizations[:, j], label=label, color=colors[j], linewidth=linewidth, alpha=alpha)

    ax_mag.set_title(f"{arguments.title} - Magnetization (Start: Pattern {start_pattern_idx + 1})")
    ax_mag.set_xlabel("Number of Sweeps")
    ax_mag.set_ylabel("Magnetization")
    ax_mag.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')
    ax_mag.grid(True, linestyle='--', alpha=0.7)
    ax_mag.set_xlim(left=0)
    ax_mag.set_ylim(min(np.min(magnetizations), -0.1), max(np.max(magnetizations), 1.1))

    # --- Energy plot ---
    ax_energy.plot(sweeps_axis, energy, color='black', label="Energy", linewidth=2)
    ax_energy.set_title(f"Energy Evolution (Start: Pattern {start_pattern_idx + 1})")
    ax_energy.set_xlabel("Number of Sweeps")
    ax_energy.set_ylabel("Energy")
    ax_energy.grid(True, linestyle='--', alpha=0.7)
    ax_energy.set_xlim(left=0)

    plt.tight_layout(rect=[0, 0, 0.95, 1])  # Leave space for legend

    if arguments.output:
        output_filename = f"{arguments.output}_start_{start_pattern_idx + 1}.png"
        try:
            plt.savefig(output_filename)
            print(f"  Saved to {output_filename}")
        except Exception as e:
            print(f"  Error saving plot: {e}", file=sys.stderr)
        plt.close(fig)
    else:
        plt.show()

if not arguments.output:
    plt.show()

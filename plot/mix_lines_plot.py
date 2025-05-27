#!/usr/bin/env python3

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

# --- Argument parsing ---
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-i", "--input", type=str, default="-", help="Input CSV file (use '-' for stdin)")
parser.add_argument("--samples", type=int, default=20, help="Number of samples to plot")
parser.add_argument("--title", type=str, default="Samples Magnetization and Energy", help="Base title for the plots")
parser.add_argument("--output", type=str, help="Base output filename (without extension)")
parser.add_argument("-a", "--asymmetry", type=float, default=None, help="Value of asymmetry parameter a for reference lines")

args = parser.parse_args()

# --- Load data ---
filename = args.input if args.input != "-" else sys.stdin
try:
    data = np.loadtxt(filename, delimiter=",", skiprows=1)
except Exception as e:
    print(f"Error loading data: {e}", file=sys.stderr)
    sys.exit(1)

# --- Check dimensions ---
num_columns = data.shape[1]
num_samples = args.samples
if num_columns % num_samples != 0:
    print(f"Error: total columns {num_columns} not divisible by number of samples {num_samples}.", file=sys.stderr)
    sys.exit(1)

# --- Infer number of patterns p ---
columns_per_sample = num_columns // num_samples
p = columns_per_sample - 1
print(f"Detected {p} patterns, {num_samples} samples.")

# --- Create plot style ---
plt.style.use('seaborn-v0_8-darkgrid')

# --- Generate plots ---
for sample_idx in range(num_samples):
    start_col = sample_idx * (p + 1)
    end_col = start_col + p

    magnetizations = data[:, start_col:end_col]   # shape (sweeps, p)
    energy = data[:, end_col]                     # shape (sweeps,)
    sweeps_axis = np.arange(1, data.shape[0] + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8), sharex=True)

    # Plot magnetizations
    for pattern_idx in range(p):
        ax1.plot(sweeps_axis, magnetizations[:, pattern_idx], label=f"$m_{{{pattern_idx + 1}}}$")

    ax1.set_ylabel("Magnetization")
    ax1.set_title(f"{args.title} (Sample {sample_idx + 1})")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(fontsize='small', ncol=2)

    # Optional reference lines based on asymmetry
    if args.asymmetry is not None:
        a = args.asymmetry
        base = 1 - a
        ref_values = [base * (a ** i) for i in range(9)]  # up to a^8
        for val in ref_values:
            ax1.axhline(y=val, linestyle="--", color="red", linewidth=1, alpha=0.5)
        ax1.text(sweeps_axis[-1], ref_values[0], f"(1-a)={base:.2f}", fontsize=8, color="gray", ha="right", va="bottom")

    # Plot energy
    ax2.plot(sweeps_axis, energy, label="Energy", color="red")
    ax2.set_xlabel("Sweep")
    ax2.set_ylabel("Energy")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend()

    plt.tight_layout()

    # --- Save or show ---
    if args.output:
        output_filename = f"{args.output}_sample_{sample_idx + 1}.png"
        try:
            plt.savefig(output_filename)
            print(f"Saved plot to {output_filename}")
        except Exception as e:
            print(f"Error saving plot {output_filename}: {e}", file=sys.stderr)
        plt.close(fig)
    else:
        plt.show()

# If no --output, show everything
if not args.output:
    plt.show()


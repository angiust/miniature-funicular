#!/usr/bin/env python3
# histogram_plot.py

import sys
import numpy as np
import matplotlib.pyplot as plt
import argparse
import math

parser = argparse.ArgumentParser(
    description="Generates histograms of FINAL absolute overlaps and FINAL energy from simulation samples.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
# Input File or Stdin
parser.add_argument("-i", "--input", type=str, default="-",
                    help="Input CSV file (m1..p_final, E_final). Use '-' for stdin.")

# Plotting Options
parser.add_argument("--title-base", type=str, default="Final State Distribution (T=0)", help="Base title for plots")
parser.add_argument("--output-base", type=str, help="Base filename for saving plots (e.g., 'hist'. Appends '_m{i}.png', '_E.png').")
parser.add_argument("--skiprows", type=int, default=1, help="Number of header rows to skip")
parser.add_argument("--bins-mag", type=int, default=50, help="Number of bins for magnetization histograms")
parser.add_argument("--bins-energy", type=int, default=50, help="Number of bins for energy histogram")
parser.add_argument("--max-patterns-plot", type=int, default=None, help="Max number of pattern histograms to plot")
# Removed --range-mag argument, will default to [0, max(1, data_max)]
parser.add_argument("--range-energy", type=float, nargs=2, default=None, help="Range for energy histogram x-axis (min max). Auto-detected if None.")


arguments = parser.parse_args()

# --- Determine Input Source ---
if arguments.input == '-':
    input_source = sys.stdin
    source_name = "stdin"
    print("Loading final overlaps and energy from standard input (stdin)...")
else:
    input_source = arguments.input
    source_name = arguments.input
    print(f"Loading final overlaps and energy from file: {arguments.input}")


# --- Load Combined Data ---
try:
    final_results = np.loadtxt(input_source, delimiter=",", skiprows=arguments.skiprows)
    if final_results.ndim == 0: raise ValueError("No data loaded.")
    if final_results.ndim == 1: final_results = final_results.reshape(1, -1)

    num_samples = final_results.shape[0]
    num_cols = final_results.shape[1]
    if num_cols < 2: raise ValueError("Expected at least 2 columns (m1, E).")
    p = num_cols - 1

    print(f"  Loaded {num_samples} samples, p={p} patterns inferred.")

    valid_rows_mask = ~np.isnan(final_results).any(axis=1)
    num_nan_rows = np.sum(~valid_rows_mask)
    if num_nan_rows > 0:
        print(f"  Warning: Removed {num_nan_rows} runs containing NaN values.")
        final_results = final_results[valid_rows_mask, :]
        num_samples = final_results.shape[0]
        if num_samples == 0: raise ValueError("No valid runs found after removing NaNs.")

    final_overlaps = final_results[:, 0:p]
    final_energy = final_results[:, p]

except Exception as e:
    print(f"Error loading or processing data from {source_name}: {e}", file=sys.stderr)
    sys.exit(1)

# --- Calculate Absolute Magnitudes ---
abs_final_overlaps = np.abs(final_overlaps)

# --- Generate Histograms ---
plt.style.use('seaborn-v0_8-darkgrid')
figures = []

# --- Plot Magnetization Histograms ---
patterns_to_plot = p if arguments.max_patterns_plot is None else min(p, arguments.max_patterns_plot)
if patterns_to_plot < p:
     print(f"Warning: Plotting only histograms for the first {patterns_to_plot} patterns.")

for i in range(patterns_to_plot):
    fig_hist_m, ax_hist_m = plt.subplots(figsize=(8, 5))
    figures.append(fig_hist_m)

    data_for_hist = abs_final_overlaps[:, i]
    pattern_label = f"$|m_{{{i+1}}}|$"
    print(f"Generating histogram for {pattern_label}...")

    # --- Determine range for magnetization histogram ---
    min_val = 0 # Absolute value is always >= 0
    max_val = np.max(data_for_hist) if data_for_hist.size > 0 else 1.0
    # Set upper bound to at least 1, or slightly more than max_val if max_val > 1
    upper_bound = max(1.0, max_val * 1.05) # Use 1.0 or 5% above max if max>1
    hist_range_mag = [min_val, upper_bound]
    print(f"  Using magnitude range for histogram calculation: [{hist_range_mag[0]:.3f}, {hist_range_mag[1]:.3f}]")
    # ----------------------------------------------------

    counts, bin_edges, patches = ax_hist_m.hist(
        data_for_hist,
        bins=arguments.bins_mag,
        range=hist_range_mag, # Use calculated range
        density=False,
        color='skyblue', edgecolor='black', alpha=0.8
    )
    mean_val = np.mean(data_for_hist) if data_for_hist.size > 0 else np.nan
    ax_hist_m.axvline(mean_val, color='red', linestyle='dashed', linewidth=1.5, label=f'Mean = {mean_val:.3f}')
    ax_hist_m.set_title(f"{arguments.title_base}\nDistribution of {pattern_label} ({num_samples} runs)")
    ax_hist_m.set_xlabel(f"Final Magnitude ({pattern_label})")
    ax_hist_m.set_ylabel("Frequency (Number of Runs)")
    ax_hist_m.grid(True, axis='y', linestyle='--', alpha=0.6)
    ax_hist_m.legend()
    ax_hist_m.set_xlim(hist_range_mag) # Set plot view limits to match calculation range
    plt.tight_layout()

    # Save Plot
    if arguments.output_base:
        output_filename = f"{arguments.output_base}_hist_m{i+1}.png"
        try: plt.savefig(output_filename); print(f"  Histogram saved to {output_filename}")
        except Exception as e: print(f"  Error saving plot {output_filename}: {e}", file=sys.stderr)
        plt.close(fig_hist_m)

# --- Plot Energy Histogram (logic remains the same) ---
print("Generating histogram for Final Energy...")
fig_hist_e, ax_hist_e = plt.subplots(figsize=(8, 5))
figures.append(fig_hist_e)

data_for_hist_e = final_energy
# Determine range for histogram calculation
hist_range_energy = arguments.range_energy
if hist_range_energy is None:
    if data_for_hist_e.size > 0:
        min_val, max_val = np.min(data_for_hist_e), np.max(data_for_hist_e)
        padding = max( (max_val - min_val) * 0.05, 1e-6 )
        hist_range_energy = [min_val - padding, max_val + padding]
        print(f"  Auto-detected energy range: [{hist_range_energy[0]:.3f}, {hist_range_energy[1]:.3f}]")
    else:
        hist_range_energy = [-1, 1] # Default if no data
        print("  Warning: No energy data found, using default range [-1, 1].")

view_xlim_energy = hist_range_energy

counts, bin_edges, patches = ax_hist_e.hist(
    data_for_hist_e,
    bins=arguments.bins_energy,
    range=hist_range_energy,
    density=False,
    color='lightcoral', edgecolor='black', alpha=0.8
)
mean_val_e = np.mean(data_for_hist_e) if data_for_hist_e.size > 0 else np.nan
ax_hist_e.axvline(mean_val_e, color='darkred', linestyle='dashed', linewidth=1.5, label=f'Mean = {mean_val_e:.3f}')
ax_hist_e.set_title(f"{arguments.title_base}\nDistribution of Final Energy ({num_samples} runs)")
ax_hist_e.set_xlabel("Final Energy")
ax_hist_e.set_ylabel("Frequency (Number of Runs)")
ax_hist_e.grid(True, axis='y', linestyle='--', alpha=0.6)
ax_hist_e.legend()
ax_hist_e.set_xlim(view_xlim_energy)
plt.tight_layout()

# Save Plot
if arguments.output_base:
    output_filename = f"{arguments.output_base}_hist_E.png"
    try: plt.savefig(output_filename); print(f"  Histogram saved to {output_filename}")
    except Exception as e: print(f"  Error saving plot {output_filename}: {e}", file=sys.stderr)
    plt.close(fig_hist_e)


# --- Show plots if not saved ---
if not arguments.output_base:
    if figures:
         print("Displaying histograms...")
         plt.show()
    else:
         print("No figures generated.")

print("Histogram generation complete.")
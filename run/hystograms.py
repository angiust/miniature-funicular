#!/usr/bin/env python3

import argparse
import sys
import os

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from hopfield import multiple_simulation_all_story


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-N", type=int, default=1000, help="number of neurons")
parser.add_argument("-p", type=int, default=9, help="number of patterns")
parser.add_argument("-t", type=int, default=3000, help="number of sweeps")
parser.add_argument("-a", type=float, default=0, help="parameter of the distribution of probability")
parser.add_argument("-T", type=float, default=0, help="temperature of the system")
parser.add_argument("-s", type=int, default=20, help="number of samples")
#parser.add_argument('--mix', action='store_true', help="if mix is true it do the mixture simulation")
parser.add_argument(
    "--init_type",
    choices=["pattern", "mixture", "random"],
    default="pattern",
    help="Initialization type: 'pattern', 'mixture', or 'random'."
)

arguments = parser.parse_args()


# --- Run Multiple Simulations ---
final_overlaps_all_runs = multiple_simulation_all_story(
    N=arguments.N,
    p=arguments.p,
    sweep_max=arguments.t,
    a=arguments.a,
    T=arguments.T, # Should be 0
    s=arguments.s,
    init_type=arguments.init_type
)

# --- Save Data ---
if final_overlaps_all_runs is not None:
    header_list = [f"m_{j+1}_final" for j in range(arguments.p)] + ["e"]
    header_str = ",".join(header_list)
    print(f"# Saving final overlaps for {np.sum(~np.isnan(final_overlaps_all_runs[:,0]))} successful runs...") # Count non-NaN runs
    np.savetxt(
        sys.stdout,
        final_overlaps_all_runs, # Shape (s, p)
        delimiter=",",
        header=header_str,
        fmt="%.6f",
    )
    print(f"\n# Saved final overlaps for {arguments.s} runs.")
else:
    print("# Simulation failed to produce results.")
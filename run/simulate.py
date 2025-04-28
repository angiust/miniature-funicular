#!/usr/bin/env python3

import argparse
import sys
import os

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from hopfield import multiple_simulation

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-N", type=int, default=1000, help="number of neurons")
parser.add_argument("-p", type=int, default=9, help="number of patterns")
parser.add_argument("-t", type=int, default=100, help="number of sweeps")
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
parser.add_argument("--seed", type=int, default=None, help="random seed for reproducibility")
parser.add_argument('--d', action='store_true', help="if delta is true it do the delta simulation")

arguments = parser.parse_args()

if arguments.seed is not None:
    np.random.seed(arguments.seed)

multiple_evolution = multiple_simulation(
    N=arguments.N,
    p=arguments.p,
    sweep_max=arguments.t,
    a=arguments.a,
    T=arguments.T,
    s=arguments.s,
    init_type=arguments.init_type,
    delta=arguments.d
)

p=arguments.p
header = ",".join([f"m_{i+1}" for i in range(p)] + ["e"] + [f"std_{i+1}" for i in range(p)] + ["std_e"])
np.savetxt(sys.stdout, multiple_evolution, delimiter=",", fmt="%.5f", header=header)

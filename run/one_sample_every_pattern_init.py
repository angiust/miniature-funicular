#!/usr/bin/env python3

import argparse
import sys
import os

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from hopfield import simulation_all_pattern_init, multiple_simulation_random

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-N", type=int, default=1000, help="number of neurons")
parser.add_argument("-p", type=int, default=9, help="number of patterns")
parser.add_argument("-t", type=int, default=100, help="number of sweeps")
parser.add_argument("-a", type=float, default=0, help="parameter of the distribution of probability")
parser.add_argument("-T", type=float, default=0, help="temperature of the system")
parser.add_argument('--random', action='store_true', help="if random is true it do the random init simulation")
parser.add_argument('--d', action='store_true', help="if delta is true it do the delta simulation")

arguments = parser.parse_args()

if arguments.random:
    multiple_evolution = multiple_simulation_random(
        N=arguments.N,
        p=arguments.p,
        sweep_max=arguments.t,
        a=arguments.a,
        T=arguments.T,
        s=20,
        mixture=True,
        delta=arguments.d
    )
else:
    multiple_evolution = simulation_all_pattern_init(
        N=arguments.N,
        p=arguments.p,
        sweep_max=arguments.t,
        a=arguments.a,
        T=arguments.T,
        delta=arguments.d
    )

# multiple_evolution shape = (p, t, p + 1)

p=arguments.p

# reshape to (t, p * (p+1) ) so each column block is the evolution of one pattern
reshaped = multiple_evolution.transpose(1, 0, 2).reshape(arguments.t, -1)

# create header
header = []
for i in range(p):
    header += [f"m_{j+1}_from_{i+1}" for j in range(p)]
header_str = ",".join(header)

# save
np.savetxt(sys.stdout, reshaped, delimiter=",", header=header_str, fmt="%.5f")

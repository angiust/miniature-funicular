#!/usr/bin/env python3

import argparse
import sys

import numpy as np

from hopfield import simulation

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-N", type=int, default=1000, help="number of neurons")
parser.add_argument("-p", type=int, default=9, help="number of patterns")
parser.add_argument("-t", type=int, default=3000, help="number of sweeps")
parser.add_argument("-a", type=float, default=0, help="parameter of the distribution of probability")
parser.add_argument("-T", type=float, default=0, help="temperature of the system")
parser.add_argument('--mix', action='store_true', help="if mix is true it do the mixture simulation")

arguments = parser.parse_args()


multiple_evolution = multiple_simulation(
    N=arguments.N,
    p=arguments.p,
    sweep_max=arguments.t,
    a=arguments.a,
    T=arguments.T,
    s=arguments.s,
    mixture=arguments.mix
)

p=arguments.p
header = ",".join([f"m_{i+1}" for i in range(p)] + [f"std_{i+1}" for i in range(p)])
np.savetxt(sys.stdout, multiple_evolution, delimiter=",", fmt="%.5f", header=header)

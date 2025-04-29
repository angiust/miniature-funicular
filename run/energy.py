#!/usr/bin/env python3

import argparse
import sys
import os

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from hopfield import varying_a_energy, compute_energy

energy_table = compute_energy(
    N=1000,
    p=9,
    distrib_param=0,
    delta=False
)

energy_varying_a = varying_a_energy(
    N=1000,
    p=9,
    delta=False
)

print(energy_table)
print(energy_varying_a)

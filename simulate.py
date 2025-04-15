import argparse
import sys

import numpy as np

from hopfield import multiple_simulation # , simulation

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-N", type=int, default=1000, help="number of neurons")
parser.add_argument("-p", type=int, default=8, help="number of patterns")
parser.add_argument("-t", type=int, default=3000, help="number of sweeps")
parser.add_argument("-a", type=float, default=0, help="parameter of the distribution of probability")
parser.add_argument("-T", type=float, default=0, help="temperature of the system")
parser.add_argument("-s", type=int, default=20, help="number of samples")
parser.add_argument("--seed", type=int, default=None, help="random seed for reproducibility")

arguments = parser.parse_args()

if arguments.seed is not None:
    np.random.seed(arguments.seed)

"""
evolution = simulation(
    N=arguments.N,
    p=arguments.p,
    t_max=arguments.t,
    a=arguments.a,
    T=arguments.T
)
"""

multiple_evolution, standard_deviation = multiple_simulation(
    N=arguments.N,
    p=arguments.p,
    t_max=arguments.t,
    a=arguments.a,
    T=arguments.T,
    s=arguments.s
)

#np.savetxt(sys.stdout, evolution, fmt=('%.4f'))
np.savetxt(sys.stdout, multiple_evolution, fmt=('%.4f'))

print(standard_deviation)

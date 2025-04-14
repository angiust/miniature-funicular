import sys

import numpy as np

import hopfield

random_seed = 42
source_file = "expected.csv"


def check_consistency():
    np.random.seed(random_seed)  # Set the random seed for reproducibility
    expected = np.loadtxt(source_file, delimiter=',')[:,1]
    actual = hopfield.simulation(N=1000, p=8, t_max=3000, a=0, T=10)["magnetization"]
    max_discrepancy = np.max(np.abs(expected - actual))
    assert max_discrepancy < 0.001, f"some discrepancies found: {max_discrepancy}"
    print("all good", file=sys.stderr)


check_consistency()

import sys
import os

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import hopfield

random_seed = 42
source_file = "expected.csv"

"""
def check_consistency():
    np.random.seed(random_seed)  # Set the random seed for reproducibility
    expected = np.loadtxt(source_file, delimiter=',')[:,1]
    actual = hopfield.simulation(N=1000, p=8, t_max=3000, a=0, T=10)["magnetization"]
    max_discrepancy = np.max(np.abs(expected - actual))
    assert max_discrepancy < 0.001, f"some discrepancies found: {max_discrepancy}"
    print("all good", file=sys.stderr)


check_consistency()
"""

def test_combinations_of_identity():
    N = 1000
    p = 9
    n = 3
    patterns = hopfield.extract_pattern(N, p, 0.3, delta=True)

    mixtures_general = hopfield.compute_all_n_mixtures(patterns, n)
    mixtures_three = hopfield.compute_all_three_mixtures(patterns)

    assert np.array_equal(mixtures_general, mixtures_three), "Mismatch in mixture results for n=3"


test_combinations_of_identity()

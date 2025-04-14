import argparse
from hopfield import run_simulation

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-N", type=int, default=1000, help="number of neurons")
parser.add_argument("-p", type=int, default=8, help="number of patterns")
parser.add_argument("-t", type=int, default=3000, help="number of temporal steps")
parser.add_argument("-a", type=float, default=0, help="parameter of the distribution of probability")
parser.add_argument("-T", type=float, default=0, help="temperature of the system")
parser.add_argument("-s", type=int, default=20, help="number of samples")
# many more arguments

arguments = parser.parse_args()

print(f"And now you can play with these: {arguments}.")

run_simulation(
    N=arguments.N,
    p=arguments.p,
    t_max=arguments.t,
    a=arguments.a,
    T=arguments.T
)

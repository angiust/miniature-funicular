#!/usr/bin/env python3

import sys

import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-i", "--input", type=str, default="-", help="name of the input file ('-' for stdin)")
parser.add_argument("--title", type=str, default="", help="custom plot title")
parser.add_argument("--output", type=str, help="output filename if saving the plot")

arguments = parser.parse_args()

filename = arguments.input if arguments.input != '-' else sys.stdin
data = np.loadtxt(filename, delimiter=",")

# Extract time and magnetization
time = np.arange(data.shape[0])
average_magnetization = data[:, 0]
std_deviation = data[:, 1]

plt.figure(figsize=(8, 5))
plt.plot(time, average_magnetization, label='Magnetization', color='blue')
plt.fill_between(time, average_magnetization - std_deviation, average_magnetization + std_deviation, color='blue', alpha=0.2, label='Std dev')
plt.title(f"Magnetization vs Time at {arguments.title}")
plt.xlabel("Time Step")
plt.ylabel("Magnetization")
plt.grid(True)
plt.legend()
plt.tight_layout()

if arguments.output:
    plt.savefig(arguments.output)
else:
    plt.show()

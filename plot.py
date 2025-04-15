#!/usr/bin/env python3

import sys

import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-i", "--input", type=str, default="-", help="name of the input file ('-' for stdin)")
parser.add_argument("--title", type=str, default="", help="custom plot title")
parser.add_argument("--mode", type=str, choices=["show", "savefig"], default="show", help="whether to show or save the plot")
parser.add_argument("--output", type=str, default="plot.png", help="output filename if saving the plot")

arguments = parser.parse_args()

filename = arguments.input if arguments.input != '-' else sys.stdin
data = np.loadtxt(filename, delimiter=",")

# Extract time and magnetization
time = np.arange(data.shape[0])
average_magnetization = data[:, 0]
std_deviation = data[:, 1]
"""
plt.figure(figsize=(8, 5))
plt.errorbar(time, average_magnetization, yerr=std_deviation, color='blue', ecolor='lightgray', elinewidth=1, capsize=3)
plt.title(f"Magnetization vs Time {arguments.title}")
plt.xlabel("Time Step")
plt.ylabel("Magnetization")
plt.grid(True)
plt.tight_layout()
plt.show()
# Plot
plt.figure(figsize=(8, 5))
plt.plot(time, magnetization, linestyle='-', color='blue')
plt.title("Magnetization vs Time")
plt.xlabel("Time Step")
plt.ylabel("Magnetization")
plt.grid(True)
plt.tight_layout()
plt.show()
"""

plt.figure(figsize=(8, 5))
plt.plot(time, average_magnetization, label='Magnetization', color='blue')
plt.fill_between(time, average_magnetization - std_deviation, average_magnetization + std_deviation, color='blue', alpha=0.2, label='Std dev')
plt.title(f"Magnetization vs Time at {arguments.title}")
plt.xlabel("Time Step")
plt.ylabel("Magnetization")
plt.grid(True)
plt.legend()
plt.tight_layout()

if arguments.mode == "show":
    plt.show()
else:
    plt.savefig(arguments.output)

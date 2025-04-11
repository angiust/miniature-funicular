import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-n", type=int, default=1000, help="number of neurons")
parser.add_argument("-p", type=int, default=8, help="specification says it is a positive integer, but not what to do with this")
# many more arguments

arguments = parser.parse_args()

print(f"And now you can play with these: {arguments}.")

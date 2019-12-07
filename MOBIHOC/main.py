import random
from Role import Role
from game_generate import create_game
import argparse
import numpy as np
import copy
import math
from test_case import test


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", type=int, default=15, help="number of users")
    parser.add_argument("--edge", type=int, default=3, help="number of edge servers")
    parser.add_argument("--port", type=int, default=3389, help="controller port")
    parser.add_argument("--helper", type=int, default=10, help="number of helpers working on this controller")
    parser.add_argument("--run", type=int, default=1, help="number of iterations")
    parser.add_argument("--min_chs", type=int, default=10, help="number of sub-channels")
    parser.add_argument("--max_chs", type=int, default=30, help="number of sub-channels")
    parser.add_argument("--min_cpu", type=int, default=3, help="number of sub-channels")
    parser.add_argument("--max_cpu", type=int, default=6, help="number of sub-channels")
    parser.add_argument("--increment", type=int, default=1, help="number of sub-channels")
    parser.add_argument("--epsilon", type=float, default=0.0005, help="number of sub-channels")
    args = parser.parse_args()
    print(args.min_chs)
    iterations = args.run
    I = args.increment
    for iteration in range(iterations):
        player = create_game(args)
        for t in range(I):
            test(args, iteration, t, 0, False, clean_cache=True, channel_allocation=1, player=copy.deepcopy(player))

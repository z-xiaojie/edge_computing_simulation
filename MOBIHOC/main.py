import random
from Role import Role
from game_generate import create_game
import argparse
import numpy as np
import copy
import math
from _thread import *
import threading
from test_case import test
from Server import Controller


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", type=int, default=15, help="number of users")
    parser.add_argument("--edge", type=int, default=3, help="number of edge servers")
    parser.add_argument("--port", type=int, default=3389, help="controller port")
    parser.add_argument("--helper", type=int, default=10, help="number of helpers working on this controller")
    parser.add_argument("--run", type=int, default=20, help="number of iterations")
    parser.add_argument("--min_chs", type=int, default=10, help="number of sub-channels")
    parser.add_argument("--max_chs", type=int, default=20, help="number of sub-channels")
    parser.add_argument("--min_cpu", type=int, default=4, help="number of sub-channels")
    parser.add_argument("--max_cpu", type=int, default=8, help="number of sub-channels")
    parser.add_argument("--increment", type=int, default=1, help="number of sub-channels")
    parser.add_argument("--epsilon", type=float, default=0.0005, help="number of sub-channels")
    args = parser.parse_args()

    controller = Controller(selection=np.zeros(args.user).astype(int) - 1, opt_delta=np.zeros(args.user).astype(int) - 1)
    # start_new_thread(controller.run, (3389,))
    controller.run(port=3389)

    compare_energy = np.zeros(3)
    compare_data = np.zeros(3)
    compare_remote = np.zeros(3)

    iterations = args.run
    I = args.increment
    for iteration in range(iterations):
        player = create_game(args)
        for t in range(I):

            controller.selection = np.zeros(args.user).astype(int) - 1
            controller.opt_delta = np.zeros(args.user).astype(int) - 1

            controller.inital_config(copy.deepcopy(player), args.epsilon, priority="energy_reduction", clean_cache=True
                                     , channel_allocation=1, full=False)
            energy, data, remote = test(controller, args, iteration, t)
            compare_energy[0] += energy
            compare_data[0] += data
            compare_remote[0] += remote

            controller.selection = np.zeros(args.user).astype(int) - 1
            controller.opt_delta = np.zeros(args.user).astype(int) - 1

            controller.inital_config(copy.deepcopy(player), args.epsilon, priority="data", clean_cache=False
                                     , channel_allocation=1, full=False)
            energy, data, remote = test(controller, args, iteration, t)
            compare_energy[1] += energy
            compare_data[1] += data
            compare_remote[1] += remote

            controller.selection = np.zeros(args.user).astype(int) - 1
            controller.opt_delta = np.zeros(args.user).astype(int) - 1

            controller.inital_config(copy.deepcopy(player), args.epsilon, priority="gain", clean_cache=False
                                     , channel_allocation=1, full=False)
            energy, data, remote = test(controller, args, iteration, t)
            compare_energy[2] += energy
            compare_data[2] += data
            compare_remote[2] += remote

        print("compare_energy >>>>> ", compare_energy)
        print("compare_data >>>>> ", compare_data)
        print("compare_remote >>>>> ", compare_remote)

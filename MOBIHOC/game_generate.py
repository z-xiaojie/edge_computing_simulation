import random
import numpy as np
import math
from Role import Role


def create_game(args):
    number_of_user, number_of_edge, epsilon = args.user, args.edge, args.epsilon
    number_of_chs = np.array([random.randint(args.min_chs,  args.max_chs) for x in range(number_of_edge)])
    cpu = np.array([random.uniform(args.min_cpu,  args.max_cpu) * math.pow(10, 9) for x in range(number_of_edge)])
    H = [[round(np.random.rayleigh(np.sqrt(2 / np.pi) * math.pow(10, -3)), 5) for y in range(number_of_edge)] for x
         in range(number_of_user)]
    d_cpu = np.array([random.uniform(1.5, 2.5) * math.pow(10, 9) for x in range(number_of_user)])
    player = Role(number_of_edge=number_of_edge, number_of_user=number_of_user, epsilon=epsilon,
                  number_of_chs=number_of_chs, cpu=cpu, d_cpu=d_cpu, H=H)
    player.initial_DAG()
    # player.initial_config_DAG()
    return player


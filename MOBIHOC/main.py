import random
from Role import Role
from run import initial_energy_all_local
import numpy as np
import matplotlib.pyplot as plt
import copy
import math
from test_case import test
from Server import Controller
# import thread module
from _thread import *
import threading
from Server import Controller

"""

adaptive= [0.13594, 0.0795, 0.06432]

"""

iterations = 15
I = 5
hist = [[np.zeros(I) for i in range(20)] for j in range(3)]
selection1, selection2, selection3 = None, None, None
opt_delta1, opt_delta2 = None, None
bandwidth1, bandwidth2 = None, None
cpus = []
for i in range(iterations):
    number_of_user, number_of_edge, epsilon = 15, 3, 0.0005
    chs = 10
    t = 0
    #f = 1.25
    # number_of_chs = np.array([60, 60, 60])
    number_of_chs = np.array([random.randint(6, 12) for x in range(number_of_edge)])
    # cpu = np.array([4.54 * math.pow(10, 9), 4.5 * math.pow(10, 9), 5.26 * math.pow(10, 9)])
    cpu = np.array([random.uniform(3.5, 4.5) * math.pow(10, 9) for x in range(number_of_edge)])
    H = [[round(np.random.rayleigh(np.sqrt(2 / np.pi) * math.pow(10, -3)), 5) for y in range(number_of_edge)] for x in range(number_of_user)]
    # H = [[0.00053, 0.00029, 0.00191], [0.00197, 0.00036, 0.00178], [0.00175, 0.00088, 0.0014], [0.00119, 0.00032, 0.00162], [0.0004, 0.00109, 0.00119], [0.00035, 0.00164, 0.00116], [0.0001, 0.001, 0.00049], [0.00171, 0.0016, 0.00105], [0.00209, 0.00071, 0.00071], [0.00171, 0.00091, 0.00095], [0.00054, 0.00074, 0.00089], [0.00077, 0.00049, 0.00112], [0.00112, 0.00069, 0.00049], [0.00153, 0.00146, 0.00075], [0.00012, 0.00103, 0.00057]]
    d_cpu = np.array([random.uniform(1.5, 2.5) * math.pow(10, 9) for x in range(number_of_user)])
    player = Role(number_of_edge=number_of_edge, number_of_user=number_of_user, epsilon=epsilon,
                  number_of_chs=number_of_chs, cpu=cpu, d_cpu=d_cpu, H=H)
    player.initial_DAG()
    # player.initial_config_DAG()

    print("H=", H)
    while t < I:
        # number_of_chs = np.array([random.randint(16, 24) for x in range(number_of_edge)])
        for k in range(number_of_edge):
            player.edges[k].freq = cpu[k]
            player.edges[k].number_of_chs = number_of_chs[k]
            cpu[k] += 0.5 * math.pow(10, 9)
            number_of_chs[k] += 1
        it1, finish_hist1, bandwidth1, opt_delta1, selection1, finished1, energy1, local1, improvement1, local, remote, local_to_remote \
            = test(i, t, 0, False, clean_cache=True, channel_allocation=1, epsilon=epsilon, number_of_user=number_of_user, number_of_edge=number_of_edge
                                      ,player=copy.deepcopy(player))
        print("sub-channel", bandwidth1)
        print("partition", opt_delta1)

        hist[0][0][t] += finished1
        hist[0][1][t] += improvement1
        hist[0][2][t] += energy1
        hist[0][3][t] += local1
        hist[0][4][t] += it1
        hist[0][5][t] += local
        hist[0][6][t] += remote
        hist[0][7][t] += local_to_remote

        it2, finish_hist2, bandwidth2, opt_delta2, selection2, finished2, energy2, local2, improvement2, local, remote, local_to_remote \
            = test(i, t, 0, True, clean_cache=False, channel_allocation=1, epsilon=epsilon, number_of_user=number_of_user, number_of_edge=number_of_edge
                                         ,player=copy.deepcopy(player))

        print("sub-channel", bandwidth2)
        print("partition", opt_delta2)

        # if finished1 != 1 or finished2 != 1:
        #    print(">>>>>>>", np.sum(finished1), np.sum(finished2))
        #    # continue

        hist[1][0][t] += finished2
        hist[1][1][t] += improvement2
        hist[1][2][t] += energy2
        hist[1][3][t] += local2
        hist[1][5][t] += local
        hist[1][6][t] += remote
        hist[1][7][t] += local_to_remote

        t += 1

    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    print("adaptive=", list(hist[0][2] / (i + 1)))
    print("full=", list(hist[1][2] / (i + 1)))
    print("local=", list(hist[0][3] / (i + 1)))


print(selection1, "finished", hist[0][0]/iterations)
print(selection2, "finished", hist[1][0]/iterations)
print(">>>>>>>>>>> partition>>>>>>>>>>")
print(opt_delta1, "finished", hist[0][0]/iterations)
print(opt_delta2, "finished", hist[1][0]/iterations)
print(">>>>>>>>>>> bandwidth1>>>>>>>>>>")
print(bandwidth1, np.sum(bandwidth1))
print(bandwidth2, np.sum(bandwidth1))
print("adaptive=", list(hist[0][2]/iterations))
print("full=", list(hist[1][2]/iterations))
print("local=", list(hist[0][3]/iterations))
print(">>>>>>>>>>> local, remote, data>>>>>>>>>>")
print("adaptive=", list(hist[0][5]/iterations), list(hist[0][6]/iterations), list(hist[0][7]/iterations))
print("full=", list(hist[1][5]/iterations), list(hist[1][6]/iterations), list(hist[1][7]/iterations))
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
print("it=", list(hist[0][4]/iterations))
print("cpu=", cpu/math.pow(10, 9))
print("chs=", number_of_chs)
print("average improvement", list(hist[0][1]/iterations), list(hist[1][1]/iterations))

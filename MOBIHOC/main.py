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


iterations = 1
I = 4
hist = [[np.zeros(I) for i in range(20)] for j in range(3)]
selection1, selection2, selection3 = None, None, None
opt_delta1, opt_delta2 = None, None
bandwidth1, bandwidth2 = None, None
cpus = []
for i in range(iterations):
    number_of_user, number_of_edge, epsilon = 15, 3, 0.001
    chs = 10
    t = 0
    #f = 1.25
    number_of_chs = np.array([random.randint(8, 10) for x in range(number_of_edge)])
    cpu = np.array([random.uniform(4, 6) * math.pow(10, 9) for x in range(number_of_edge)])
    H = [[round(np.random.rayleigh(np.sqrt(2 / np.pi) * math.pow(10, -3)), 5) for y in range(number_of_edge)] for x in
         range(number_of_user)]
    d_cpu = np.array([random.uniform(1.5, 2.5) * math.pow(10, 9) for x in range(number_of_user)])
    player = Role(number_of_edge=number_of_edge, number_of_user=number_of_user, epsilon=epsilon,
                  number_of_chs=number_of_chs, cpu=cpu, d_cpu=d_cpu, H=H)
    player.initial_DAG()
    print("H=", H)

    while t < I:
        #number_of_chs = np.array([random.randint(16, 24) for x in range(number_of_edge)])
        for k in range(number_of_edge):
            player.edges[k].freq = cpu[k]
            cpu[k] += math.pow(10, 9)
        it1, finish_hist1, bandwidth1, opt_delta1, selection1, finished1, energy1, local, improvement1 \
            = test(1, False, channel_allocation=1, epsilon=epsilon, number_of_user=number_of_user, number_of_edge=number_of_edge
                                      ,player=copy.deepcopy(player))
        print(bandwidth1)
        hist[0][0][t] += finished1
        hist[0][1][t] += improvement1
        hist[0][2][t] += energy1
        hist[0][3][t] += local
        hist[0][4][t] += it1


        it2, finish_hist2, bandwidth2, opt_delta2, selection2, finished2, energy2, local, improvement2 \
            = test(1, True, channel_allocation=1, epsilon=epsilon, number_of_user=number_of_user, number_of_edge=number_of_edge
                                         ,player=copy.deepcopy(player))
        print(bandwidth2)

        # if finished1 != 1 or finished2 != 1:
        #    print(">>>>>>>", np.sum(finished1), np.sum(finished2))
        #    # continue

        hist[1][0][t] += finished2
        hist[1][1][t] += improvement2
        hist[1][2][t] += energy2
        hist[1][3][t] += local

        """
        selection3, finished, energy, local, improvement = test(model=2, epsilon=epsilon, number_of_user=number_of_user,
                                                                number_of_edge=number_of_edge
                                                                , player=copy.deepcopy(player), network=network,
                                                                cpu=cpu)
        hist[2][0][t] += finished
        hist[2][1][t] += improvement
        hist[2][2][t] += energy
        hist[2][3][t] += local
        """

        # plt.subplot(1, 2, 2)
        # plt.plot(np.array(finish_hist1)/number_of_user)

        #plt.subplot(2, 2, 4)
        #plt.plot(finish_hist2)
        # chs += 10
        # f += 0.5
        # number_of_user += 3
        t += 1

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
print("it=", list(hist[0][4]/iterations))
print("H=", H)
print("cpu=", cpu/math.pow(10, 9))
print("chs=", number_of_chs)
"""
print(selection1, "finished", hist[2][0]/iterations)
print("average improvement", hist[2][1]/iterations, hist[2][2]/iterations, hist[2][3]/iterations)
"""
# print("improvement", 1 - (hist[0][2]/iterations)/(hist[1][2]/iterations))#, 1 - (hist[0][2]/iterations)/hist[2][2]/iterations)

# plt.show()

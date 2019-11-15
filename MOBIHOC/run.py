import random
from worker import *
import numpy as np
import time
import math
# import thread module
from _thread import *
import threading
import copy
from concurrent.futures import ThreadPoolExecutor
from Server import Controller


def initial_energy_all_local(selection, player):
    energy, finished = 0, 0
    user_hist = [[] for n in range(player.number_of_user)]
    # print(selection)
    ee = []
    for n in range(player.number_of_user):
        player.users[n].local_only_execution()
        if selection[n] == -1:
            energy += player.users[n].local_only_energy
            ee.append(round(player.users[n].local_only_energy, 5))
            if player.users[n].local_only_enabled:
                finished += 1
            user_hist[n].append(player.users[n].local_only_energy)
    # print(selection, "finished", finished, "energy", energy)
    return ee, finished, user_hist


def energy_update(player, selection, user_hist, save=True):
    energy, finished, transmission, computation, edge_computation = [], [], [], [], []
    for n in range(player.number_of_user):
        if selection[n] == -1:
            if save:
                user_hist[n].append(round(player.users[n].local_only_energy, 5))
            energy.append(round(player.users[n].local_only_energy, 5))
            if player.users[n].local_only_enabled:
                finished.append(1)
            else:
                finished.append(0)
            transmission.append(0)
            computation.append(0)
            edge_computation.append(0)
        else:
            config = player.users[n].config
            k = selection[n]
            if config is not None:
                e, f, tt, ct, et = player.users[n].remote_execution()
                energy.append(round(e, 5))
                transmission.append(round(tt, 4))
                computation.append(round(ct, 4))
                edge_computation.append(round(et, 4))
                finished.append(f)
                if save:
                    user_hist[n].append(round(e, 5))
            else:
                transmission.append(0)
                computation.append(0)
                edge_computation.append(0)
                finished.append(0)
    # if np.sum(finished) == player.number_of_user:
    return user_hist, energy, finished, transmission, computation, edge_computation


def local_helper(id):
    from Client import Helper
    import socket
    helper = Helper("192.168.1.162", 0, 7)
    print("helper")
    while True:
        try:
            helper.connect()
            helper.optimize()
        except socket.error:
            print("no request, waiting...")


def get_request(x, current_t, opt_delta, channel_allocation, just_updated, player, selection, full, epsilon):
    # reset_request_pool(player.number_of_user)
    start = time.time()
    controller = Controller(current_t)
    controller.initial_info(player=player, selection=selection, opt_delta=opt_delta
                            , full=full, channel_allocation=channel_allocation, epsilon=epsilon)
    controller.reset_request_pool(player.number_of_user)
    # controller.optimize_locally(controller.info, [0, 1, 2, 3, 4, 5, 6, 7])
    controller.run(8080)
    # controller.notify_opt()
    print("waiting...", controller.finish)
    while not controller.check_worker([n for n in range(player.number_of_user)]):
        d = 1

    opt_delta = []
    for n in range(player.number_of_user):
        if player.users[n].config is not None:
            opt_delta.append(player.users[n].config[5])
        else:
            opt_delta.append(-1)
    print("request finished in >>>>>>>>>>>>>>>>", time.time() - start, selection, opt_delta)

    # print("request", controller.request)
    if x == 0:
        # ordered based on channel gain
        avg_H = []
        for n in range(player.number_of_user):
            if controller.request[n] is not None:
                avg_H.append([0, n])
                continue
            if controller.request[n] is not None and controller.request[n]['validation'] is not None:
                ip = player.users[n].local_only_energy - controller.request[n]['validation']['config'][0]
            else:
                if player.users[n].config is None:
                    ip = 0
                else:
                    ip = player.users[n].config[0] - player.users[n].local_only_energy
            avg_H.append([ip, n])
        avg_H = sorted(avg_H, key=lambda x:x[0], reverse=True)

        """
        req = [None for k in range(player.number_of_edge)]
        h_k = [0 for k in range(player.number_of_edge)]
        for n in range(player.number_of_user):
            if controller.request[n] is not None and controller.request[n]['validation'] is not None:
                k = controller.request[n]['validation']["edge"]
                if player.users[n].H[k] < h_k[k]:
                    req[k] = controller.request[n]
                    h_k[k] = player.users[n].H[k]
        not_tested_k = [k for k in range(player.number_of_edge)]
        while len(not_tested_k) > 0:
            k = random.choice(not_tested_k)
            if req[k] is not None:
                return [req[k]]
            else:
                not_tested_k.remove(k)
        for item in controller.request:
            if item is not None and item["local"]:
                return [item]
        """

        not_tested = [avg_H[n][1] for n in range(player.number_of_user)]
        print("not_tested", not_tested)
        n = 0
        while len(not_tested) > n:
            if controller.request[not_tested[n]] is not None:
                return [controller.request[not_tested[n]]] # get_requests(controller.request, controller.request[not_tested[n]], selection)
            else:
                n += 1
        return None
    else:
        not_tested = [n for n in range(player.number_of_user)]
        print("not_tested", not_tested)
        while len(not_tested) > 0:
            n = random.choice(not_tested)
            if controller.request[n] is not None:
                return controller.request[n] #get_requests(controller.request, controller.request[n], selection)
            else:
                not_tested.remove(n)
        return None

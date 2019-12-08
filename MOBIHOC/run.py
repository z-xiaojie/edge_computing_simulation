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
            transmission.append(player.users[n].DAG.D / 1000)
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
                computation.append(player.users[n].DAG.D / 1000)
                edge_computation.append(0)
                finished.append(0)
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


def get_request(controller, current_t):
    # reset_request_pool(player.number_of_user)
    for n in range(controller.player.number_of_user):
        controller.player.users[n].partition()

    # start = time.time()
    controller.reset_request_pool(controller.player.number_of_user)
    controller.initial_info(player=copy.deepcopy(controller.player), current_t=current_t)
    controller.notify_all()
    # controller.optimize_locally(controller.info, [n for n in range(args.helper)])
    # start_new_thread(controller.run, (3389,))
    # controller.run(3389)
    # print("waiting...", controller.finish)
    while not controller.check_worker([n for n in range(controller.player.number_of_user)]):
        pass

    opt_delta = []
    for n in range(controller.player.number_of_user):
        if controller.player.users[n].config is not None:
            opt_delta.append(controller.player.users[n].config[5])
        else:
            opt_delta.append(-1)
    # print("\t + req get in >>>>>>>>>>", round(time.time() - start), controller.selection, opt_delta)

    print("request", controller.request)
    if controller.priority == "energy_reduction":
        # ordered based on channel gain
        avg_H = []
        for n in range(controller.player.number_of_user):
            if controller.request[n] is None:
                avg_H.append([0., n])
                continue
            if controller.request[n] is not None and controller.request[n]['validation'] is not None:
                ip = controller.player.users[n].local_only_energy - controller.request[n]['validation']['config'][0]
                avg_H.append([ip, n])
            else:
                avg_H.append([0., n])
        avg_H = sorted(avg_H, key=lambda x:x[0], reverse=True)
        not_tested = [avg_H[n][1] for n in range(controller.player.number_of_user)]
        # print("not_tested", not_tested)
        n = 0
        while len(not_tested) > n:
            if controller.request[not_tested[n]] is not None:
                return [controller.request[not_tested[n]]]
                # get_requests(controller.request, controller.request[not_tested[n]], selection)
            else:
                n += 1
        return None
    else:
        not_tested = [n for n in range(controller.player.number_of_user)]
        # print("not_tested", not_tested)
        # n = 0
        while len(not_tested) > 0:
            n = random.choice(not_tested)
            if controller.request[n] is not None:
                # [controller.request[n]]
                return [controller.request[n]] # get_requests(controller.request, controller.request[n], controller.selection)
            else:
                not_tested.remove(n)
                # n += 1
        return None

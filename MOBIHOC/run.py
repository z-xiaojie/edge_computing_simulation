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


def get_request(current_t, opt_delta, channel_allocation, just_updated, player, selection, full, epsilon):
    # reset_request_pool(player.number_of_user)
    start = time.time()
    controller = Controller(current_t)
    controller.initial_info(player=player, selection=selection, opt_delta=opt_delta
                            , full=full, channel_allocation=channel_allocation, epsilon=epsilon)
    controller.reset_request_pool(player.number_of_user)
    controller.optimize_locally(controller.info, [0, 1, 2, 3, 4, 5])
    controller.run(12345)
    # controller.notify_opt()
    print("waiting...", controller.finish)
    while not controller.check_worker([n for n in range(player.number_of_user)]):
        d = 1

    """
    info = controller.info
    copied_info = []
    for n in range(player.number_of_user):
        info["who"] = player.users[n]
        copied_info.append(copy.deepcopy(info))
    
    with ThreadPoolExecutor(max_workers=player.number_of_user) as executor:
        executor.map(worker, copied_info)

    for n in range(player.number_of_user):
        # 为每个worker创建一个线程
        info["who"] = player.users[n]
        # x = threading.Thread(target=worker, args=(copy.deepcopy(info),))
        # x.start()
        # worker(copy.deepcopy(info))
    """

    opt_delta = []
    for n in range(player.number_of_user):
        if player.users[n].config is not None:
            opt_delta.append(player.users[n].config[5])
        else:
            opt_delta.append(-1)

    print("request finished in >>>>>>>>>>>>>>>>", time.time() - start, selection, opt_delta)
    # print("request", controller.request)

    not_tested = [n for n in range(player.number_of_user)]
    n = 0
    while len(not_tested) > 0:
        # n = random.choice(not_tested)
        #if n == just_updated:
            #continue
        if controller.request[n] is not None:
            return get_requests(controller.request, controller.request[n], selection)
        else:
            not_tested.remove(n)
            n += 1
    return None

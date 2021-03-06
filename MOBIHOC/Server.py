# import socket programming library
import socket
import math
import numpy as np
from threading import Lock
from _thread import *
from Client import worker, create_state, check_worker
import threading
import json
import time
import traceback
from Device import Device
from Optimization import Optimization
import copy
import struct
from multiprocessing import Process, Manager

lock = Lock()


class Controller(threading.Thread, Optimization):
    def __init__(self, selection, opt_delta):
        # self.print_lock = threading.Lock()
        self.clean_cache = None
        self.current_t = None
        self.player = None
        self.selection = selection
        self.opt_delta = opt_delta
        self.full = None
        self.channel_allocation = None
        self.epsilon = None
        self.info = None
        self.s = None
        self.c = []
        self.cache = []
        # self.initial_info()
        self.request = None
        self.finish = None
        self.lock = Lock()
        self.priority = None
        self.number_of_opt = 0
        self.number_of_finished_opt = 0
        self.validation = None

    def reset_request_pool(self, number_of_user):
        self.request = [None for n in range(number_of_user)]
        self.finish = [0 for n in range(number_of_user)]

    def check_worker(self, doing):
        for n in doing:
            if self.finish[n] != 1:
                return False
        return True

    def inital_config(self, player, epsilon, priority="energy_reduction", clean_cache=True, channel_allocation=1, full=False):
        self.player = player
        self.full = full
        self.priority = priority
        self.channel_allocation = channel_allocation
        self.epsilon = epsilon
        self.clean_cache = clean_cache

    def initial_info(self, player=None, current_t=None):
        self.current_t = current_t
        job_list = []
        for n in range(self.player.number_of_user):
            self.player.users[n].partition()
            job_list.append(self.player.users[n].DAG.display())

        D_n = np.array([self.player.users[n].DAG.D / 1000 for n in range(self.player.number_of_user)]).tolist()
        X_n = np.array([self.player.users[n].remote for n in range(self.player.number_of_user)]).tolist()
        Y_n = np.array([self.player.users[n].local for n in range(self.player.number_of_user)]).tolist()
        user_cpu = np.array([self.player.users[n].freq for n in range(self.player.number_of_user)]).tolist()
        edge_cpu = np.array([self.player.edges[k].freq for k in range(self.player.number_of_edge)]).tolist()
        number_of_chs = np.array([self.player.edges[k].number_of_chs for k in range(self.player.number_of_edge)]).tolist()
        P_max = np.array([self.player.users[n].p_max for n in range(self.player.number_of_user)]).tolist()
        B = np.array([self.player.users[n].local_to_remote_size for n in range(self.player.number_of_user)]).tolist()
        H = np.array(
            [[self.player.users[n].H[k] for k in range(self.player.number_of_edge)] for n in range(self.player.number_of_user)]).tolist()
        configs = [self.player.users[n].config for n in range(self.player.number_of_user)]

        self.info = {
            "current_t": self.current_t,
            "clean_cache": self.clean_cache,
            "configs": configs,
            "tasks": job_list,
            "local_only_enabled": [item.local_only_enabled for item in player.users],
            "local_only_energy": [item.local_only_energy for item in player.users],
            "opt_delta": self.opt_delta.tolist(),
            "selection": self.selection.tolist(),
            "number_of_edge": self.player.number_of_edge,
            "number_of_user": self.player.number_of_user,
            "D_n": D_n,
            "X_n": X_n,
            "Y_n": Y_n,
            "user_cpu": user_cpu,
            "edge_cpu": edge_cpu,
            "number_of_chs": number_of_chs,
            "P_max": P_max,
            "B": B,
            "H": H,
            "W": 2 * math.pow(10, 6),
            "who": None,
            "full": self.full,
            "default_channel": 1,
            "channel_allocation": self.channel_allocation,
            "step": 0.005,
            "interval": 10,
            "stop_point": self.epsilon
        }

    def optimize_locally(self, info, doing):
        state = create_state(doing, info, self.cache)
        processes = list()
        for n in doing:
            info["who"] = Device(info["user_cpu"][n], n, info["H"][n]
                                 , transmission_power=info["P_max"][n], epsilon=info["stop_point"])
            info["who"].inital_DAG(n, info["tasks"][n], info["D_n"][n], info["D_n"][n])
            info["who"].local_only_execution()
            info["who"].config = info["configs"][n]
            x = Process(target=worker, args=(copy.deepcopy(info), state))
            x.start()
            processes.append(x)
        for process in processes:
            process.join()

    def run(self, port=12345):
        host = ""
        port = port
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.bind((host, port))
        self.s.listen(5)
        while True:
            try:
                c, addr = self.s.accept()
                self.c.append(c)
                start_new_thread(self.client_threaded, (c,))
                print("connected to ", addr)
                if len(self.c) == 2:
                    break
            except Exception as e:
                break

    def close(self):
        for c in self.c:
            try:
                message = "close"
                self.send_msg(c, message.encode('ascii'))
                c.close()
            except socket.error:
                pass

    def send_msg(self, c, msg):
        # Prefix each message with a 4-byte length (network byte order)
        msg = struct.pack('>I', len(msg)) + msg
        c.sendall(msg)

    def notify_all(self):
        # print("sending computing requests...")
        for c in self.c:
            try:
                self.info["who"] = None
                json_data = json.dumps(self.info).encode("ascii")
                self.send_msg(c, json_data)
            except socket.error:
                pass
        # print("sending computing requests...Done!")

    def client_threaded(self, c):
        message = "start_opt"
        self.send_msg(c, message.encode('ascii'))
        while True:
            try:
                data = c.recv(5024)
                str_data = str(data.decode('ascii'))
                if str_data[0] != 'm':
                    result = json.loads(str(data.decode('ascii')))
                    request = result["req"]
                    doing = result["doing"]
                    # print("request", request)
                    # message = "waiting"
                    # self.send_msg(c, message.encode('ascii'))
                    self.lock.acquire()
                    for n in doing:
                        self.request[n] = request[n]
                        self.finish[n] = 1
                    self.lock.release()
            except:
                pass

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
from Device import Device
from Optimization import Optimization
import copy
import struct
from multiprocessing import Process, Manager

lock = Lock()


class Controller(threading.Thread, Optimization):
    def __init__(self, clean_cache, current_t):
        # self.print_lock = threading.Lock()
        self.clean_cache = clean_cache
        self.current_t = current_t
        self.player = None
        self.selection = None
        self.opt_delta = None
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

    def initial_info(self, player=None, selection=None, opt_delta=None, full=None, channel_allocation=None, epsilon=None):
        self.player = player
        self.selection = selection
        self.opt_delta = opt_delta
        self.full = full
        self.channel_allocation = channel_allocation
        self.epsilon = epsilon
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
            "interval": 5,
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
            except Exception as e:
                print(e.__str__())
                break

    def close(self):
        for c in self.c:
            try:
                message = "close"
                self.send_msg(c, message.encode('ascii'))
                c.close()
            except socket.error:
                continue
        self.s.close()

    def notify_opt(self):
        try:
            for c in self.c:
                self.info["who"] = None
                json_data = json.dumps(self.info).encode("utf-8")
                c.send(json_data)
        except socket.error:
            return

    def send_msg(self, c, msg):
        # Prefix each message with a 4-byte length (network byte order)
        msg = struct.pack('>I', len(msg)) + msg
        c.sendall(msg)

    def client_threaded(self, c):
        message = "start_opt"
        self.send_msg(c, message.encode('ascii'))
        self.info["who"] = None
        json_data = json.dumps(self.info).encode("ascii")
        self.send_msg(c, json_data)
        while True:
            try:
                data = c.recv(5024)
                str_data = str(data.decode('ascii'))
                if str_data[0] != 'm':
                    result = json.loads(str(data.decode('ascii')))
                    request = result["req"]
                    doing = result["doing"]
                    # print("request", request)
                    message = "waiting"
                    self.send_msg(c, message.encode('ascii'))
                    self.lock.acquire()
                    for n in doing:
                        self.request[n] = request[n]
                        self.finish[n] = 1
                    self.lock.release()
                    while not self.check_worker([n for n in range(self.player.number_of_user)]):
                        pass
                    self.close()
            except socket.error:
                self.close()
                return

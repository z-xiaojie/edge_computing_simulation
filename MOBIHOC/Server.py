# import socket programming library
import socket
import math
import numpy as np
# import thread module
from _thread import *
from Client import Helper
import threading
import json
from Offloading_Mobihoc import Offloading
import time
from Device import Device
from Optimization import opt
import copy
import struct


class Controller(threading.Thread):
    def __init__(self, current_t):
        # self.print_lock = threading.Lock()
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
            "configs": configs,
            "tasks": job_list,
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

    def search_cache(self, info, edge_id):
        for s, d, config, task_id, k in self.cache:
            if task_id != info["who"].task_id and k != edge_id:
                continue
            same = True
            selected = []
            delta = []
            for n in range(len(info["selection"])):
                if info["selection"][n] == edge_id:
                    selected.append(n)
                    delta.append(info["opt_delta"][n])
            if s != selected or d != delta:
                same = False
            if same:
                return config
        return None

    def opt(self, info):
        target = info["who"]
        validation = []
        if not info["full"]:
            feasible = [0]
            small_data = target.DAG.jobs[0].input_data
            for delta in range(1, len(target.DAG.jobs) - 1):
                if target.DAG.jobs[delta].input_data < small_data:
                    feasible.append(delta)
                    small_data = target.DAG.jobs[delta].input_data
                else:
                    break
            for delta in feasible:
                info["Y_n"][target.task_id], info["X_n"][target.task_id], info["B"][target.task_id] = 0, 0, 0
                for m in range(0, delta):
                    info["Y_n"][target.task_id] += target.DAG.jobs[m].computation
                for m in range(delta, target.DAG.length):
                    info["X_n"][target.task_id] += target.DAG.jobs[m].computation
                if delta == 0:
                    info["B"][target.task_id] = target.DAG.jobs[delta].input_data
                else:
                    info["B"][target.task_id] = target.DAG.jobs[delta - 1].output_data
                for k in range(info["number_of_edge"]):
                    optimize = Offloading(info, k)
                    info["selection"][target.task_id] = k
                    info["opt_delta"][target.task_id] = delta
                    config = self.search_cache(info, k)
                    if config is None:
                        config = optimize.start_optimize(delta=delta)
                        self.cache.append((copy.copy(info["selection"]), copy.copy(info["opt_delta"]), config, target.task_id, k ))
                    else:
                        print("read from cache: get user info", target.task_id)
                    if config is not None and (config[0] < target.local_only_energy or not target.local_only_enabled):
                        validation.append({
                            "edge": k,
                            "config": config
                        })
        else:
            delta = 0
            info["Y_n"][target.task_id], info["X_n"][target.task_id], info["B"][target.task_id] = 0, 0, 0
            for m in range(0, target.DAG.length):
                info["X_n"][target.task_id] += target.DAG.jobs[m].computation
            info["B"][target.task_id] = target.DAG.jobs[delta].input_data
            for k in range(info["number_of_edge"]):
                optimize = Offloading(info, k)
                config = optimize.start_optimize(delta=delta)
                if config is not None and (config[0] < target.local_only_energy or not target.local_only_enabled):
                    validation.append({
                        "edge": k,
                        "config": config
                    })
        return validation, target

    def worker(self, info):
        # start = time.time()
        validation, target = self.opt(info)
        if len(validation) > 0:
            validation.sort(key=lambda x: x["config"][0])
            if validation[0]["edge"] != info["selection"][target.task_id] \
                    or validation[0]["config"] != target.config:
                self.request[target.task_id] = {
                    "user": target.task_id,
                    "validation": validation[0],
                    "local": False
                }
        else:
            if info["selection"][target.task_id] != -1:
                self.request[target.task_id] = {
                    "user": target.task_id,
                    "validation": None,
                    "local": True
                }
        self.finish[target.task_id] = 1

    def optimize_locally(self, info, doing):
        for n in doing:
            # 为每个worker创建一个线程
            info["who"] = Device(info["user_cpu"][n], n, info["H"][n]
                                 , transmission_power=info["P_max"][n], epsilon=info["stop_point"])
            info["who"].inital_DAG(n, info["tasks"][n], info["D_n"][n], info["D_n"][n])
            info["who"].local_only_execution()
            info["who"].config = info["configs"][n]
            x = threading.Thread(target=self.worker, args=(copy.deepcopy(info),))
            x.start()

    def run(self, port=12345):
        host = ""
        port = port
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.bind((host, port))
        # print("socket binded to port", port)
        # put the socket into listening mode
        self.s.listen(5)
        # print("socket is listening")
        # a forever loop until client wants to exit
        while True:
            # establish connection with client
            try:
                c, addr = self.s.accept()
                # lock acquired by client
                # self.print_lock.acquire()
                self.c.append(c)
                # print('Connected to :', addr[0], ':', addr[1])
                # Start a new thread and return its identifier
                start_new_thread(self.client_threaded, (c,))
            except:
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
                    for n in doing:
                        self.request[n] = request[n]
                        self.finish[n] = 1
                    message = "waiting"
                    self.send_msg(c, message.encode('ascii'))
                    if self.check_worker([n for n in range(self.player.number_of_user)]):
                        self.close()
                else:
                    #print(str_data)
                    ddd = 1
            except socket.error:
                return

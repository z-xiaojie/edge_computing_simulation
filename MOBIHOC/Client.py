# Import socket module
import socket
import json
from Device import Device
import threading
import copy
from threading import Lock
from Offloading_Mobihoc import *
from Optimization import Optimization
import time
import struct
from multiprocessing import Process, Manager

lock = Lock()


def reset_request_pool(number_of_user):
    request = [{
                "user": None,
                "validation": None,
                "local": False
            } for n in range(number_of_user)]
    finish = [0 for n in range(number_of_user)]


def search_cache(info, user_id, edge_id, cache):
    for s, d, config, task_id, k in cache:
        if task_id != user_id or k != edge_id:
            continue
        selected = []
        delta = []
        for n in range(len(info["selection"])):
            if info["selection"][n] == edge_id:
                selected.append(n)
                delta.append(info["opt_delta"][n])
        if np.array_equal(s, selected) and np.array_equal(delta, d):
            return config
    return None


def energy_opt(info, delta, state, small_config):
    target = info["who"]
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
        lock.acquire()
        config = search_cache(info, target.task_id, k, state['cache'])
        lock.release()
        save = False
        if config is None:
            config = optimize.start_optimize(delta=delta,
                                             local_only_energy=info["local_only_energy"][target.task_id])
            save = True
        else:
            # print("read from cached times", target.task_id, "edge=", k, "delta=", delta)
            d = 1
        if config is not None and (
                config[0] < info["local_only_energy"][target.task_id] or not info["local_only_enabled"][target.task_id])\
                and (small_config is None or small_config["config"][0] > config[0]):
            small_config = {
                "edge": k,
                "config": copy.copy(config)
            }
            """
            state["validation"][target.task_id].append({
                "edge": k,
                "config": config
            })
            """
            lock.acquire()
            # print("user", target.task_id, "delta", delta, ">>>>>>>>", small_config)
            if save:
                selected = []
                partition_delta = []
                for n in range(info["number_of_user"]):
                    if info["selection"][n] == k:
                        selected.append(n)
                        partition_delta.append(info["opt_delta"][n])
                state['cache'].append(
                    (selected, partition_delta, config, target.task_id, k))
            lock.release()
    lock.acquire()
    state["number_of_finished_opt"] += 1
    # print("validation", state["validation"])
    lock.release()
    return small_config


def worker(info, state):
    target = info["who"]
    feasible = [0]
    if not info["full"]:
        small_data = target.DAG.jobs[0].input_data
        for delta in range(1, len(target.DAG.jobs) - 1):
            if target.DAG.jobs[delta].input_data < small_data:
                feasible.append(delta)
                small_data = target.DAG.jobs[delta].input_data
            else:
                break
    # print(feasible)
    small_config = None
    for delta in feasible:
        state["number_of_opt"] += 1
        small_config = energy_opt(copy.deepcopy(info), delta, state, small_config)
        # x = threading.Thread(target=energy_opt, args=(copy.deepcopy(info), delta, target.task_id))
        # x.start()
    lock.acquire()
    if small_config is not None:
        state["validation"][target.task_id].append(small_config)
    if len(state["validation"][target.task_id]) > 0:
        # state["validation"][target.task_id].sort(key=lambda x: x["config"][0])
        # print(state["validation"][target.task_id])
        if state["validation"][target.task_id][0]["edge"] != info["selection"][target.task_id] \
                or state["validation"][target.task_id][0]["config"] != target.config:
            state["request"][target.task_id] = {
                "user": target.task_id,
                "validation": state["validation"][target.task_id][0],
                "local": False
            }
    else:
        if info["selection"][target.task_id] != -1:
            state["request"][target.task_id] = {
                "user": target.task_id,
                "validation": None,
                "local": True
            }
    state['finish'][target.task_id] = 1
    lock.release()


class Helper(Optimization):
    def __init__(self, host, port, start, end):
        self.host = host
        start1 = int(start)
        end1 = int(end)
        self.doing = [n for n in range(start1, end1+1)]
        self.port = int(port)
        self.done = False
        self.cache = []

        self.number_of_opt = 0
        self.number_of_finished_opt = 0
        self.validation = None
        self.lock = Lock()

    def connect(self):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.connect((self.host, self.port))
        self.request = None
        self.finish = None

    def check_worker(self, doing, state):
        for n in doing:
            if state['finish'][n] != 1:
                return False
        return True

    def recv_msg(self):
        # Read message length and unpack it into an integer
        raw_msglen = self.recvall(4)
        if not raw_msglen:
            return None
        msglen = struct.unpack('>I', raw_msglen)[0]
        # Read the message data
        return self.recvall(msglen)

    def recvall(self, n):
        # Helper function to recv n bytes or return None if EOF is hit
        data = bytearray()
        while len(data) < n:
            packet = self.s.recv(n - len(data))
            if not packet:
                return None
            data.extend(packet)
        return data

    def create_state(self, doing, info, cache):
        # multiprocessing POST
        manager = Manager()
        state = manager.dict()
        state['number_of_opt'] = 0
        state['number_of_finished_opt'] = 0
        state['cache'] = manager.list()
        for item in cache:
            state['cache'].append(item)
        state['finish'] = manager.list([0 for n in range(info['number_of_user'])])
        state['request'] = manager.list()
        state['validation'] = manager.list()
        for n in range(info['number_of_user']):
            if doing.__contains__(n):
                state['request'].append(manager.list())
                state['validation'].append(manager.list())
            else:
                state['request'].append(None)
                state['validation'].append(None)
        return state

    def optimize(self):
        global validation, finish
        message = "m: greeting from edge server, doing list " + str(self.doing)
        self.s.send(message.encode('ascii'))
        start = time.time()
        while True:
            data = self.recv_msg()
            if str(data.decode('ascii')) == "start_opt":
                print("waiting data...")
                self.done = False
            elif str(data.decode('ascii')) == "waiting":
                print("no request...")
            elif str(data.decode('ascii')) == "close":
                print("done, close connection")
                self.s.close()
                return
            elif not self.done:
                info = json.loads(str(data.decode('ascii')))
                print("start to optimizing...", self.doing, info["number_of_user"])
                if info["clean_cache"] is True and info["current_t"] == 0:
                    self.cache = []
                    print("clean cache.........................................................")
                state = self.create_state(self.doing, info, self.cache)
                processes = list()
                for n in self.doing:
                    info["who"] = Device(info["user_cpu"][n], n, info["H"][n]
                                         , transmission_power=info["P_max"][n], epsilon=info["stop_point"])
                    info["who"].inital_DAG(n, info["tasks"][n], info["D_n"][n], info["D_n"][n])
                    info["who"].local_only_execution()
                    info["who"].config = info["configs"][n]
                    # x = threading.Thread(target=self.worker, args=(copy.deepcopy(info),))
                    x = Process(target=worker, args=(copy.deepcopy(info), state))
                    x.start()
                    processes.append(x)
                for process in processes:
                    process.join()
                while not self.check_worker(self.doing, state):
                    # print("finish", state['finish'])
                    dd = 0
                print(info["current_t"], "request finished in >>>>>>>>>>>>>>>>", time.time() - start)
                self.cache = copy.copy(list(state['cache']))
                for n in self.doing:
                    if state['request'][n] is not None:
                        if len(state['request'][n]) == 0:
                            state['request'][n] = None
                            print("update request for user", n, "= None")
                        else:
                            if state['request'][n]["validation"] is not None:
                                print("update request for user", n, "=", state['request'][n]["validation"]["config"])
                            else:
                                print("update request for user", n, "= local")
                    else:
                        print("update request for user", n, "= None")
                self.s.send(json.dumps({"req": list(state['request']), "doing": self.doing}).encode("utf-8"))
                self.done = True






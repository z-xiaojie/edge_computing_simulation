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


class Helper(Optimization):
    def __init__(self, host, start, end):
        self.host = host
        start1 = int(start)
        end1 = int(end)
        self.doing = [n for n in range(start1, end1+1)]
        self.port = 12345
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

    def reset_request_pool(self, number_of_user):
        self.request = [None for n in range(number_of_user)]
        self.finish = [0 for n in range(number_of_user)]

    def worker(self, info):
        validation, target = self.opt(copy.deepcopy(info))
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

    def check_worker(self, doing):
        for n in doing:
            if self.finish[n] != 1:
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

    def optimize(self):
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
                print("start to optimizing...")
                info = json.loads(str(data.decode('ascii')))
                if info["current_t"] == 0:
                    self.clean_cache()
                self.reset_request_pool(info["number_of_user"])
                self.validation = [[] for n in range(info["number_of_user"])]
                for n in self.doing:
                    info["who"] = Device(info["user_cpu"][n], n, info["H"][n]
                                         , transmission_power=info["P_max"][n], epsilon=info["stop_point"])
                    info["who"].inital_DAG(n, info["tasks"][n], info["D_n"][n], info["D_n"][n])
                    info["who"].local_only_execution()
                    info["who"].config = info["configs"][n]
                    x = threading.Thread(target=self.worker, args=(copy.deepcopy(info),))
                    x.start()
                while not self.check_worker(self.doing):
                    # print("working....")
                    dd = 0
                print(info["current_t"], "request finished in >>>>>>>>>>>>>>>>", time.time() - start)
                for n in self.doing:
                    if self.request[n] is not None:
                        if self.request[n]["validation"] is not None:
                            print("update request for user", n, "=", self.request[n]["validation"]["config"])
                        else:
                            print("update request for user", n, "= local")
                    else:
                        print("update request for user", n, "= None")
                self.s.send(json.dumps({"req": self.request, "doing": self.doing}).encode("utf-8"))
                self.done = True






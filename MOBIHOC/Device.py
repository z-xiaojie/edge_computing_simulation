from DAG import DAG
import random
import math
import json
import numpy as np
import matplotlib.pyplot as plt


class Device:
    def __init__(self, frequency, task_id, H=None, transmission_power=0.5, epsilon=0.01):
        self.freq = round(frequency)
        self.task_id = task_id
        self.DAG = None
        self.p_max = round(transmission_power, 3)
        self.epsilon = epsilon
        # local information
        self.k = math.pow(10, -28)
        self.N_0 = math.pow(10, -9)
        self.W = 1 * math.pow(10, 6)

        self.H = H
        self.preference = None

        self.alpha = 0.5

        # local and remote after partition
        self.delta = 0
        self.config = None

        self.local = 0
        self.remote = 0
        self.remote_deadline = 0
        self.local_to_remote_size = 0
        self.local_only_energy = 0

        self.local_only_enabled = True

        self.total_computation = 0

        # job queue
        self.queue = []

    """
       Given network and computation resource from the edge node, select optimal DAG partition
    """
    def select_partition(self, full, edge, epsilon=0.0001, p_adjust=0.9, default_channel=5):
        validation = []
        edge.clean_history()
        if not full:
            # print("user", self.task_id)
            feasible = [0]
            small_data = self.DAG.jobs[0].input_data
            for delta in range(1, len(self.DAG.jobs) - 1):
                if self.DAG.jobs[delta].input_data < small_data:
                    feasible.append(delta)
                    small_data = self.DAG.jobs[delta].input_data
                else:
                    break
            # print("FFFFFFFFFF", feasible, len(self.DAG.jobs))
            for delta in feasible:
                self.local, self.remote, data = 0, 0, self.DAG.jobs[delta].output_data
                for m in range(0, delta):
                    self.local += self.DAG.jobs[m].computation
                for m in range(delta, self.DAG.length):
                    self.remote += self.DAG.jobs[m].computation
                if delta == 0:
                    self.local_to_remote_size = self.DAG.jobs[delta].input_data
                else:
                    self.local_to_remote_size = self.DAG.jobs[delta - 1].output_data
                config = edge.resource_allocation(delta, who=self, epsilon=epsilon, p_adjust=p_adjust, default_channel=default_channel)
                if config is not None and (config[0] < self.local_only_energy or not self.local_only_enabled):
                    validation.append(config)
        else:
            delta = 0
            self.local, self.remote = 0, 0
            for m in range(0, self.DAG.length):
                self.remote += self.DAG.jobs[m].computation
            self.local_to_remote_size = self.DAG.jobs[delta].input_data
            config = edge.resource_allocation(delta, who=self, epsilon=epsilon, p_adjust=p_adjust, default_channel=default_channel)
            if config is not None and (config[0] < self.local_only_energy or not self.local_only_enabled):
                validation.append(config)
        if len(validation) > 0:
            validation.sort(key=lambda x: x[0])
            # print(self.task_id, "set config", validation)
            # self.delta = validation[0][0]
            return validation[0]
        else:
            # print(self.task_id, "run task locally", self.local_only_energy)
            return None

    def remote_execution(self):
        if self.config is None:
            return -1, 0

        f_e = self.config[2]
        bw = self.config[4]
        cpu = self.config[1]
        power = self.config[3]
        delta = self.config[5]
        edge_id = self.config[6]

        self.local, self.remote = 0, 0
        for m in range(0, delta):
            self.local += self.DAG.jobs[m].computation
        for m in range(delta, self.DAG.length):
            self.remote += self.DAG.jobs[m].computation

        if delta == 0:
            self.local_to_remote_size = self.DAG.jobs[delta].input_data
            computation_time = 0
        else:
            self.local_to_remote_size = self.DAG.jobs[delta - 1].output_data
            computation_time = self.local / cpu

        rate = bw * self.W * math.log2(1 + power * math.pow(self.H[edge_id], 2) / math.pow(10, -9))
        t = self.local_to_remote_size / rate

        # minimal energy
        a = self.k * self.local * math.pow(cpu, 2)
        energy = a + t * power * bw

        diff = math.fabs(t + self.remote/f_e + computation_time - self.DAG.D/1000)

        if diff <= self.epsilon:
            return energy, 1, t, computation_time, self.remote/f_e
        else:
           print(self.task_id, ">>>>>", diff, "rate", self.H[edge_id], "power", power, "cpu", cpu, "fe", f_e, "chs", bw)
           return energy, 0, t, computation_time, self.remote/f_e

    def local_only_execution(self):
        total_computation = 0
        for m in range(self.DAG.length):
            total_computation += self.DAG.jobs[m].computation
        self.total_computation = total_computation / math.pow(10, 9)
        cpu = 1000 * total_computation / self.DAG.D
        if cpu > self.freq:
            self.local_only_enabled = False
        self.local_only_energy = self.k * total_computation * math.pow(cpu, 2)

    def inital_DAG(self, task_id, config=None, T=None, D=None):
        self.DAG = DAG(task_id, self.freq, T=T, D=D)
        if config is not None:
            self.DAG.create_from_config(config)
        else:
            self.DAG.create(self.freq)
        self.DAG.get_valid_partition()
        self.DAG.local_only_compute_energy_time(self.freq, self.k)

    def partition(self):
        if self.config is not None:
            delta = self.config[5]
        else:
            delta = self.DAG.length

        self.local, self.remote = 0, 0
        for m in range(0, delta):
            self.local += round(self.DAG.jobs[m].computation)
        for m in range(delta, self.DAG.length):
            self.remote += round(self.DAG.jobs[m].computation)
        if delta == 0:
            self.local_to_remote_size = round(self.DAG.jobs[delta].input_data)
        else:
            self.local_to_remote_size = round(self.DAG.jobs[delta - 1].output_data)

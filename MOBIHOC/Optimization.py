from Offloading_Mobihoc import Offloading
import numpy as np
import threading
from threading import Lock
import copy


class Optimization:
    def __init__(self):
        self.cache = []
        self.lock = Lock()

    def clean_cache(self):
        self.cache = []

    def search_cache(self, info, user_id, edge_id):
        for s, d, config, task_id, k in self.cache:
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

    def energy_opt(self, info, target, k, delta):
        optimize = Offloading(info, k)
        info["selection"][target.task_id] = k
        info["opt_delta"][target.task_id] = delta
        config = self.search_cache(info, target.task_id, k)
        save = False
        if config is None:
            config = optimize.start_optimize(delta=delta, local_only_energy=info["local_only_energy"][target.task_id])
            save = True
        else:
            print("read from cached times", target.task_id, "edge=", k, "delta=", delta)
        if config is not None and (
                config[0] < info["local_only_energy"][target.task_id] or not info["local_only_enabled"][target.task_id]):
            self.lock.acquire()
            self.validation.append({
                "edge": k,
                "config": config
            })
            if save:
                selected = []
                partition_delta = []
                for n in range(info["number_of_user"]):
                    if info["selection"][n] == k:
                        selected.append(n)
                        partition_delta.append(info["opt_delta"][n])
                self.cache.append(
                    (selected, partition_delta, config, target.task_id, k))
            self.lock.release()
        self.lock.acquire()
        self.number_of_finished_opt += 1
        self.lock.release()

    def opt(self, info):
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
                self.number_of_opt += 1
                x = threading.Thread(target=self.energy_opt, args=(copy.deepcopy(info), target, k, delta))
                x.start()
                # self.energy_opt(info, target, k, delta)
        while True:
            if self.number_of_finished_opt == self.number_of_opt:
                break
        return self.validation, target

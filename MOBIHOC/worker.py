import time
from Offloading_Mobihoc import *

request = []
finish = []


def worker(info):
    global request
    # start = time.time()
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
        # print("FFFFFFFFFF", feasible, len(target.DAG.jobs))
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
                # [self.f_n[who], self.f_n_k[who][0], self.p_n[who], self.get_ch_number(who)]
                config = optimize.start_optimize(delta=delta)
                if config is not None and (config[0] < target.local_only_energy or not target.local_only_enabled):
                    validation.append({
                        "edge": k,
                        "config": config
                        # "info": optimize
                    })
    else:
        delta = 0
        info["Y_n"][target.task_id], info["X_n"][target.task_id], info["B"][target.task_id] = 0, 0, 0
        for m in range(0, target.DAG.length):
            info["X_n"][target.task_id] += target.DAG.jobs[m].computation
        info["B"][target.task_id] = target.DAG.jobs[delta].input_data
        for k in range(info["number_of_edge"]):
            optimize = Offloading(info, k)
            # [self.f_n[who], self.f_n_k[who][0], self.p_n[who], self.get_ch_number(who)]
            config = optimize.start_optimize(delta=delta)
            if config is not None and (config[0] < target.local_only_energy or not target.local_only_enabled):
                validation.append({
                    "edge": k,
                    "config": config
                    # "info": optimize
                })
            # print("thread", target.task_id, "finish edge test", k ,"in", time.time() - start)
    if len(validation) > 0:
        validation.sort(key=lambda x: x["config"][0])
        if validation[0]["edge"] != info["selection"][target.task_id]\
                or validation[0]["config"] != target.config:
            request[target.task_id] = {
                "user": target.task_id,
                "validation": validation[0],
                "local": False
            }
    else:
        if info["selection"][target.task_id] != -1:
            request[target.task_id] = {
                "user": target.task_id,
                "validation": None,
                "local": True
            }
    finish[target.task_id] = 1
    # print(validation)


def check_worker(doing):
    global finish
    for n in doing:
        if finish[n] != 1:
            return False
    return True


def get_request_pool():
    return request


def get_requests(request, item, selection):
    pool = [item]
    for req in request:
        if req == item or req is None:
            continue
        this_user = req["user"]
        if not req["local"]:
            this_k = req["validation"]["edge"]
        else:
            this_k = -1
        feasible = 0
        for req2 in pool:
            target = req2["user"]
            if not req2["local"]:
                edge_id = req2["validation"]["edge"]
                if this_k != edge_id and (selection[target] != selection[this_user] or (
                        selection[target] == -1 and selection[this_user] == -1)):
                    feasible += 1
            else:
                if (selection[target] != selection[this_user] or
                        (selection[target] == -1 and selection[this_user] == -1)):
                    feasible += 1
        if feasible == len(pool):
            pool.append(req)
    return pool


def reset_request_pool(number_of_user):
    global request, finish
    request = [None for n in range(number_of_user)]
    finish  = [0 for n in range(number_of_user)]

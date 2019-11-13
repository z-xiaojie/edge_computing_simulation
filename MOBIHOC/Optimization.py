from Offloading_Mobihoc import Offloading


def opt(info):
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
    # print("thread", target.task_id, "finish edge test", k)
    return validation, target


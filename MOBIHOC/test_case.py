from run import initial_energy_all_local, energy_update, get_request
import numpy as np
import math
from Client import check_worker


def test(controller, args, iteration, increment):
    ee_local, finished, user_hist = initial_energy_all_local(controller.selection, controller.player)
    print("local=", finished)
    print("total=", [item.total_computation for item in controller.player.users])
    t = 0
    hist = []
    finish_hist = []
    pre_energy = np.sum(ee_local)
    local_sum = pre_energy
    finish_hist.append(np.sum(finished))
    hist.append(np.sum(pre_energy))
    while True:
        changed = True
        req = get_request(controller, t)
        if req is None:
            print(">>>>>>>>>> no more request", controller.selection)
            changed = False
        else:
            for target in req:
                n, validation, local = target["user"], target["validation"], target["local"]
                # just_updated = n
                if local:
                    controller.player.edges[controller.selection[n]].remove(controller.player.users[n])
                    controller.player.users[n].config = None
                    controller.selection[n] = -1
                else:
                    k = validation["edge"]
                    controller.player.users[n].config = validation["config"]
                    if controller.selection[n] != -1 and controller.selection[n] != k:
                        controller.player.edges[controller.selection[n]].remove(controller.player.users[n])
                    if controller.selection[n] != k:
                        controller.player.edges[k].accept(controller.player.users[n])
                    # player.edges[k].update_resource_allocation(validation["info"])
                        controller.selection[n] = k

        for n in range(controller.player.number_of_user):
            if controller.player.users[n].config is not None:
                controller.opt_delta[n] = controller.player.users[n].config[5]
            else:
                controller.opt_delta[n] = -1
        t += 1
        if t % 1 == 0:
            user_hist, energy, finished, transmission, computation, edge_computation = energy_update(controller.player,
                                                                                                     controller.selection,
                                                                                                     user_hist)
            finish_hist.append(np.sum(finished))
            hist.append(np.sum(energy))
            if changed:
                opt_e_cpu = np.zeros(controller.player.number_of_edge)
                bandwidth = np.zeros(controller.player.number_of_edge)
                for n in range(controller.player.number_of_user):
                    if controller.player.users[n].config is not None:
                        bandwidth[controller.selection[n]] += controller.player.users[n].config[4]
                        opt_e_cpu[controller.selection[n]] = round(controller.player.users[n].config[2] / math.pow(10, 9), 4)
                F = True
                for k in range(controller.player.number_of_edge):
                    if opt_e_cpu[k] > controller.player.edges[k].freq or bandwidth[k] > controller.player.edges[k].number_of_chs:
                        F = False
                        break
                for target in req:
                    n, validation, local = target["user"], target["validation"], target["local"]
                    print(iteration, increment, t, round(np.sum(energy), 5), "/", local_sum, np.sum(finished), F, ">>>", n)
                    #       validation)

        if not changed:
            break

        if (t > 70 and F) or t > 150:
            break

    user_hist, energy, finished, transmission, computation, edge_computation = energy_update(controller.player,
                                                                                             controller.selection,
                                                                                             user_hist)
    finish_hist.append(np.sum(finished))
    hist.append(np.sum(energy))

    opt_cpu = []
    opt_e_cpu = []
    opt_power = []
    opt_delta = []
    bandwidth = []
    local, remote, local_to_remote = 0, 0, 0
    for n in range(controller.player.number_of_user):
        local += round(controller.player.users[n].local/math.pow(10, 9), 5)
        remote += round(controller.player.users[n].remote/math.pow(10, 9), 5)
        local_to_remote += round(controller.player.users[n].local_to_remote_size / 8000, 5)

        if controller.player.users[n].config is not None:
            opt_cpu.append(round(controller.player.users[n].config[1] / math.pow(10, 9), 4))
            opt_power.append(round(controller.player.users[n].config[3] * controller.player.users[n].config[4], 4))
            opt_delta.append(controller.player.users[n].config[5])
            bandwidth.append(controller.player.users[n].config[4])
            opt_e_cpu.append(round(controller.player.users[n].config[2] / math.pow(10, 9), 4))
        else:
            opt_delta.append(-1)
            opt_power.append(0)
            opt_cpu.append(0)
            bandwidth.append(0)
            opt_e_cpu.append(0)

    """
    print(">>>>>>>>>>>>>>>> TIME >>>>>>>>>>>>>>>>>>")
    print("adjusted local power", opt_power)
    print("adjusted local   CPU", opt_cpu)
    print("adjusted remote  CPU", opt_e_cpu, np.sum(opt_e_cpu))
    print("data", [round(controller.player.users[n].local_to_remote_size / 8000, 5) for n in range(controller.player.number_of_user)])
    print("              deadline", [round(controller.player.users[n].DAG.D / 1000, 4) for n in range(controller.player.number_of_user)])
    print("           finish time", list(np.round(np.array(transmission) + np.array(computation) + np.array(edge_computation),4)))
    print("     transmission time", transmission)
    print("     computation  time", computation)
    print("edge computation  time", edge_computation)
    
    print(">>>>>>>>>>>>>>>> energy >>>>>>>>>>>>>>>>>>")
    print("edge based energy", round(np.sum(energy), 6), energy)
    print("local only energy", round(np.sum(ee_local), 6), ee_local)
    print("            delta", opt_delta)
    print("local computation",
          sum([round(controller.player.users[n].local/math.pow(10, 9), 5) for n in range(controller.player.number_of_user)]))
    print("remote computation",
          [round(controller.player.users[n].remote/math.pow(10, 9), 5) for n in range(controller.player.number_of_user)])
    """
    print("improve path =", hist)
    print("finish path =", finish_hist)

    data = sum([round(controller.player.users[n].local_to_remote_size / 8000, 5) for n in
               range(controller.player.number_of_user)])
    remote = sum([round(controller.player.users[n].remote / math.pow(10, 9), 5) for n in
                  range(controller.player.number_of_user)])
    print("local computation",
          sum([round(controller.player.users[n].local/math.pow(10, 9), 5) for n in range(controller.player.number_of_user)]))
    print("data",
          )
    print("remote computation", remote)
    return hist[-1], data, remote


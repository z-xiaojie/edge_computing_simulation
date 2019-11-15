import random
from Role import Role
from run import initial_energy_all_local, energy_update, get_request
import numpy as np
import matplotlib.pyplot as plt
import copy
import math


def test(x, full, channel_allocation=1, epsilon=0.001, number_of_user=5, number_of_edge=1, player=None):

    selection = np.zeros(number_of_user).astype(int) - 1
    opt_delta = np.zeros(number_of_user).astype(int) - 1
    ee_local, finished, user_hist = initial_energy_all_local(selection, player)
    print("local=", finished)
    print("total=", [item.total_computation for item in player.users])
    t = 0
    hist = []
    finish_hist = []
    pre_energy = np.sum(ee_local)
    local_sum = pre_energy
    finish_hist.append(np.sum(finished))
    hist.append(np.sum(pre_energy))
    ttt = 0
    while True:
        changed = True
        just_updated = -2
        # changed_k = -1
        req = get_request(x, t, opt_delta, channel_allocation, just_updated, player, selection, full, epsilon=epsilon)
        if req is None:
            print(">>>>>>>>>> no more request")
            changed = False
        else:
            for target in req:
                n, validation, local = target["user"], target["validation"], target["local"]
                # just_updated = n
                if local:
                    player.edges[selection[n]].remove(player.users[n])
                    player.users[n].config = None
                    selection[n] = -1
                else:
                    k = validation["edge"]
                    player.users[n].config = validation["config"]
                    if selection[n] != -1 and selection[n] != k:
                        player.edges[selection[n]].remove(player.users[n])
                    if selection[n] != k:
                        player.edges[k].accept(player.users[n])
                    # player.edges[k].update_resource_allocation(validation["info"])
                    selection[n] = k

        for n in range(number_of_user):
            if player.users[n].config is not None:
                opt_delta[n] = player.users[n].config[5]
            else:
                opt_delta[n] = -1
        t += 1
        if t % 1 == 0:
            user_hist, energy, finished, transmission, computation, edge_computation = energy_update(player,
                                                                                                     selection,
                                                                                                     user_hist)
            finish_hist.append(np.sum(finished))
            hist.append(np.sum(energy))
        if changed:
            opt_e_cpu = np.zeros(number_of_edge)
            bandwidth = np.zeros(number_of_edge)
            for n in range(number_of_user):
                if player.users[n].config is not None:
                    bandwidth[selection[n]] += player.users[n].config[4]
                    opt_e_cpu[selection[n]] = round(player.users[n].config[2] / math.pow(10, 9), 4)
            F = True
            for k in range(number_of_edge):
                if opt_e_cpu[k] > player.edges[k].freq or bandwidth[k] > player.edges[k].number_of_chs:
                    F = False
                    break
            for target in req:
                n, validation, local = target["user"], target["validation"], target["local"]
                print(t, round(np.sum(energy), 5), "/", local_sum , np.sum(finished), F,  ">>>", n, ">>>", validation)

        if not changed:
            break

        if t > 45 and F:
            break

    opt_cpu = []
    opt_e_cpu = []
    opt_power = []
    opt_delta = []
    bandwidth = []
    for n in range(number_of_user):
        if player.users[n].config is not None:
            opt_cpu.append(round(player.users[n].config[1] / math.pow(10, 9), 4))
            opt_power.append(round(player.users[n].config[3] * player.users[n].config[4], 4))
            opt_delta.append(player.users[n].config[5])
            bandwidth.append(player.users[n].config[4])
            opt_e_cpu.append(round(player.users[n].config[2] / math.pow(10, 9), 4))
        else:
            opt_delta.append(-1)
            opt_power.append(0)
            opt_cpu.append(0)
            bandwidth.append(0)
            opt_e_cpu.append(0)

    print(">>>>>>>>>>>>>>>> TIME >>>>>>>>>>>>>>>>>>")
    print("adjusted local power", opt_power)
    print("adjusted local   CPU", opt_cpu)
    print("adjusted remote  CPU", opt_e_cpu, np.sum(opt_e_cpu))
    print("data", [round(player.users[n].local_to_remote_size / 8000, 5) for n in range(player.number_of_user)])
    print("              deadline", [round(player.users[n].DAG.D / 1000, 4) for n in range(number_of_user)])
    print("           finish time", list(np.round(np.array(transmission) + np.array(computation) + np.array(edge_computation),4)))
    print("     transmission time", transmission)
    print("     computation  time", computation)
    print("edge computation  time", edge_computation)
    
    print(">>>>>>>>>>>>>>>> energy >>>>>>>>>>>>>>>>>>")
    print("edge based energy", round(np.sum(energy), 6), energy)
    print("local only energy", round(np.sum(ee_local), 6), ee_local)
    print("            delta", opt_delta)
    print("local computation",
          [round(player.users[n].local/math.pow(10, 9), 5) for n in range(player.number_of_user)])
    print("remote computation",
          [round(player.users[n].remote/math.pow(10, 9), 5) for n in range(player.number_of_user)])

    # print("improvement", 1 - round(np.sum(energy), 5) / round(np.sum(ee_local), 5))
    #p rint(finished, "finished", np.sum(finished))

    #plt.subplot(1, 2, x)
    #plt.plot(hist, label="overall")

    for n in range(number_of_user):
        plt.plot(user_hist[n], label="user"+str(n+1))
        print("u"+str(n)+"=", user_hist[n])
    print("all=", list(hist))
    print("total computation", [round(player.users[n].total_computation, 4) for n in range(number_of_user)])
    print("finish time", [round((transmission[n] + computation[n] + edge_computation[n] - player.users[n].DAG.D/1000), 5) for n in range(player.number_of_user)])

    matched = 0
    """
    for n in range(number_of_user):
        validation = []
        for k in range(number_of_edge):
            config = player.users[n].select_partition(full, player.edges[k], epsilon=0.001, p_adjust=0.9, default_channel=5)
            if config is not None:
                validation.append({
                    "edge": k,
                    "config": config
                })
        # print("user", n, "validation", validation)
        if len(validation) > 0:
            validation.sort(key=lambda x: x["config"][0])
            if selection[n] == validation[0]["edge"]:
                matched += 1
        else:
            if selection[n] == -1:
                matched += 1
    """
    print("matched", matched)
    # for n in range(number_of_user):
       # print(player.users[n].config)
    #plt.legend()

    return t, finish_hist, bandwidth, opt_delta, selection, np.sum(finished)/number_of_user, round(np.sum(energy), 5), round(np.sum(ee_local), 5), 1 - round(np.sum(energy), 5) / round(np.sum(ee_local), 5)

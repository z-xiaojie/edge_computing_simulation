import math
import numpy as np
import random
import copy
from Offloading_Mobihoc import Offloading
import matplotlib.pyplot as plt


total_run = 1
ec_o_avg = np.zeros(total_run)
ec_o_opt = np.zeros(total_run)
ec_l = np.zeros(total_run)
ec_i = np.zeros(total_run)
e_data_opt = np.zeros(total_run)
e_data_avg = np.zeros(total_run)

i = 0
I = 1
while i < I:
    run = 0
    chs = 24
    f = 9.5
    info = {
        "selection": [0, 0, -1, -1, -1, -1, 0, 0],
        "number_of_edge": 1,
        "number_of_user": 8,
        "D_n": np.array([0.6, 0.95, 0.75, 0.9, 0.7, 0.55, 0.5, 0.55]),
        "X_n": np.array([416491671.6272385, 804247955.7522968, 507856767.3542032, 507111973.9158937, 407419890.822788, 435344591.14832044, 425451798.1277247, 439940048.8440688]),
        "Y_n": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        "user_cpu": np.array([1695674007.8171883, 2278071058.913955, 2448517951.575177, 2250999403.00921, 1875865373.9171755, 1645704170.6747947, 1806243046.7437308, 1832921135.0405736]),
        "edge_cpu": [9.5],
        "number_of_chs": 24,
        "P_max": np.array([0.8031527973206095, 0.8747348478820005, 0.8422611360288265, 0.7823784221287078, 0.6653549485683844,0.651064006522968, 0.6248772234598523, 0.5375540908755745]),
        "B": np.array([6336000, 5408000, 5544000, 4064000, 3872000, 2440000, 4672000, 5248000]),
        "H": np.array([([0.00144265]), ([0.00061738]), ([0.00080854]), ([0.00125665]), ([0.00097458]), ([0.00049686]), ([0.00150513]), ([0.00083923])]),
        "W": 2 * math.pow(10, 6),
        "who": None,
        "full": True,
        "default_channel": 1,
        "step": 0.002,
        "interval": 50,
        "stop_point": 0.001
    }

    r = Offloading(info, 0)
    r.f_n = np.array([r.user_cpu[n] for n in range(r.number_of_user)])
    for n in range(r.number_of_user):
        f_opt = min((r.X_n[n] + r.Y_n[n]) / r.D_n[n], r.f_n[n])
        e_n = r.g * r.k * (r.X_n[n] + r.Y_n[n]) * math.pow(f_opt, 2)
        r.loc_only_e[n] = e_n
    restart = False
    while run < total_run:
        for k in range(r.number_of_edge):
            r.edge_cpu[k] = f * math.pow(10, 9)
        r1 = copy.deepcopy(r)
        r2 = copy.deepcopy(r)
        # r1 : opt sub-channel allocation
        r1.set_initial_sub_channel(chs, 1, chs=None)
        step = 0.005
        if chs <= 24:
            r1.set_multipliers(step=step, p_adjust=0.98, delta_l_n=1, delta_d_n=1)
        else:
            r1.set_multipliers(step=0.01 / (2 * chs), p_adjust=0.98, delta_l_n=1, delta_d_n=1)
        r1.set_initial_values()
        restart, ee = r1.run(run, t=1, intervel=50, stop_point=0.0005)
        if restart:
            break
        # r2 : average sub-channel allocation
        r2.set_initial_sub_channel(chs, int(chs / 8), chs=None)
        if chs <= 24:
            r2.set_multipliers(step=step, p_adjust=0.98, delta_l_n=1, delta_d_n=1)
        else:
            r2.set_multipliers(step=0.01 / (2 * chs), p_adjust=0.98, delta_l_n=1, delta_d_n=1)
        r2.set_initial_values()
        restart, ee = r2.run(run, t=1, intervel=50, stop_point=0.0005)
        if restart:
            break

        energy_opt, energy_local, energy_opt_data, _ = test(r1)
        ec_o_opt[run] = (ec_o_opt[run] * i + energy_opt) / (i+1)
        e_data_opt[run] = (e_data_opt[run] * i + energy_opt_data) / (i + 1)
        ec_l[run] = (ec_l[run] * i + energy_local) / (i+1)

        energy_opt, _, energy_opt_data, _ = test(r2)
        ec_o_avg[run] = (ec_o_avg[run] * i + energy_opt) / (i+1)
        e_data_avg[run] = (e_data_avg[run] * i + energy_opt_data) / (i + 1)

        run = run + 1
        chs += 8
        # f += 0.5

    if not restart:
        i += 1

ec_o_opt = np.array(ec_o_opt)
ec_o_avg = np.array(ec_o_avg)
e_data_opt = np.array(e_data_opt)
e_data_avg = np.array(e_data_avg)
ec_l = np.array(ec_l)

print("ec_o_opt=", list(ec_o_opt))
print("ec_o_avg=", list(ec_o_avg))
print("e_data_opt=", list(e_data_opt))
print("e_data_avg=", list(e_data_avg))
print("l=", list(ec_l))
plt.plot(ec_o_opt, label="edge_computing_opt")
plt.plot(ec_o_avg, label="edge_computing_avg")
plt.plot(e_data_opt, label="edge_computing_opt-data")
plt.plot(e_data_avg, label="edge_computing_avg-data")
plt.plot(ec_l, label="local_computing")
plt.legend()
plt.show()
import math
import numpy as np
import time
import random
import copy
from scipy.special import lambertw
# import matplotlib.pyplot as plt


class Offloading:
    def __init__(self, info, edge_id):
        self.number_of_user = info["number_of_user"]
        self.number_of_edge = 1
        self.k = math.pow(10, -28)
        self.N_0 = math.pow(10, -9)
        # initialize
        self.edge_cpu = np.array([info["edge_cpu"][edge_id] for k in range(self.number_of_edge)])
        self.user_cpu = info["user_cpu"]
        self.selection = copy.copy(info["selection"])
        self.user_id = info["who"].task_id
        self.selection[self.user_id] = edge_id
        self.edge_id = edge_id
        # number of channel
        self.number_of_chs = info["number_of_chs"][edge_id]
        self.c_k = [info["number_of_chs"][edge_id] for k in range(self.number_of_edge)]
        self.D_n = info["D_n"]
        self.Y_n = info["Y_n"]
        self.X_n = info["X_n"]
        self.P_max = info["P_max"]
        self.B = info["B"]
        self.H = info["H"]
        self.W = info["W"]
        self.interval = info["interval"]
        self.stop_point = info["stop_point"]
        self.step = info["step"]
        self.c_n_k = None
        self.number_of_offloaded_user = 0
        for n in range(self.number_of_user):
            if self.selection[n] == edge_id:
                self.number_of_offloaded_user += 1
        self.channel_allocation = info["channel_allocation"]
        if self.channel_allocation == 1:
            self.default_channel = info["default_channel"]
        else:
            self.default_channel = int(self.number_of_chs/self.number_of_offloaded_user)
        self.ch = np.zeros(self.number_of_user) + self.default_channel
        self.l_n = None
        self.d_n = None
        self.t_n = None
        self.p_n = None
        self.f_n = np.array([self.user_cpu[n] for n in range(self.number_of_user)])
        self.f_n_k = None
        self.delta_l_n = None
        self.delta_d_n = None
        self.chs = False
        self.loc_only_e = np.zeros(self.number_of_user)
        self.printed = False
        self.t_n_n = np.zeros(self.number_of_user).astype(float)

    def set_initial_sub_channel(self, chs=None):
        # initial channel
        self.c_n_k = [[[0 for c in range(self.c_k[k])] for k in range(self.number_of_edge)]for n in range(self.number_of_user)]
        if chs is not None:
            for k in range(self.number_of_edge):
                c = 0
                for n in range(self.number_of_user):
                    if self.selection[n] != self.edge_id:
                        continue
                    rep = 0
                    while rep < chs[n]:
                        self.c_n_k[n][k][c] = 1
                        c += 1
                        rep += 1
            self.chs = True
        else:
            for k in range(self.number_of_edge):
                n = 0
                c = 0
                while c < self.c_k[k] and n < self.number_of_user:
                    if self.selection[n] != self.edge_id:
                        n += 1
                        continue
                    rep = 0
                    while rep < self.default_channel:
                        self.c_n_k[n][k][c] = 1
                        c += 1
                        rep += 1
                    n += 1
                if self.channel_allocation == 1:
                    return
                n = 0
                while c < self.c_k[k]:
                    if self.selection[n] != self.edge_id:
                        n += 1
                        continue
                    self.c_n_k[n][k][c] = 1
                    self.ch[n] += 1
                    c += 1
                    n += 1
        #print(self.user_id, "ch finished")

    def set_multipliers(self, step=0.00001, p_adjust=0.5, delta_l_n=math.pow(10, -12), delta_d_n=math.pow(10, -14)):
        self.step = step
        self.d_n = np.array([(math.pow(self.user_cpu[n], 3) * 2 * self.k) ** (1. / 3) for n in range(self.number_of_user)])
        self.l_n = [0 for n in range(self.number_of_user)]
        self.delta_l_n = delta_l_n
        self.delta_d_n = delta_d_n

    def set_initial_values(self):
        weight = 0
        for n in range(self.number_of_user):
            if self.selection[n] != self.edge_id:
                continue
            weight += math.sqrt(self.X_n[n] * self.d_n[n])
        self.f_n_k = np.array([[self.edge_cpu[k] * math.sqrt(self.X_n[n] * self.d_n[n]) / weight
                                for k in range(self.number_of_edge)] for n in range(self.number_of_user)])
        self.t_n = np.array([[self.get_t(n, k) for k in range(self.number_of_edge)] for n in range(self.number_of_user)])
        self.t_n = self.t_n.reshape(-1)
        self.p_n = np.zeros(self.number_of_user)
        for n in range(self.number_of_user):
            if self.selection[n] != self.edge_id:
                continue
            self.p_n[n] = min(self.N_0 * (math.pow(2, self.B[n]/(self.t_n[n] * self.get_ch_number(n) * self.W))-1)
                                  /math.pow(self.H[n][self.edge_id], 2), self.P_max[n]/self.get_ch_number(n))
        chs = []
        for n in range(self.number_of_user):
            chs.append(self.get_ch_number(n))

    def calculate_energy(self):
        e_n_k = 0
        for n in range(self.number_of_user):
            for k in range(self.number_of_edge):
                if self.selection[n] != self.edge_id:
                    continue
                e_n_k += self.p_n[n] * self.get_ch_number(n) * self.t_n[n] \
                    + self.k * self.Y_n[n] * math.pow(self.f_n[n], 2)
        return e_n_k

    def get_t(self, n, k):
        if self.selection[n] != self.edge_id:
            return 0
        a = math.log(2) * self.B[n]
        e = np.exp(1)
        c = (self.d_n[n] * math.pow(self.H[n][self.edge_id], 2) / ((1+self.l_n[n])*self.get_ch_number(n) * self.N_0 * e)) - 1/e
        b = self.get_ch_number(n) * self.W * (lambertw(c, 0)+1)
        r_max = self.W * self.get_ch_number(n) * math.log2(
            1 + (self.P_max[n] / self.get_ch_number(n)) * math.pow(self.H[n][self.edge_id], 2) / self.N_0)
        return max((a/b).real, self.B[n]/r_max)

    def update(self, t, time):
        stop = True
        diff_2 = 0
        for n in range(self.number_of_user):
            if self.selection[n] != self.edge_id:
                continue
            diff = 0
            for k in range(self.number_of_edge):
                diff += self.get_ch_number(n) * self.p_n[n] - self.P_max[n]
            new_l_n = max(0, self.l_n[n] + self.delta_l_n * math.sqrt(self.step / t) * diff)
            if diff < 0:
                self.l_n[n] = 0
            else:
                if new_l_n == 0:
                    self.delta_l_n = self.delta_l_n * 0.9
                    continue
                self.l_n[n] = new_l_n
            diff_2 += diff

        diff_ = []
        finished = 0
        for n in range(self.number_of_user):
            if self.selection[n] != self.edge_id:
                continue
            diff = 0
            for k in range(self.number_of_edge):
                rate = self.W * self.get_ch_number(n) * math.log2(
                    1 + self.p_n[n] * math.pow(self.H[n][self.edge_id], 2) / self.N_0)
                diff += self.t_n[n] + self.X_n[n]/self.f_n_k[n][k] + self.Y_n[n]/self.f_n[n] - self.D_n[n]
                if round(self.B[n]/rate, 6) != round(self.t_n[n], 6):
                    print(self.user_id, t, "xxxxxxxxxxxxxxxx", rate, self.t_n_n)
                if diff <= 0:
                    finished += 1
                diff_.append(diff)
                new_d_n = max(0, self.d_n[n] + self.delta_d_n * math.sqrt(self.step / t) * diff)
                if new_d_n == 0:
                    self.delta_d_n = self.delta_d_n * 0.8
                    continue
                self.d_n[n] = new_d_n
        # print(self.t_n_n)
        if time % 1000 == 0 and time > 1 and not self.printed:
            self.delta_d_n = self.delta_d_n * 2
            self.printed = True

        for item in diff_:
            if math.fabs(item) > self.stop_point:
                stop = False
        return diff_, stop

    def get_ch_number(self, n):
        if self.selection[n] != self.edge_id:
            return 1
        """
        total = 0
        if self.ch[n] != -1:
            total = self.ch[n]
        else:
            for k in range(self.number_of_edge):
                for c in range(self.c_k[k]):
                    if self.c_n_k[n][k][c] == 1:
                        total += 1
            self.ch[n] = total
        """
        return self.ch[n]

    def new_value(self):
        for k in range(self.number_of_edge):
            weight = 0
            for n in range(self.number_of_user):
                if self.selection[n] != self.edge_id:
                    continue
                weight += math.sqrt(self.X_n[n] * self.d_n[n])
            for n in range(self.number_of_user):
                if self.selection[n] != self.edge_id:
                    continue
                if self.Y_n[n] != 0:
                    self.f_n[n] = min((self.d_n[n] / (2 * self.k)) ** (1. / 3), self.user_cpu[n])
                self.f_n_k[n][k] = self.edge_cpu[k] * math.sqrt(self.X_n[n] * self.d_n[n]) / weight
                self.t_n[n] = self.get_t(n, k)
                self.p_n[n] = min(self.N_0 * (math.pow(2, self.B[n]/(self.t_n[n] * self.get_ch_number(n) * self.W))-1)
                                  / math.pow(self.H[n][self.edge_id], 2), self.P_max[n]/self.get_ch_number(n))
                # rate = self.W * self.get_ch_number(n) * math.log2(
                #    1 + self.p_n[n] * math.pow(self.H[n][self.edge_id], 2) / self.N_0)
                # self.t_n_n[n] = round(rate/8000, 0)
        # print(self.user_id, self.t_n_n)

    def assign_ch(self):
        if self.channel_allocation == 0:
            return
        self.ch = np.zeros(self.number_of_user) + self.default_channel
        for k in range(self.number_of_edge):
            for c in range(self.number_of_offloaded_user * self.default_channel, self.c_k[k]):
                opt = -1
                max_gain = 999
                for n in range(self.number_of_user):
                    if self.selection[n] != self.edge_id:
                        continue
                    need_rate = self.B[n] / self.t_n[n]
                    cgs = self.get_ch_number(n)
                    a = need_rate / (self.W * (cgs + 1))
                    b = need_rate / (self.W * cgs)
                    p_new = self.t_n[n] * (cgs + 1) * (self.N_0 / math.pow(self.H[n][self.edge_id], 2)) * (math.pow(2, a) - 1)
                    p_old = self.t_n[n] * cgs * (self.N_0 / math.pow(self.H[n][self.edge_id], 2)) * (math.pow(2, b) - 1)
                    gain = p_old - p_new
                    if gain > max_gain or opt == -1:
                        opt = n
                        max_gain = gain
                if opt != -1:
                    for n in range(self.number_of_user):
                        if n == opt and self.selection[n] == self.edge_id:
                            self.c_n_k[n][k][c] = 1
                            self.ch[n] += 1
                        else:
                            self.c_n_k[n][k][c] = 0

    def run(self):
        diff, pre_diff = [], [0]
        ee = []
        t = 1
        # start = time.time()
        while True:
            diff, stop1 = self.update(t, t)
            t = t + 1
            if stop1:
                break
            if (round(np.sum(pre_diff), 6) == round(np.sum(diff), 6) and t > 800) or t >= 2500:
                return True, None, t
            else:
                pre_diff = diff
            if t % self.interval == 0 and t <= 10000:
                self.assign_ch()
            self.new_value()
            #if t % 50 == 0 and t > 50:
                #ee.append(self.calculate_energy())
        # print("finished in", time.time() - start, t)
        # plt.plot(ee)
        # plt.show()
        return False, ee, t

    def start_optimize(self, delta=None, local_only_energy=None):
        who = self.user_id
        self.set_initial_sub_channel()
        self.set_multipliers(step=self.step, p_adjust=0.5, delta_l_n=self.number_of_offloaded_user * 5
                             , delta_d_n=math.pow(2,  self.number_of_offloaded_user))
        self.set_initial_values()
        restart, ee, t = self.run()
        if not restart:
            target_energy = self.p_n[who] * self.get_ch_number(who) * self.t_n[who] \
                    + self.k * self.Y_n[who] * math.pow(self.f_n[who], 2)
            rate = self.W * self.get_ch_number(who) * math.log2(1 + self.p_n[who] * math.pow(self.H[who][self.edge_id], 2) / self.N_0)
            #finish_time1 = self.t_n[who] + self.X_n[who]/self.f_n_k[who][0] + self.Y_n[who]/self.f_n[who] - self.D_n[who]
            finish_time2 = self.B[who]/rate + self.X_n[who] / self.f_n_k[who][0] + self.Y_n[who] / self.f_n[who] - \
                           self.D_n[who]
            if delta == 0:
                self.f_n[who] = 0
            return [target_energy, self.f_n[who], self.f_n_k[who][0], self.p_n[who], self.get_ch_number(who), delta, self.edge_id, t, round(finish_time2, 7), who]
        else:
            return None


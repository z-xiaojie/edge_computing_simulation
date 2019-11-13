import math
import numpy as np
from scipy.special import lambertw
import matplotlib.pyplot as plt


class EdgeCloud:
    def __init__(self, id, frequency, number_of_chs=None):
        self.freq = round(frequency)
        self.number_of_chs = number_of_chs
        self.tasks = []
        self.id = id
        self.number_of_user = 0
        self.chs = None
        self.k = math.pow(10, -28)
        self.N_0 = math.pow(10, -9)
        self.W = 1 * math.pow(10, 6)

        self.f_n = None
        self.p_n = None
        self.d_n = None
        self.l_n = None
        self.f_n_k = None
        self.t_n = None
        self.history = None
        self.d_n_min = None

    def accept(self, task):
        for n in range(self.number_of_user):
            if self.tasks[n].task_id == task.task_id:
                return True, n
        self.tasks.append(task)
        self.number_of_user = len(self.tasks)
        return False, self.number_of_user - 1

    def remove(self, task):
        self.tasks.remove(task)
        self.number_of_user = len(self.tasks)

    def get_ch_number(self, n):
        return self.chs[n]

    def set_initial_sub_channel(self,  default_channel):
        allocation = [[0 for c in range(self.number_of_chs)] for n in range(self.number_of_user)]
        n = 0
        c = 0
        while c < default_channel and n < self.number_of_user:
            rep = 0
            while rep < default_channel:
                allocation[n][c] = 1
                c += 1
                rep += 1
            n += 1
        self.chs = [default_channel for n in range(self.number_of_user)]
        return allocation

    def get_min_d_n(self):
        for n in range(self.number_of_user):
            w = (self.tasks[n].local_to_remote_size * math.log(2) / (self.W * self.tasks[n].DAG.D/100)) - 1
            d = 0.000005
            x = d * math.pow(self.tasks[n].H[self.id], 2) / (self.get_ch_number(n) * self.N_0 * np.exp(1)) - 1 / np.exp(
                1)
            current = lambertw(x, 0)
            while current.real < w:
                d = d + 0.000005
                x = d * math.pow(self.tasks[n].H[self.id], 2) / (self.get_ch_number(n) * self.N_0 * np.exp(1)) - 1 / np.exp(1)
                current = lambertw(x, 0)
                # print(d, current.real, w)
            self.d_n_min[n] = d
        # print("min d_n", self.d_n_min)

    def get_optimal_transmission_time(self, n):
        Z = self.tasks[n].local_to_remote_size
        a = math.log(2) * Z
        e = np.exp(1)
        c = (self.d_n[n] * math.pow(self.tasks[n].H[self.id], 2) / ((1+self.l_n[n])*self.get_ch_number(n) * self.N_0 * e)) - 1/e
        b = self.get_ch_number(n) * self.W * (lambertw(c, 0)+1)
        r_max = self.W * self.get_ch_number(n) * math.log2(
            1 + (self.tasks[n].p_max / self.get_ch_number(n)) * math.pow(self.tasks[n].H[self.id], 2) / self.N_0)
        return max((a/b).real, Z/r_max)

    def update_multiplier(self, t, current_time, epsilon=0.001, step=0.001):
        stop = True
        for n in range(self.number_of_user):
            diff = self.get_ch_number(n) * self.p_n[n] - self.tasks[n].p_max
            new_l_n = max(0, self.l_n[n] + math.sqrt(step / current_time) * diff)
            if diff < 0:
                self.l_n[n] = 0
            else:
                self.l_n[n] = new_l_n
        diff_ = []
        finished = 0
        for n in range(self.number_of_user):
            diff = self.t_n[n] + self.tasks[n].remote / self.f_n_k[n] + self.tasks[n].local / self.f_n[n] - self.tasks[n].DAG.D/1000
            if diff <= 0:
                finished += 1
            max_d_n = math.pow(self.tasks[n].freq, 3) * 2 * self.k
            """
            r_max = self.W * self.get_ch_number(n) * math.log2(
                1 + (self.tasks[n].p_max / self.get_ch_number(n)) * math.pow(self.tasks[n].H[self.id], 2) / self.N_0)
            Z = self.tasks[n].local_to_remote_size
            if self.tasks[n].local != 0:
                 max_local_compuation_time = self.tasks[n].DAG.D / 1000 - Z/r_max - self.tasks[n].remote / self.freq
                 min_d_n = math.pow(self.tasks[n].local/max_local_compuation_time, 3) * 2 * self.k
            else:
                max_server_compuation_time = self.tasks[n].DAG.D / 1000 - Z / r_max
                min_d_n = math.pow(self.tasks[n].remote / max_server_compuation_time, 2)/(math.pow(self.freq, 2) * self.tasks[n].remote)
            """
            if current_time > 350:
                self.d_n[n] = max(0, self.d_n[n] + math.sqrt(step / current_time) * diff)
            else:
                self.d_n[n] = max(0, self.d_n[n] + math.sqrt(0.01 * step / current_time) * diff)
            """
            if diff < 0 and min(max(self.d_n_min[n], self.d_n[n] + math.sqrt(step / current_time) * diff), max_d_n) == 0:
                self.d_n[n] = self.d_n[n] / 1.1
            else:
                self.d_n[n] = min(max(self.d_n_min[n], self.d_n[n] + math.sqrt(step / current_time) * diff), max_d_n)
            """
            #self.d_n[n] = min(max(self.d_n_min[n], self.d_n[n] + math.sqrt(step / current_time) * diff), max_d_n)
            diff_.append(diff)
        for item in diff_:
            if math.fabs(item) > epsilon:
                stop = False
        #if t > 1:
            #print(t, self.d_n, self.l_n, diff_)
        return stop, diff_

    def new_value(self):
        sum_weight = 0
        for n in range(self.number_of_user):
            sum_weight += math.sqrt(self.tasks[n].remote * self.d_n[n])
        if sum_weight == 0:
            print(">>>>>>>>>", self.tasks[n].remote/math.pow(10, 9), self.tasks[n].local/math.pow(10, 9), self.d_n[n])
        for n in range(self.number_of_user):
            self.f_n[n] = min((self.d_n[n] / (2 * self.k)) ** (1. / 3), self.tasks[n].freq)
            self.f_n_k[n] = self.freq * math.sqrt(self.tasks[n].remote * self.d_n[n]) / sum_weight
            Z = self.tasks[n].local_to_remote_size
            self.t_n[n] = self.get_optimal_transmission_time(n)
            self.p_n[n] = min(self.N_0 * (math.pow(2, Z / (self.t_n[n] * self.get_ch_number(n) * self.W)) - 1) / math.pow(
                self.tasks[n].H[self.id], 2), self.tasks[n].p_max/self.get_ch_number(n))

    def assign_ch(self, default_channel):
        allocation = self.set_initial_sub_channel(default_channel)
        for c in range(self.number_of_user * default_channel, self.number_of_chs):
            opt = -1
            max_gain = 999
            for n in range(self.number_of_user):
                need_rate = self.tasks[n].local_to_remote_size / self.t_n[n]
                cgs = self.get_ch_number(n)
                a = need_rate / (self.W * (cgs + 1))
                b = need_rate / (self.W * cgs)
                p_new = self.t_n[n] * (cgs + 1) * (self.N_0 / math.pow(self.tasks[n].H[self.id], 2)) * (math.pow(2, a) - 1)
                p_old = self.t_n[n] * cgs * (self.N_0 / math.pow(self.tasks[n].H[self.id], 2)) * (math.pow(2, b) - 1)
                gain = p_old - p_new
                if gain > max_gain or opt == -1:
                    opt = n
                    max_gain = gain
            if opt != -1:
                for n in range(self.number_of_user):
                    if n == opt:
                        allocation[n][c] = 1
                        self.chs[n] += 1
                    else:
                        allocation[n][c] = 0

    def calculate_energy(self):
        e_n_k = np.zeros(self.number_of_user)
        finish_time = np.zeros(self.number_of_user)
        for n in range(self.number_of_user):
            z = self.tasks[n].local_to_remote_size
            rate = self.W * self.get_ch_number(n) * math.log2(
            1 + self.p_n[n] * math.pow(self.tasks[n].H[self.id], 2) / self.N_0)
            transmission_time = z/rate
            e_n_k[n] = self.p_n[n] * self.get_ch_number(n) * transmission_time + self.k * self.tasks[n].local * math.pow(self.f_n[n], 2)
            finish_time[n] = transmission_time + self.tasks[n].local / self.f_n[n] + self.tasks[n].remote / self.f_n_k[n]\
                             - self.tasks[n].DAG.D/1000
        return e_n_k, finish_time

    def clean_history(self):
        self.history = []

    def save_temp_resource_allocation(self, delta, energy_vector):
        temp = {
            "delta": delta,
            "configs": []
        }
        for n in range(self.number_of_user):
            temp["configs"].append([round(energy_vector[n], 5), self.f_n_k[n], self.get_ch_number(n), self.tasks[n].task_id
                                   , self.f_n[n], self.p_n[n]])
        self.history.append(temp)

    def update_resource_allocation(self, info):
        for n in range(self.number_of_user):
            task_id = self.tasks[n].task_id
            self.tasks[n].config[1] = info.f_n[task_id]
            self.tasks[n].config[2] = info.f_n_k[task_id][0]
            self.tasks[n].config[3] = info.p_n[task_id]
            self.tasks[n].config[4] = info.get_ch_number(task_id)

    def restore_partition(self, who):
        for n in range(self.number_of_user):
            if n != who.task_id:
                 self.tasks[n].partition()

    def resource_allocation(self, delta, who=None, epsilon=0.0001, p_adjust=0.9, default_channel=3):
        self.restore_partition(who)
        exist, index = self.accept(who)
        # 初始化 sub-channel
        self.set_initial_sub_channel(default_channel)
        self.d_n_min = np.zeros(self.number_of_user)
        self.get_min_d_n()
        # 初始化参数
        self.f_n = np.array([self.tasks[n].freq/5 for n in range(self.number_of_user)])
        self.p_n = np.array([0.2 * self.tasks[n].p_max / self.get_ch_number(n) for n in range(self.number_of_user)])
        self.d_n = np.array([(math.pow(self.tasks[n].freq, 3) * 2 * self.k) ** (1. / 3) for n in range(self.number_of_user)])
        self.l_n = [0 for n in range(self.number_of_user)]
        sum_weight = 0
        for n in range(self.number_of_user):
            sum_weight += math.sqrt(self.tasks[n].remote * self.d_n[n])
        self.f_n_k = np.array([self.freq * math.sqrt(self.tasks[n].remote * self.d_n[n]) / sum_weight
                               for n in range(self.number_of_user)])
        self.t_n = np.array([self.get_optimal_transmission_time(n) for n in range(self.number_of_user)])
        self.t_n = self.t_n.reshape(-1)
        t = 1
        ee = []
        energy_vector = None
        finish_time = None
        diff, pre_diff = 0, 0
        step = 0.01 / (2 * self.number_of_chs)
        feasible = True
        while True:
            stop1, diff = self.update_multiplier(t, 0.0015 * t, epsilon=epsilon, step=step)
            if stop1:
                break
            if round(np.sum(pre_diff), 6) == round(np.sum(diff), 6) and t >= 45000:
                feasible = False
                break
            else:
                pre_diff = diff
            self.new_value()
            if t % 50 == 0:
                self.assign_ch(default_channel)
            #if t % 100 == 0:
            #    energy_vector, finish_time = self.calculate_energy()
            #    ee.append(np.sum(energy_vector))
            t = t + 1
        energy_vector, finish_time = self.calculate_energy()
        self.save_temp_resource_allocation(delta, energy_vector)
        #plt.plot(ee, label="user"+str(who.task_id)+","+str(delta))
        #plt.legend()
        #plt.show()

        local, remote, size = 0, 0, 0
        for n in range(self.number_of_user):
            local += self.tasks[n].local
            remote += self.tasks[n].remote
            size += self.tasks[n].local_to_remote_size

        if not exist:
            self.remove(who)
        # print("test user", who.task_id,"edge", self.id,[round(energy_vector[index], 5), round(self.f_n[index]), self.p_n[index],
        #        round(self.f_n_k[index]), self.get_ch_number(index), delta, self.id, local, remote, size, round(finish_time[index],5), t])
        if feasible:
            return [round(energy_vector[index], 5), round(self.f_n[index]), self.p_n[index], round(self.f_n_k[index]),self.get_ch_number(index), delta, self.id, local, remote, size, t]
        else:
            return None

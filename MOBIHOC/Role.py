from EdgeCloud import EdgeCloud
from Device import Device
import random
import math
import numpy as np
import copy
import operator


class Role:
    def __init__(self, number_of_edge=3, number_of_user=3, epsilon=0.01, number_of_chs=None, cpu=None, d_cpu=None, H=None):

        self.number_of_user = number_of_user
        self.users = []
        for n in range(self.number_of_user):
            self.users.append(Device(d_cpu[n], n, H[n], transmission_power=random.uniform(0.5, 1), epsilon=epsilon))

        self.number_of_edge = number_of_edge
        self.edges = []
        for k in range(self.number_of_edge):
            self.edges.append(EdgeCloud(k, cpu[k], number_of_chs=number_of_chs[k]))

        self.network = number_of_chs

    def initial_config_DAG(self):
        config = [[[1308.0, 0.09287, 629.0], [629.0, 0.4451, 0.0]],
                  [[951.0, 0.22686, 600.0], [600.0, 0.18808, 648.0], [648.0, 0.12107, 530.0], [530.0, 0.19671, 0.0]],
                  [[1185.0, 0.19363, 598.0], [598.0, 0.70142, 0.0]],
                  [[1451.0, 0.07325, 721.0], [721.0, 0.18146, 284.0], [284.0, 0.23095, 389.0], [389.0, 0.17382, 0.0]],
                  [[672.0, 0.04304, 496.0], [496.0, 0.13375, 491.0], [491.0, 0.10419, 0.0]],
                  [[1454.0, 0.15509, 488.0], [488.0, 0.50718, 0.0]], [[380.0, 0.13253, 363.0], [363.0, 0.51685, 0.0]],
                  [[1045.0, 0.16953, 721.0], [721.0, 0.23164, 289.0], [289.0, 0.17713, 0.0]],
                  [[851.0, 0.12276, 546.0], [546.0, 0.12959, 710.0], [710.0, 0.23784, 0.0]],
                  [[395.0, 0.13813, 403.0], [403.0, 0.22602, 401.0], [401.0, 0.26724, 0.0]],
                  [[1390.0, 0.10122, 422.0], [422.0, 0.18975, 519.0], [519.0, 0.10341, 0.0]],
                  [[905.0, 0.08288, 524.0], [524.0, 0.11964, 533.0], [533.0, 0.15551, 432.0], [432.0, 0.1224, 0.0]],
                  [[602.0, 0.15117, 261.0], [261.0, 0.18939, 738.0], [738.0, 0.2108, 479.0], [479.0, 0.19191, 0.0]],
                  [[489.0, 0.09162, 346.0], [346.0, 0.53629, 0.0]], [[1072.0, 0.18888, 524.0], [524.0, 0.34483, 0.0]]]
        cpu = [2203715405.0, 1951824082.0, 2342159284.0, 2024753308.0, 1763514355.0, 2042290971.0, 1643883163.0,
               2138105380.0, 2303907840.0, 2020344610.0, 1608469945.0, 1590316600.0, 2311988495.0, 2293565533.0,
               1700067420.0]
        T = [650, 750, 550, 1000, 500, 850, 600, 800, 900, 950, 550, 650, 900, 650, 600]
        D = [800, 500, 600, 900, 600, 650, 1000, 950, 800, 600, 800, 600, 800, 850, 700]
        for n in range(self.number_of_user):
            self.users[n].freq = cpu[n]
            self.users[n].inital_DAG(n, config[n], T[n], D[n])

    def initial_DAG(self):
        job_list = []
        cpu = []
        T = []
        D = []
        for n in range(self.number_of_user):
            self.users[n].inital_DAG(n)
            job_list.append(self.users[n].DAG.display())
            cpu.append(self.users[n].freq)
            T.append(self.users[n].DAG.T)
            D.append(self.users[n].DAG.D)
        print("config=", job_list)
        print("cpu=", cpu)
        print("T=", T)
        print("D=", D)

    def create_partition_map(self, alpha):
        partition = []
        CPU_MAX = 6 * math.pow(10, 9)
        BANDWIDTH_MAX = 30
        resource = []
        for i in range(1, 22):
            for j in range(22, 1, -1):
                resource.append([CPU_MAX/i, BANDWIDTH_MAX/j])
        #print(">>>>>", resource)
        for n in range(self.number_of_user):
            p = self.users[n].transmission_power
            H = np.average(self.users[n].H)
            G = 1 + p * math.pow(H, 2) / math.pow(10, -9)
            EE = []
            for delta in range(len(self.users[n].DAG.jobs)):
                local, remote, data = 0, 0, self.users[n].DAG.jobs[delta].output_data
                for m in range(0, delta + 1):
                    local += self.users[n].DAG.jobs[m].computation
                for m in range(delta + 1, self.users[n].DAG.length):
                    remote += self.users[n].DAG.jobs[m].computation
                EI = []
                for item in resource:
                    R = item[1] * math.pow(10, 6) * math.log2(G)
                    T = data / R
                    E = local * self.users[n].k * self.users[n].freq * self.users[n].freq + T * p
                    F = T + local/self.users[n].freq
                    #value = (1 - E/self.users[n].DAG.local_only_energy) * alpha \
                    #        + (1 - alpha) * (1 - F/self.users[n].DAG.local_only_time)
                    if F <= self.users[n].DAG.D / 1000:
                        EI.append(E)
                    else:
                        EI.append(99)

                EE.append(EI)

            rank = np.zeros(self.users[n].DAG.length)
            for i in range(len(resource)):
                s = 0
                index = -1
                for delta in range(len(self.users[n].DAG.jobs)):
                    if index == -1 or EE[delta][i] < s:
                        s = EE[delta][i]
                        index = delta
                rank[index] += 1.

            #print(rank
            #      , [round(self.users[n].DAG.jobs[i].output_data/8000,0) for i in range(len(self.users[n].DAG.jobs))]
            #      , [round(self.users[n].DAG.jobs[i].computation / math.pow(10, 9), 4) for i in range(len(self.users[n].DAG.jobs))]
            #)
            max_index, max_value = max(enumerate(rank), key=operator.itemgetter(1))
            partition.append(max_index)
        print("partition:", partition)
        return partition

    def find_k_by_id(self, id):
        for k in range(self.number_of_edge):
            if id == self.edges[k].id:
                return k
        return -1


    """
        Given a partition, calculate local-only, remote-only and data size
    """
    def partition(self, policy):
        for n in range(self.number_of_user):
            delta = policy[n]
            self.users[n].delta = delta
            self.users[n].local, self.users[n].remote = 0, 0
            for m in range(0, delta + 1):
                self.users[n].local += self.users[n].DAG.jobs[m].computation
            for m in range(delta + 1, self.users[n].DAG.length):
                self.users[n].remote += self.users[n].DAG.jobs[m].computation
            self.users[n].local_to_remote_size = self.users[n].DAG.jobs[delta].output_data

    def EDPA(self, policy=0, partition=None, alpha=None):
        for k in range(self.number_of_edge):
            self.edges[k].tasks = []

        for k in range(self.number_of_edge):
            self.edges[k].channel_allocation()

        sum_job_size = 0
        sum_job_computation = 0
        for n in range(self.number_of_user):
            sum_job_size += self.users[n].local_to_remote_size
            sum_job_computation += self.users[n].remote

        for k in range(self.number_of_edge):
            H_K = 0
            for n in range(self.number_of_user):
                H_K += self.users[n].H[k]
            #self.users.sort(key=lambda x: x.priority(k, H_K, sum_job_size, sum_job_computation))
            self.users.sort(key=lambda x: x.H[k], reverse=True)
            order = []
            for user in self.users:
                order.append(user.task_id)
            self.edges[k].preference = order

            #print("preference ", k, order)

        self.users.sort(key=lambda x: x.task_id)

        selection = [-1 for x in range(self.number_of_user)]

        self.update_user_preference(selection)

        while True:
            proposing = []
            if policy == 0 or policy == 1:
                for k in range(self.number_of_edge):
                    p_list = []
                    for n in self.edges[k].preference:
                        self.edges[k].tasks.append(self.users[n])
                        if self.edges[k].check():
                            proposing.append({
                                "k": k,
                                "n": n
                            })
                            p_list.append(self.users[n])
                            break
                        else:
                            p_list.append(self.users[n])
                            break
                    for item in p_list:
                        self.edges[k].preference.remove(item.task_id)
                        self.edges[k].tasks.remove(item)
            """
            if policy == 1:
                dist = [[self.edges[x].id, self.edges[x].edge_density()] for x in range(self.number_of_edge)]
                dist.sort(key=lambda x: x[1])
                for item in dist[:int(len(dist)/2)]:
                    k = item[0]
                    p_list = []
                    for n in self.edges[k].preference:
                        self.edges[k].tasks.append(self.users[n])
                        if self.edges[k].check():
                            proposing.append({
                                "k": k,
                                "n": n
                            })
                            p_list.append(self.users[n])
                            break
                        else:
                            p_list.append(self.users[n])
                            break
                    for item in p_list:
                        self.edges[k].preference.remove(item.task_id)
                        self.edges[k].tasks.remove(item)
                #print(proposing)
            """
            if len(proposing) == 0:
                break

            for n in range(self.number_of_user):
                request = []
                for item in proposing:
                    if item["n"] == n:
                        request.append(item["k"])
                if len(request) == 0:
                    continue
                pre_k = selection[n]
                if pre_k > -1:
                    like_k = pre_k
                else:
                    like_k = request[0]
                for proposer in request:
                    if self.users[n].compare_preference(like_k, proposer):
                        like_k = proposer
                if pre_k != like_k:
                    if pre_k > -1:
                        self.edges[pre_k].tasks.remove(self.users[n])
                    self.edges[like_k].tasks.append(self.users[n])

                selection[n] = like_k
                #print("match", (n, like_k))

            self.update_user_preference(selection)

        for n in range(self.number_of_user):
            if selection[n] == -1:
                for k in range(self.number_of_edge):
                    self.edges[k].tasks.append(self.users[n])
                    if self.edges[k].check():
                        selection[n] == k
                    else:
                        self.edges[k].tasks.remove(self.users[n])

        if policy == 0:
            return selection

        matched = []
        not_matched= []
        for n in range(self.number_of_user):
            if self.users[n].preference[0] == selection[n]:
                matched.append({
                    "user": n,
                    "edge": selection[n]
                })
            else:
                want_edge = self.users[n].preference[0]
                self.edges[want_edge].tasks.append(self.users[n])
                if self.edges[want_edge].check():
                    not_matched.append({
                        "user": n,
                        "now-edge": selection[n],
                        "want-edge": self.users[n].preference[0]
                    })
                self.edges[want_edge].tasks.remove(self.users[n])
        #print("selection:", selection)
        #print("matched", matched)
        #print("not_matched", not_matched)
        #_, _, _, _, _, reward_e, reward_t, _ = self.assign_compute_resource(partition=partition, selection=selection)
        #reward = reward_e * 1 + (1 - 1) * reward_t
        #print(">>>>>>>>>", reward)

        while len(not_matched) > 0:
            target = max(0, random.randint(0, len(not_matched)-1))
            want_edge = not_matched[target]["want-edge"]
            now_edge  = not_matched[target]["now-edge"]
            user_id   = not_matched[target]["user"]

            self.edges[want_edge].tasks.append(self.users[user_id])
            if self.edges[want_edge].check():
                # print("changed")
                selection[user_id] = want_edge
                if now_edge != -1:
                    self.edges[now_edge].tasks.remove(self.users[user_id])
            else:
                self.edges[want_edge].tasks.remove(self.users[user_id])

            self.update_user_preference(selection)

            not_matched = []
            for n in range(self.number_of_user):
                for j in range(len(self.users[n].preference)):
                    if self.users[n].preference[0] != selection[n]:
                        want_edge = self.users[n].preference[0]
                        self.edges[want_edge].tasks.append(self.users[n])
                        if self.edges[want_edge].check():
                            not_matched.append({
                                "user": n,
                                "now-edge": selection[n],
                                "want-edge": self.users[n].preference[0]
                            })
                            self.edges[want_edge].tasks.remove(self.users[n])
                            break
                        self.edges[want_edge].tasks.remove(self.users[n])
            #print(selection)
            #print("not_matched", not_matched)
            #_, _, _, _, _, reward_e, reward_t, _ = self.assign_compute_resource(partition=partition,selection=selection)
            #reward = reward_e * alpha + (1 - alpha) * reward_t
            #print(">>>>>>>>>", reward)

        return selection


    def TTC(self):
        sum_job_size = 0
        sum_job_computation = 0
        for n in range(self.number_of_user):
            sum_job_size += self.users[n].local_to_remote_size
            sum_job_computation += self.users[n].remote

        for k in range(self.number_of_edge):
            H_K = 0
            for n in range(self.number_of_user):
                H_K += self.users[n].H[k]
            self.users.sort(key=lambda x: x.priority(k, H_K, sum_job_size, sum_job_computation), reverse=True)
            order = []
            for user in self.users:
                order.append(user.task_id)
            self.edges[k].preference = order
            #print("preference ", k, order)

        self.users.sort(key=lambda x: x.task_id)
        #self.edges.sort(key=lambda x: x.id)

        self.update_user_preference()

        selection = [-1 for x in range(self.number_of_user)]

        stop = False

        while -1 in selection and not stop:

            stop = True

            while True:
                changed = False
                for n in range(self.number_of_user):
                    favt = self.users[n].preference[0]
                    if selection[n] == -1 and self.edges[favt].preference[0] == n:
                        self.edges[favt].tasks.append(self.users[n])
                        if self.edges[favt].check():
                            selection[n] = favt
                            for k in range(self.number_of_edge):
                                if n in self.edges[k].preference:
                                    self.edges[k].preference.remove(self.users[n].task_id)
                            stop = False
                            changed = True
                            # print(selection)
                            self.update_user_preference()
                        else:
                            self.users[n].preference.remove(favt)
                            self.edges[favt].tasks.remove(self.users[n])
                if not changed:
                    break

            for n in range(self.number_of_user):
                if selection[n] == -1:
                    visit = [0 for x in range(self.number_of_user)]
                    path = [n]
                    next = self.edges[self.users[n].preference[0]].preference[0]
                    while visit[next] == 0:
                        visit[next] = 1
                        path.append(next)
                        next = self.edges[self.users[next].preference[0]].preference[0]
                    start = -1
                    for i in range(len(path)):
                        if next == path[i]:
                            start = i
                            break
                    if start != -1:
                        path = path[start:]
                        for value in path:
                            favt = self.users[value].preference[0]
                            self.edges[favt].tasks.append(self.users[value])
                            if self.edges[favt].check():
                                selection[value] = favt
                                for k in range(self.number_of_edge):
                                    if value in self.edges[k].preference:
                                        self.edges[k].preference.remove(self.users[value].task_id)
                                stop = False
                            else:
                                self.users[value].preference.remove(favt)
                                #print(self.edges[favt].density)
                                self.edges[favt].tasks.remove(self.users[value])
                                #ds, _ = self.edges[favt].summary()
                                #print(">>>>>>>>", ds)
                        self.update_user_preference()
        return selection

    def lblj(self, policy=1):
        unassigned_users = copy.copy(self.users)
        c_edge = copy.copy(self.edges)

        selection = []

        sum_job_size = 0
        sum_job_computation = 0
        for n in range(self.number_of_user):
            sum_job_size += unassigned_users[n].local_to_remote_size
            sum_job_computation += unassigned_users[n].remote
            selection.append(-1)

        #for n in range(self.number_of_user):
            #print(unassigned_users[n].task_id, unassigned_users[n].local_to_remote_size, unassigned_users[n].remote/math.pow(10,9))

        #unassigned_users.sort(key=lambda x: x.priority(sum_job_size, sum_job_computation), reverse=True)

        run = 1
        stop = False

        while len(unassigned_users) > 0 and not stop:

            sum_bandwidth, sum_freq = self.edge_capacity()

            c_edge.sort(key=lambda x: x.priority(sum_bandwidth, sum_freq), reverse=True)

            #for n in range(len(c_edge)):
                #print("priority edge", c_edge[n].id, c_edge[n].priority(sum_bandwidth, sum_freq))

            k = 0

            #print("highest priority edge", c_edge[0].id)

            just_assigned = []
            stop = True

            H_K = 0
            for n in range(self.number_of_user):
                H_K += self.users[n].H[k]

            unassigned_users.sort(key=lambda x: x.priority(c_edge[k].id, H_K, sum_job_size, sum_job_computation), reverse=True)

            #unassigned_users.sort(key=lambda x: x.H[c_edge[k].id])

            for n in range(len(unassigned_users)):
                just_assigned.append(unassigned_users[n])
                c_edge[k].tasks.append(unassigned_users[n])
                """
                  minimum rate reqirement for (n, k, m)
                """
                #delta = unassigned_users[n].delta
                #c_edge[k].channel_allocation()
                #service_rate = c_edge[k].rate(unassigned_users[n].transmission_power)

                job_size = 0
                job_computation = 0
                for n_ in range(len(c_edge[k].tasks)):
                    job_size += c_edge[k].tasks[n_].local_to_remote_size
                    job_computation += c_edge[k].tasks[n_].remote

                a = 0.5
                weight = (job_size/sum_job_size)*a + (job_computation/sum_job_computation)*(1-a)

                if policy == 1:
                    if c_edge[k].check():  # and ( weight <= c_edge[k].weight or run >=2 )
                        selection[unassigned_users[n].task_id] = self.find_k_by_id(c_edge[k].id)
                        stop = False
                        break
                    else:
                        just_assigned.remove(unassigned_users[n])
                        c_edge[k].tasks.remove(unassigned_users[n])
                elif policy == 2:
                    if c_edge[k].check() and ( weight <= c_edge[k].weight or run >=2 ):
                        selection[unassigned_users[n].task_id] = self.find_k_by_id(c_edge[k].id)
                        stop = False
                        break
                    else:
                        just_assigned.remove(unassigned_users[n])
                        c_edge[k].tasks.remove(unassigned_users[n])
                elif policy == 3:
                    if c_edge[k].check():
                        selection[unassigned_users[n].task_id] = self.find_k_by_id(c_edge[k].id)
                        stop = False
                    else:
                        just_assigned.remove(unassigned_users[n])
                        c_edge[k].tasks.remove(unassigned_users[n])
                else:
                    if c_edge[k].check() and ( weight <= c_edge[k].weight or run >=2 ):
                        selection[unassigned_users[n].task_id] = self.find_k_by_id(c_edge[k].id)
                        stop = False
                    else:
                        just_assigned.remove(unassigned_users[n])
                        c_edge[k].tasks.remove(unassigned_users[n])

            for item in just_assigned:
                if item in unassigned_users:
                    unassigned_users.remove(item)
            run += 1
        return np.array(selection)

    def assign_compute_resource(self, partition=None, selection=None):

        for k in range(self.number_of_edge):
            self.edges[k].tasks = []

        for n in range(len(selection)):
            if selection[n] == -1:
                continue
            self.edges[selection[n]].tasks.append(self.users[n])

        for k in range(self.number_of_edge):
            self.edges[k].channel_allocation()

        info = []  # energy and time
        reward = [0, 0]
        fail = 0

        total_time = 0
        total_energy = 0
        total_local_time = 0
        total_local_energy = 0

        for n in range(self.number_of_user):
            delta = partition[n] # random.randint(1, self.users[n].DAG.length)
            cloud = selection[n] # random.randint(2, self.number_of_edge + self.number_of_user)
            if cloud == -1:
                delta = self.users[n].DAG.length - 1
            task = self.users[n].DAG
            energy = 0
            time = 0
            for m in range(0, delta+1):
                energy += task.jobs[m].computation * self.users[n].k * self.users[n].freq * self.users[n].freq
                time += task.jobs[m].computation / self.users[n].freq

            rate = 0
            if delta < self.users[n].DAG.length - 1:
                rate = self.edges[cloud].rate(self.users[n].transmission_power, self.users[n].H[self.edges[cloud].id])
                energy += self.users[n].transmission_power * task.jobs[delta].output_data \
                          / rate
                time += task.jobs[delta].output_data / rate
                for m in range(delta + 1, task.length):
                    time += task.jobs[m].computation / self.edges[cloud].freq

                waiting = 0

                self.users[n].remote_deadline = self.users[n].DAG.D / 1000 - self.users[n].local / self.users[n].freq - \
                                                self.users[n].local_to_remote_size / rate

                for item in self.edges[cloud].tasks:
                    if item.task_id == self.users[n].task_id:
                        continue
                    else:
                        #y = item.DAG.T * self.users[n].DAG.T /
                        waiting += item.remote / self.edges[cloud].freq
                #time += waiting

            info.append({
                  "sucess": time <= (self.users[n].DAG.D/1000),
                  "task"  : n,
                  "remote": [round(energy,5), round(time,3), round(rate/8000,1)],
                  "local" : [round(self.users[n].DAG.local_only_energy,5), round(self.users[n].DAG.local_only_time,3)],
                  "partition": [round(self.users[n].local / math.pow(10, 9),4), round(self.users[n].remote / math.pow(10, 9),4)],
                  "deadline" : round(task.D/1000,3),
                  "density": self.edges[cloud].get_density(task.task_id)
            })
            if time > self.users[n].DAG.D/1000:
                fail += 1

            reward[0] += 1 - energy / self.users[n].DAG.local_only_energy
            reward[1] += 1 - time / self.users[n].DAG.local_only_time

            total_local_energy += self.users[n].DAG.local_only_energy
            total_energy += energy
            total_time += time
            total_local_time += self.users[n].DAG.local_only_time

        return total_local_time/self.number_of_user, total_local_energy, total_energy\
            , fail, info, reward[0]/self.number_of_user, reward[1]/self.number_of_user, total_time/self.number_of_user

    def get_comp(self, n , m):
        comp = 0
        for m_ in range(m+1):
            comp += self.users[n].DAG.jobs[m_].computation
        return comp

    def clean(self):
        for k in range(self.number_of_edge):
            self.edges[k].tasks = []
            self.edges[k].queue = []
        for n in range(self.number_of_user):
            self.users[n].local = 0
            self.users[n].remote = 0
            self.users[n].queue = []

    def edge_capacity(self):
        sum_bandwidth = 0
        sum_freq = 0
        for k in range(self.number_of_edge):
            sum_bandwidth += self.edges[k].update_bandwidth()
            sum_freq += self.edges[k].freq
        return sum_bandwidth, sum_freq

    def update_user_preference(self, selection):
        sum_bandwidth, sum_freq = self.edge_capacity()
        #print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        for user in self.users:
            dist = [[self.edges[x].id, user.edge_preference(selection[user.task_id], self.edges[x], sum_bandwidth, sum_freq)] for x in range(self.number_of_edge)]
            dist.sort(key=lambda x: x[1], reverse=True)
            order = []
            for k in range(self.number_of_edge):
                order.append(dist[k][0])
                user.preference = order
            #print("user preference ", user.task_id, dist)

    """
        def get_task_min_rate(self, n):
        jobs = self.users[n].DAG.jobs
        hist = []
        for m in range(self.users[n].DAG.length - 1):
            r1 = jobs[m].output_data * self.users[n].transmission_power / \
                (self.users[n].k * (self.get_comp(n, self.users[n].DAG.length-1)-self.get_comp(n, m)) * math.pow(self.users[n].freq,2))
            hist.append(m, round(r1/8000, 1))
        return np.min(hist)

    def task_min_rate_k_m(self, n,m,k):
        jobs = self.users[n].DAG.jobs
        locol = self.get_comp(n, m) / self.users[n].freq
        remote = (self.get_comp(n, self.users[n].DAG.length - 1) - self.get_comp(n, m)) / self.edges[k].freq
        r1 = jobs[m].output_data / (self.users[n].DAG.D / 1000 - locol - remote)
        return  r1

    def task_min_rate_list(self, n):
        jobs = self.users[n].DAG.jobs
        hist = []
        for m in range(self.users[n].DAG.length):
            r = []
            for k in range(self.number_of_edge):
                locol = self.get_comp(n, m) / self.users[n].freq
                remote = (self.get_comp(n, self.users[n].DAG.length - 1) - self.get_comp(n, m))/self.edges[k].freq
                r1 = jobs[m].output_data  / (self.users[n].DAG.D/1000 - locol - remote)
                #print(locol, remote, self.users[n].DAG.D/1000)
                r.append(round(r1/8000,1))
            hist.append(r)
        return hist

    """
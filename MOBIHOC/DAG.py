import random
import math


class JOB:
    # data unit bit
    def __init__(self, job_id, computation, input_data, output_data, task_id, pred_id, succ_id, resource):

        self.computation= computation
        self.output_data = output_data
        self.input_data = input_data
        self.data = 0

        self.deadtime = 0

        self.job_id = job_id
        self.task_id = task_id
        self.pred_id = pred_id
        self.succ_id = succ_id


class DAG:
    def __init__(self, task_id, f_local, T=None, D=None):
        self.length = random.randint(2, 4)
        # time interval = 50 ms
        if T is not None:
            self.T = T
        else:
            self.T = random.randint(10, 20) * 50

        if D is not None:
            self.D = D
        else:
            self.D = random.randint(10, 20) * 50

        self.task_id = task_id
        self.f_local = f_local

        self.valid = []

        self.timer = 0

        self.release_id = 0

        self.compute = None
        self.channel = None
        self.edges = None
        self.jobs = None

        self.local_only_energy = 0
        self.local_only_time = 0

        #print("Task id=", self.task_id, ", local CPU =", f_local, ", DAG-jobs[ T=", self.T, ",D=", self.D, "]")

    def display_edge(self):
        for link in self.edges:
            print(link)

    def local_only_compute_energy_time(self, freq, k):
        total_computation = 0
        for m in range(self.length):
            total_computation += self.jobs[m].computation
        # optimal_cpu_cycles = total_computation / (self.D * math.pow(10, -3))
        self.local_only_energy = k * total_computation * freq * freq
        self.local_only_time = total_computation / freq
        #print("local-only energy", round(self.local_only_energy, 4), "response time", round(self.local_only_time, 4))

    """
       5000 KB =    10^9 hz  = 1 GHz
       1 KB = 8000 bit
       10^9 / (5000 * 8000) = 25 / bit
       250 KB - 500 KB
    """

    def create_from_config(self, config):
        self.length = len(config)
        self.jobs = []
        for m in range(self.length):
            pred_id = m - 1
            succ_id = m + 1
            self.jobs.append(JOB(m + 1, config[m][1] * math.pow(10, 9), config[m][0] * 8000, config[m][2] * 8000
                                 , self.task_id, pred_id, succ_id, None))

    def create(self, freq):
        self.jobs = []
        output_data = int(random.randint(1000, 1500)) * 8000
        # density = random.randint(0, self.length - 1)
        for m in range(self.length):
            input_data = output_data
            output_data = int(random.uniform(250, 700)) * 8000
            if m <= 0:
                # computation = int(input_data * complexity)  # input_data
                computation = random.uniform(0.05, 0.08) * freq
            else:
                # computation = int(input_data * low_complexity)  # input_data
                computation = random.uniform(0.15, 0.35) * freq / (self.length - 1)
            pred_id = m - 1
            succ_id = m + 1
            self.jobs.append(JOB(m+1, computation, input_data, output_data
                                 , self.task_id, pred_id, succ_id, None))
        self.jobs[-1].output_data = 0

    def display(self):
        """
                    DAG-job display

        output_str = "DAG:"
        for m in range(self.length):
            output_str += "(" + str(self.jobs[m].input_data / 8000) + "KB," + str(
                self.jobs[m].computation / math.pow(10, 9)) \
                          + "," + str(self.jobs[m].output_data / 8000) + "KB)->"
        print(output_str)
        """
        job_dict = []
        for m in range(self.length):
            job_dict.append([round(self.jobs[m].input_data/8000,0)
                                , round(self.jobs[m].computation/math.pow(10, 9), 5)
                                , round(self.jobs[m].output_data/8000, 5)])
        return job_dict


    def get_valid_partition(self):
        self.valid = [0]
        for m in range(1, self.length):
            if self.jobs[m].output_data < self.jobs[0].output_data:
                self.valid.append(m)





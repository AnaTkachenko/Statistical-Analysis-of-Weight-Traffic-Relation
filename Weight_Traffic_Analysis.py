# -- Anastasiia Tkachenko RA and TA

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import csv
import warnings
from scipy.stats import chi2_contingency
import copy
import math
from scipy.stats import spearmanr, kendalltau, pearsonr
import plotly.express as px

warnings.filterwarnings("ignore")


def makeFreqTabER(W, T, size_W, size_T):
    ### AT: Equi-width ranges:
    min_W, max_W = np.min(W), np.max(W)
    T_copy = copy.deepcopy(T)
    T_copy.sort()
    i = 0
    while T_copy[i]==float('-inf'):
        i+=1
    min_T, max_T = T_copy[i], np.max(T)
    table = np.zeros([size_W, size_T])
    for i in range(size_W-1):
        for j in range(size_T-1):
            for l in range(len(W)):
                if ((min_W + (max_W - min_W) / size_W * i) <= W[l] < (min_W + (max_W - min_W) / size_W * (i + 1)) and
                    (min_T + (max_T - min_T) / size_T * j) <= T[l] < (min_T + (max_T - min_T) / size_T * (j + 1))):
                    table[i][j] +=1

    for j in range(size_T-1):
        for l in range(len(W)):
            if ((min_W + (max_W - min_W) / size_W * (size_W-1)) <= W[l] <= max_W and
                (min_T + (max_T - min_T) / size_T * j) <= T[l] < (min_T + (max_T - min_T) / size_T * (j + 1))):
                table[size_W-1][j] +=1

    for i in range(size_W-1):
        for l in range(len(W)):
            if ((min_W + (max_W - min_W) / size_W * i) <= W[l] < (min_W + (max_W - min_W) / size_W * (i + 1)) and
                (min_T + (max_T - min_T) / size_T * (size_T-1)) <= T[l] <= max_T):
                table[i][size_T-1] +=1

    for l in range(len(W)):
        if ((min_W + (max_W - min_W) / size_W * (size_W-1)) <= W[l] <= max_W and
                (min_T + (max_T - min_T) / size_T * (size_T - 1)) <= T[l] <= max_T):
            table[size_W-1][size_T-1] += 1

    for i in range(size_W - 1):
        for l in range(len(W)):
            if ((min_W + (max_W - min_W) / size_W * (size_W-1)) <= W[l] <= max_W and
                T[l]==float('-inf')):
                table[i][0] += 1

    return table

def makeFreqTabED(W, T, size_W, size_T):
    ### AT: Equi-depth ranges:
    n = len(W)
    step_W = round(n/size_W)
    step_T = round(n/size_T)
    # print("number of elements: min and max: ", step_W, n-(size_W-1)*step_W)
    if min(step_W, n-(size_W-1)*step_W) <= 5: #and (n-(size_W-1)*step_W!=0):
        print("not enough elements in W")
        return None
    if min(step_T, n-(size_T-1)*step_T) <= 5: # and (n-(size_T-1)*step_T)!=0:
        print("not enough elements in T")
        return None

    W_copy = copy.deepcopy(W)
    T_copy = copy.deepcopy(T)
    W_copy.sort()
    T_copy.sort()
    div_poin_W = [W_copy[i*step_W] for i in range(size_W)]+[W_copy[-1]]
    div_poin_T = [T_copy[i*step_T] for i in range(size_T)]+[T_copy[-1]]
    table = np.zeros([size_W, size_T])
    for i in range(size_W-1):
        for j in range(size_T-1):
            for l in range(len(W)):
                if ((div_poin_W[i] <= W[l] < div_poin_W[i+1]) and (div_poin_T[j] <= T[l] < div_poin_T[j+1])):
                    table[i][j] += 1

    for j in range(size_T-1):
        for l in range(len(W)):
            if ((div_poin_W[-2] <= W[l] <= div_poin_W[-1]) and
                (div_poin_T[j] <= T[l] < div_poin_T[j+1])):
                table[size_W-1][j] +=1

    for i in range(size_W-1):
        for l in range(len(W)):
            if (div_poin_W[i] <= W[l] < div_poin_W[i+1]) and (div_poin_T[-2] <= T[l] <= div_poin_T[-1]):
                table[i][size_T-1] +=1

    for l in range(len(W)):
        if (div_poin_W[-2] <= W[l] <= div_poin_W[-1]) and (div_poin_T[-2] <= T[l] <= div_poin_T[-1]):
            table[size_W-1][size_T-1] += 1

    return table


class ParAd:
    def __init__(self, dataset):
        self.total = dataset['TOTAL'].to_numpy()
        self.weight = dataset['MEAN_WEIGHT'].to_numpy()
        self.incoming = dataset['IN'].to_numpy()
        self.outgoing = dataset['OUT'].to_numpy()
        self.ImO = dataset['IN'].to_numpy() - dataset['OUT'].to_numpy()
        self.IpO = dataset['IN'].to_numpy() + dataset['OUT'].to_numpy()
        self.lateral = dataset['LATERAL'].to_numpy()
        self.time = dataset['DATETIME'].to_numpy()

    def time_period(self, time_period):
        a = time_period[0]
        b = time_period[1]
        records = self.time[a:b]
        dates = np.array([i[0:10] for i in records])
        labels = []
        steps = []
        counter = 1
        for i, j in enumerate(dates[:-1]):
            if j != dates[i + 1]:
                labels.append(j)
                steps.append(counter)
                counter = 1
            else:
                counter += 1
            if i == len(dates) - 2:
                steps.append(counter)
        labels.append(dates[-1])

        return labels, steps

    @staticmethod
    def statistics(tp, data):
        mean = np.mean(data[tp[0]: tp[1]])
        var = np.var(data[tp[0]: tp[1]])
        return mean, var


    def dW_dT_Stats(self, whole_tp, quarterhours, name):
        warnings.filterwarnings("ignore")
        with open(name, 'w', newline='') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            wr.writerow(['WEIGHT',
                         'VarIn', 'VarOut', 'VarTotal', 'VarImO', 'VarIpO',
                         'EIn', 'EOut', 'ETotal','EImO', 'EIpO'])

        _, steps = self.time_period(whole_tp)
        startD = whole_tp[0]
        var_o, var_in, var_t = [0], [0], [0]
        var_ImO, var_IpO = [0], [0]
        mean_o, mean_in, mean_t = [0], [0], [0]
        mean_ImO, mean_IpO = [0], [0]
        weight = [self.weight[startD]]
        # looping over days
        for i in range(0, len(steps)):
            endD = startD + steps[i]
            # looping over quarter hours in the day
            for j in range(startD, endD, quarterhours):
                if j < endD - quarterhours - 1:
                    weight.append(self.weight[j + quarterhours])

                    var_in.append(self.statistics([j, j+quarterhours], self.incoming)[1])
                    var_o.append(self.statistics([j, j+quarterhours], self.outgoing)[1])
                    var_t.append(self.statistics([j, j+quarterhours], self.total)[1])
                    var_ImO.append(self.statistics([j, j + quarterhours], self.incoming-self.outgoing)[1])
                    var_IpO.append(self.statistics([j, j + quarterhours], self.incoming+self.outgoing)[1])

                    mean_o.append(self.statistics([j, j + quarterhours], self.incoming)[0])
                    mean_in.append(self.statistics([j, j + quarterhours], self.outgoing)[0])
                    mean_t.append(self.statistics([j, j + quarterhours], self.total)[0])
                    mean_ImO.append(self.statistics([j, j + quarterhours], self.incoming - self.outgoing)[0])
                    mean_IpO.append(self.statistics([j, j + quarterhours], self.incoming + self.outgoing)[0])
                else:
                    if j != endD - 1:
                        weight.append(self.weight[endD-1])

                        var_in.append(self.statistics([j, endD], self.incoming)[1])
                        var_o.append(self.statistics([j, endD], self.outgoing)[1])
                        var_t.append(self.statistics([j, endD], self.total)[1])
                        var_ImO.append(self.statistics([j, endD], self.incoming - self.outgoing)[1])
                        var_IpO.append(self.statistics([j, endD], self.incoming + self.outgoing)[1])

                        mean_in.append(self.statistics([j, endD], self.incoming)[0])
                        mean_o.append(self.statistics([j, endD], self.outgoing)[0])
                        mean_t.append(self.statistics([j, endD], self.total)[0])
                        mean_ImO.append(self.statistics([j, endD], self.incoming - self.outgoing)[0])
                        mean_IpO.append(self.statistics([j, endD], self.incoming + self.outgoing)[0])
                        break

            with open(name, 'a', newline='') as myfile:
                wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
                for index in range(1, len(weight)):
                    wr.writerow([abs(weight[index]-weight[index-1]),
                                np.log(abs(var_in[index] - var_in[index-1])),
                                np.log(abs(var_o[index] - var_o[index-1])),
                                np.log(abs(var_t[index] - var_t[index-1])),
                                np.log(abs(var_ImO[index] - var_ImO[index - 1])),
                                np.log(abs(var_IpO[index] - var_IpO[index - 1])),
                                np.log(abs(mean_in[index] - mean_in[index - 1])),
                                np.log(abs(mean_o[index] - mean_o[index - 1])),
                                np.log(abs(mean_t[index] - mean_t[index - 1])),
                                np.log(abs(mean_ImO[index] - mean_ImO[index - 1])),
                                np.log(abs(mean_IpO[index] - mean_IpO[index - 1]))
                                ])
            startD = endD
            var_o, var_in, var_t = [0], [0], [0]
            var_ImO, var_IpO = [0], [0]
            mean_o, mean_in, mean_t = [0], [0], [0]
            mean_ImO, mean_IpO = [0], [0]
            weight = [self.weight[startD]]



    def dW_dT_values(self, whole_tp, quarterhours, name):
        warnings.filterwarnings("ignore")
        with open(name, 'w', newline='') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            wr.writerow(['WEIGHT',
                         'In', 'Out', 'Total', 'In-Out', 'In+Out'])

        _, steps = self.time_period(whole_tp)
        startD = whole_tp[0]
        I, O, T = [0], [0], [0]
        I_O, IO = [0], [0]
        weight = [self.weight[startD]]
        # looping over days
        for i in range(0, len(steps)):
            endD = startD + steps[i]
            # looping over quarter hours in the day
            for j in range(startD, endD, quarterhours):
                if j < endD - quarterhours - 1:
                    weight.append(self.weight[j + quarterhours])

                    I.append(sum(self.incoming[j: j + quarterhours]))
                    O.append(sum(self.outgoing[j: j + quarterhours]))
                    T.append(sum(self.total[j: j + quarterhours]))
                    I_O.append(sum(self.incoming[j: j + quarterhours] - self.outgoing[j: j + quarterhours]))
                    IO.append(sum(self.incoming[j: j + quarterhours] + self.outgoing[j: j + quarterhours]))

                else:
                    if j != endD - 1:
                        weight.append(self.weight[endD - 1])

                        I.append(sum(self.incoming[j: endD]))
                        O.append(sum(self.outgoing[j: endD]))
                        T.append(sum(self.total[j: endD]))
                        I_O.append(sum(self.incoming[j: endD] - self.outgoing[j: endD]))
                        IO.append(sum(self.incoming[j: endD] + self.outgoing[j: endD]))
                        break

            with open(name, 'a', newline='') as myfile:
                wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
                for index in range(1, len(weight)):
                    wr.writerow([abs(weight[index] - weight[index - 1]),
                                 np.log(abs(I[index] - I[index - 1])),
                                 np.log(abs(O[index] - O[index - 1])),
                                 np.log(abs(T[index] - T[index - 1])),
                                 np.log(abs(I_O[index] - I_O[index - 1])),
                                 np.log(abs(IO[index] - IO[index - 1]))
                                 ])
            startD = endD
            I, O, T = [0], [0], [0]
            I_O, IO = [0], [0]
            weight = [self.weight[startD]]

    def W_T_values(self, whole_tp, quarterhours, name):
        warnings.filterwarnings("ignore")
        with open(name, 'w', newline='') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            wr.writerow(['WEIGHT',
                         'In', 'Out', 'Total', 'In-Out', 'In+Out'])

        _, steps = self.time_period(whole_tp)
        startD = whole_tp[0]
        I, O, T, I_O, IO, weight = [], [], [], [], [], []
        n = len(steps)
        # looping over days
        for i in range(n):
            endD = startD + steps[i]
            # looping over quarter hours in the day
            for j in range(startD, endD-quarterhours, quarterhours):
                weight.append(sum(self.weight[j:j + quarterhours]))
                I.append(sum(self.incoming[j: j + quarterhours]))
                O.append(sum(self.outgoing[j: j + quarterhours]))
                T.append(sum(self.total[j: j + quarterhours]))
                I_O.append(sum(self.incoming[j: j + quarterhours] - self.outgoing[j: j + quarterhours]))
                IO.append(sum(self.incoming[j: j + quarterhours] + self.outgoing[j: j + quarterhours]))


            with open(name, 'a', newline='') as myfile:
                wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
                for index in range(len(weight)):
                    wr.writerow([weight[index], I[index], O[index],
                                 T[index], I_O[index], IO[index]])
            startD = endD
            I, O, T, I_O, IO, weight = [], [], [], [], [], []


    def Probability(self, filename, index, eps):
        rows = []
        with open(filename, 'r') as csvfile:
            # creating a csv reader object
            csvreader = csv.reader(csvfile)
            fields = next(csvreader)
            for row in csvreader:
                rows.append(row)
        rows_T = []
        for row in rows:
            helper = []
            for i, value in enumerate([0, index]):
                if float(row[value]) < eps[i]:
                    helper.append(0)
                else:
                    helper.append(1)
            rows_T.append(helper)

        count_joint = sum([1 for row in rows_T if (row[0]==1 and row[1]==1)])
        count_nonzero_T = sum([1 for row in rows_T if (row[1]==1)])
        count_nonzero_w = sum([1 for row in rows_T if (row[0]==1)])
        a = len(rows_T)
        return count_joint/a, count_nonzero_T/a * count_nonzero_w/a, count_nonzero_T/a, count_nonzero_w/a

    def MakeBinary(self, filename, name, index, eps):
        with open(name, 'w', newline='') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            wr.writerow(['WEIGHT', str(index)])
        rows = []
        with open(filename, 'r') as csvfile:
            # creating a csv reader object
            csvreader = csv.reader(csvfile)
            fields = next(csvreader)
            for row in csvreader:
                rows.append(row)
        for row in rows:
            helper = []
            for i, value in enumerate([0, index]):
                if float(row[value]) < eps[i]:
                    helper.append(0)
                else:
                    helper.append(1)
            with open(name, 'a', newline='') as myfile:
                wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
                wr.writerow(helper)

    def max_ProbDif(self, maxes, filename):
        data_transformed = pd.read_csv(filename)
        weights_eps_set = list(data_transformed.iloc[:,0])
        weights_eps_set.sort()
        max_argmax = []
        suprema = [max(weights_eps_set)]
        if isinstance(maxes, np.float64): numMes = 2
        else: numMes = len(maxes)
        for T_mes in range(1, numMes):
            max_dif = 0
            arg_md = [0, 0]
            traffic_eps_set = list(data_transformed.iloc[:, T_mes])
            traffic_eps_set.sort()
            traffic_eps_set = np.array(traffic_eps_set)
            traffic_eps_set_ind = np.where(traffic_eps_set != -float('inf'))
            traffic_eps_set = traffic_eps_set[traffic_eps_set_ind]
            suprema.append(max(traffic_eps_set))
            for i in weights_eps_set:
                for j in traffic_eps_set:
                    a,b,_,_ = self.Probability(filename, T_mes, [i, j])
                    if max_dif < abs(a-b):
                        max_dif = abs(a-b)
                        arg_md = [i,j]
            max_argmax.append([max_dif, arg_md[0], arg_md[1], self.Probability(filename, T_mes,[arg_md[0], arg_md[1]])[2], self.Probability(filename, T_mes,[arg_md[0], arg_md[1]])[3]])
        return max_argmax, suprema


if __name__ == '__main__':

    pass
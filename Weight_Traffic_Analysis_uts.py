# -- Anastasiia Tkachenko RA and TA
import unittest
from Weight_Traffic_Analysis import ParAd, makeFreqTabED
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, spearmanr, kendalltau, pearsonr

"""
Please, replace file_path accordingly.
"""

class WT_uts(unittest.TestCase):

    def test_correlation_raw_data(self):
        print('\n******** Correlation: Pearson, Spearman, Kendall')
        H = 17
        print("hive: H", H, ", correlation on raw data")
        file_path = '../az_data_adj/'
        data = pd.read_csv(file_path+'4_' + str(H) + '_az_bmc_with_weights_srt.csv')
        md = ParAd(data)
        Weight, Traffic = md.weight, md.incoming #md.incoming, md.outgoing, md.total, md.ImP, md.IpP
        coef, p = pearsonr(Weight, Traffic)
        assert abs(coef + 0.244) < 0.001
        assert abs(p - 0) < 0.001
        print('Pearson correlation coefficient and p-value: {:.3f}, {:.3f}'.format(coef, p))
        coef, p = spearmanr(Weight, Traffic)
        assert abs(coef + 0.344) < 0.001
        assert abs(p - 0) < 0.001
        print('Spearman correlation coefficient and p-value: {:.3f}, {:.3f}'.format(coef, p))
        coef, p = kendalltau(Weight, Traffic)
        assert abs(coef + 0.236) < 0.001
        assert abs(p - 0) < 0.001
        print('Kendall correlation coefficient and p-value: {:.3f}, {:.3f}'.format(coef, p))

        print('\n******** Correlation: Pearson, Spearman, Kendall: passed...')

    def test_correlation_lagged_data(self):
        print('\n******** Correlation: Pearson on lagged Weight and Traffic')
        H = 17
        print("hive: H", H, ", correlation on lagged data")
        file_path = '../az_data_adj/'
        data = pd.read_csv(file_path + '4_' + str(H) + '_az_bmc_with_weights_srt.csv')
        md = ParAd(data)
        time_period, k = [0, len(md.time) - 1], 24
        file_name = 'W_T_lag' + str(k // 4) + '.csv'
        md.W_T_values(time_period, k, file_name)

        """
        AT: W_T_values makes csv file, which columns are 
        ['WEIGHT','In', 'Out', 'Total', 'In-Out', 'In+Out'].
        lagged_data means that we combine(sum) values of weight/traffic on intervals of length k. 
        """

        dataset = pd.read_csv(file_name)
        lag_Weight, lag_Traffic = dataset['WEIGHT'].to_numpy(), dataset['In'].to_numpy()
        """
        AT: replacing 'In' with 'Out', 'Total', 'In-Out', or 'In+Out', 
        you will get correlation coefficient and p-value for outgoing,
        total, difference between incoming and outgoing, or sum of incoming 
        and outgoing, respectively.
        """
        c, p = pearsonr(lag_Weight, lag_Traffic)
        assert abs(c + 0.496) < 0.001
        assert abs(p - 0) < 0.001
        print('({:.3f},{:.2e})'.format(c, p))

        print('\n******** Correlation: Pearson on lagged Weight and Traffic: passed...')

    def test_maxD_eW_eT_Pr_exact(self):
        print('\n******** MaxDif, argmaxima, marg.probs at argmaxima: exact counts')
        H = 17
        print("hive: H", H, ", exact")
        file_path = '../az_data_adj/'
        data = pd.read_csv(file_path+'4_' + str(H) + '_az_bmc_with_weights_srt.csv')
        md = ParAd(data)
        time_period, k = [0, len(md.time) - 1], 24
        file_name = 'dW_dT_lag'+str(k//4)+'_EC.csv'
        md.dW_dT_values(time_period, k, file_name)
        """
        AT: dW_dT_values makes csv file, which columns are 
        ['WEIGHT','In', 'Out', 'Total', 'In-Out', 'In+Out'].
        Each entry is of format delta_k Z_t, where Z = ['WEIGHT','In', 'Out', 'Total', 'In-Out', 'In+Out']. 
        """
        dataset = pd.read_csv(file_name)
        Traffic = dataset['In'] #'In', 'Out', 'Total', 'In-Out', 'In+Out'
        maxes = Traffic.max()
        exact = md.max_ProbDif(maxes, file_name)[0]
        """
                AT: max_ProbDif can handle multiple traffic measurements
                See test_maxD_eW_eT_Pr_stats for more details
        """
        assert abs(exact[0][0] - 0.077) < 0.001
        assert abs(exact[0][1] - 0.181) < 0.001
        assert abs(exact[0][2] - 7.840) < 0.001
        assert abs(exact[0][3] - 0.643) < 0.001
        assert abs(exact[0][4] - 0.531) < 0.001
        print("Maximum Difference & corresponding argmaxima (e_W, e_T): {:.3f}, ({:.3f}, {:.3f})".format(exact[0][0], exact[0][1], exact[0][2]))
        print("Marginals probabilities at argmaxima e_W & e_T: {:.3f}, {:.3f}".format(exact[0][3], exact[0][4]))
        print('\n******** MaxDif, argmaxima, marg.probs at argmaxima: exact counts: passed...')

    def test_maxD_eW_eT_Pr_stats(self):
        print('\n******** MaxDif, argmaxima, marg.probs at argmaxima: stats')
        H = 17
        print("hive: H", H, ", statistics")
        file_path = '../az_data_adj/'
        data = pd.read_csv(file_path+'4_' + str(H) + '_az_bmc_with_weights_srt.csv')
        md = ParAd(data)
        time_period, k = [0, len(md.time) - 1], 24
        file_name = 'dW_dT_lag'+str(k//4)+'_STATS.csv'
        md.dW_dT_Stats(time_period, k, file_name)
        """
        AT: dW_dT_Stats makes csv file, which columns are 
        ['WEIGHT','VarIn', 'VarOut', 'VarTotal', 'VarImO', 
        'VarIpO', 'EIn', 'EOut', 'ETotal','EImO', 'EIpO'].
        
        Each entry is of format delta_k Z_t, where Z = ['WEIGHT','VarIn', 'VarOut', 'VarTotal', 'VarImO', 
        'VarIpO', 'EIn', 'EOut', 'ETotal','EImO', 'EIpO']. 
        """
        dataset = pd.read_csv(file_name)
        Traffic = dataset['VarIn'] #'VarIn', 'VarOut', 'VarTotal', 'VarImO', 'VarIpO', 'EIn', 'EOut', 'ETotal','EImO', 'EIpO'
        maxes = Traffic.max()
        stats = md.max_ProbDif(maxes, file_name)[0]

        """
        AT: max_ProbDif can handle multiple traffic measurements, e.g.:
        Traffic = dataset[1:] 
        maxes = Traffic.max()
        stats = md.max_ProbDif(maxes, file_name)[0]
        
        >>> [[0.09829237817576003, 0.1125119999999952, 8.453513190533554, 0.7244897959183674, 0.6530612244897959],  <--- 'VarIn'
        [0.10474802165764263, 0.1317162799999991, 10.364452789143227, 0.47959183673469385, 0.6326530612244898],     <--- 'VarOut'
        [0.09600166597251145, 0.4039160000000024, 12.007620690545656, 0.3673469387755102, 0.37755102040816324],     <--- 'VarTotal'
        [0.06768013327780092, 0.1317162799999991, 7.077375624603007, 0.4897959183673469, 0.6326530612244898],       <--- 'VarImO'
        [0.09985422740524783, 0.4039160000000024, 11.861523440341395, 0.35714285714285715, 0.37755102040816324],    <--- 'VarIpO'
        [0.08663057059558521, 0.1523817499999964, 4.768209813232596, 0.5918367346938775, 0.6122448979591837],       <--- 'EIn'
        [0.08163265306122452, 0.2141694099999966, 4.703883658924124, 0.5714285714285714, 0.5],                      <--- 'EOut'
        [0.08673469387755106, 0.2141694099999966, 5.547615875339098, 0.5612244897959183, 0.5],                      <--- 'ETotal'
        [0.04966680549770927, 0.3525654400000064, 2.1629124323474405, 0.6632653061224489, 0.3979591836734694],      <--- 'EImO'
        [0.09183673469387754, 0.2141694099999966, 5.448531737839488, 0.5714285714285714, 0.5]]                      <--- 'EIpO'
        """
        assert abs(stats[0][0] - 0.098) < 0.001
        assert abs(stats[0][1] - 0.113) < 0.001
        assert abs(stats[0][2] - 8.454) < 0.001
        assert abs(stats[0][3] - 0.724) < 0.001
        assert abs(stats[0][4] - 0.653) < 0.001
        print("Maximum Difference & corresponding argmaxima (e_W, e_T): {:.3f}, ({:.3f}, {:.3f})".format(stats[0][0], stats[0][1], stats[0][2]))
        print("Marginals probabilities at argmaxima e_W & e_T: {:.3f}, {:.3f}".format(stats[0][3], stats[0][4]))
        print('\n******** MaxDif, argmaxima, marg.probs at argmaxima: stats: passed...')

    def test_suprema_eW_eT_exact(self):
        print('\n******** Suprema of e_W & e_T: exact counts')
        H = 17
        print("hive: H", H, ", exact")
        file_path = '../az_data_adj/'
        data = pd.read_csv(file_path+'4_' + str(H) + '_az_bmc_with_weights_srt.csv')
        md = ParAd(data)
        time_period, k = [0, len(md.time) - 1], 24
        file_name = 'dW_dT_lag'+str(k//4)+'_EC.csv'
        dataset = pd.read_csv(file_name)
        Traffic = dataset['In'] #'In', 'Out', 'Total', 'In-Out', 'In+Out'
        maxes = Traffic.max()
        suprema = md.max_ProbDif(maxes, file_name)[1]
        assert abs(suprema[0] - 1.455) < 0.001
        assert abs(suprema[1] - 9.254) < 0.001

        print("Supremum of eW: {:.3f}".format(suprema[0]))
        print("Supremum of eT: {:.3f}".format(suprema[1]))
        print('\n******** Suprema of e_W & e_T: exact counts: passed...')

    def test_suprema_eW_eT_stats(self):
        print('\n******** Suprema of e_W & e_T: stats')
        H = 17
        print("hive: H", H, ", statistics")
        file_path = '../az_data_adj/'
        data = pd.read_csv(file_path+'4_' + str(H) + '_az_bmc_with_weights_srt.csv')
        md = ParAd(data)
        time_period, k = [0, len(md.time) - 1], 24
        file_name = 'dW_dT_lag'+str(k//4)+'_STATS.csv'
        dataset = pd.read_csv(file_name)
        Traffic = dataset['VarIn'] #'VarIn', 'VarOut', 'VarTotal', 'VarImO', 'VarIpO', 'EIn', 'EOut', 'ETotal','EImO', 'EIpO'
        maxes = Traffic.max()
        suprema = md.max_ProbDif(maxes, file_name)[1]
        assert abs(suprema[0] - 1.455) < 0.001
        assert abs(suprema[1] - 11.672) < 0.001

        print("Supremum of eW: {:.3f}".format(suprema[0]))
        print("Supremum of eT: {:.3f}".format(suprema[1]))
        print('\n******** Suprema of e_W & e_T: stats: passed...')

    def test_chi2_exact(self):
        print('\n******** Chi square test of independence of dW & dT: exact counts')
        H = 17
        print("hive: H", H, ", exact")
        file_path = '../az_data_adj/'
        file_path = '../az_data_adj/'
        data = pd.read_csv(file_path + '4_' + str(H) + '_az_bmc_with_weights_srt.csv')
        md = ParAd(data)
        time_period, k = [0, len(md.time) - 1], 24
        file_name = 'dW_dT_lag'+str(k//4)+'_EC.csv'
        dataset = pd.read_csv(file_name)
        numBins = 4
        dW, dT = dataset['WEIGHT'].to_numpy(), dataset['In'].to_numpy()
        table = makeFreqTabED(dW, dT, numBins, numBins)
        c, p, _, _ = chi2_contingency(table)
        assert abs(c - 10.964) < 0.001
        assert abs(p - 0.278)  < 0.001
        print('Chi square statistic and p-value: {:.3f} {:.3f}'.format(c, p))
        print('\n******** Chi square test of independence of dW & dT: exact counts: passed...')

    def test_chi2_stats(self):
        print('\n******** Chi square test of independence of dW & dT: stats')
        H = 17
        print("hive: H", H, ", statistics")
        file_path = '../az_data_adj/'
        file_path = '../az_data_adj/'
        data = pd.read_csv(file_path + '4_' + str(H) + '_az_bmc_with_weights_srt.csv')
        md = ParAd(data)
        time_period, k = [0, len(md.time) - 1], 24
        file_name = 'dW_dT_lag'+str(k//4)+'_STATS.csv'
        dataset = pd.read_csv(file_name)
        numBins = 4
        dW, dT = dataset['WEIGHT'].to_numpy(), dataset['VarIn'].to_numpy()
        table = makeFreqTabED(dW, dT, numBins, numBins)
        c, p, _, _ = chi2_contingency(table)
        assert abs(c - 26.373) < 0.001
        assert abs(p - 0.002)  < 0.001
        print('Chi square statistic and p-value: {:.3f} {:.3f}'.format(c, p))
        print('\n******** Chi square test of independence of dW & dT: stats: passed...')

    def runTest(self):
        pass


if __name__ == '__main__':
    unittest.main()

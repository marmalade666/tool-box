'''
    __author__ = 'sladesal'
    __time__ = '20171128'
    __bolg__ = 'www.shataowei.com'
'''

from sklearn.feature_selection import VarianceThreshold
import numpy as np
import pandas as pd
import sys
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import GradientBoostingClassifier

'''
    k_var : 方差选择需要满足的最小值
    pearson_value_k : 想剔除的feature个数
    vif_k ： 想剔除的feature个数
    wrapper_k ：想保留的feature个数
    way ： 'l1'正则或者'l2'正则
    C_0 : 惩罚力度
'''


class feature_filter:
    def __init__(self, k_var=None, pearson_value_k=None, vif_k=None, wrapper_k=None, C_0=0.1, way='l2'):
        self.k_var = k_var
        self.pearson_value_k = pearson_value_k
        self.vif_k = vif_k
        self.wrapper_k = wrapper_k
        self.way = way
        self.C_0 = C_0

    # 方差选择法
    def var_filter(self, data):
        k = self.k_var
        var_data = data.var().sort_values()
        if k is not None:
            new_data = VarianceThreshold(threshold=k).fit_transform(data)
            return var_data, new_data
        else:
            return var_data

    # 线性相关系数衡量
    def pearson_value(self, data, label):
        k = self.pearson_value_k
        label = str(label)
        # k为想删除的feature个数
        Y = data[label]
        x = data[[x for x in data.columns if x != label]]
        res = []
        for i in range(x.shape[1]):
            data_res = np.c_[Y, x.iloc[:, i]].T
            cor_value = np.abs(np.corrcoef(data_res)[0, 1])
            res.append([label, x.columns[i], cor_value])
        res = sorted(np.array(res), key=lambda x: x[2], reverse=True)
        if k is not None:
            if k < len(res):
                new_c = []  # 保留的feature
                for i in range(len(res) - k):
                    new_c.append(res[i][1])
                return res, new_c
            else:
                print('feature个数越界～')
        else:
            return res

    # 共线性检验
    def vif_test(self, data, label):
        label = str(label)
        # k为想删除的feature个数
        x = data[[x for x in data.columns if x != label]]
        res = np.abs(np.corrcoef(x.T))
        vif_value = []
        for i in range(res.shape[0]):
            for j in range(res.shape[0]):
                if j > i:
                    vif_value.append([x.columns[i], x.columns[j], res[i, j]])
        vif_value = sorted(vif_value, key=lambda x: x[2])
        if k is not None:
            if k < len(vif_value):
                new_c = []  # 保留的feature
                for i in range(len(x)):
                    if vif_value[-i][1] not in new_c:
                        new_c.append(vif_value[-i][1])
                    else:
                        new_c.append(vif_value[-i][0])
                    if len(new_c) == k:
                        break
                out = [x for x in x.columns if x not in new_c]
                return vif_value, out
            else:
                print('feature个数越界～')
        else:
            return vif_value

    # Mutual Information
    def MI(self, X, Y):
        # len(X) should be equal to len(Y)
        # X,Y should be the class feature
        total = len(X)
        X_set = set(X)
        Y_set = set(Y)
        if len(X_set) > 10:
            print('%s非分类变量，请检查后再输入' % X_set)
            sys.exit()
        elif len(Y_set) > 10:
            print('%s非分类变量，请检查后再输入' % Y_set)
            sys.exit()
        # Mutual information
        MI = 0
        eps = 1.4e-45
        for i in X_set:
            for j in Y_set:
                indexi = np.where(X == i)
                indexj = np.where(Y == j)
                ijinter = np.intersect1d(indexi, indexj)
                px = 1.0 * len(indexi[0]) / total
                py = 1.0 * len(indexj[0]) / total
                pxy = 1.0 * len(ijinter) / total
                MI = MI + pxy * np.log2(pxy / (px * py) + eps)
        return MI

    def mic_entroy(self, data, label):
        label = str(label)
        # k为想删除的feature个数
        x = data[[x for x in data.columns if x != label]]
        Y = data[label]
        mic_value = []
        for i in range(x.shape[1]):
            if len(set(x.iloc[:, i])) <= 10:
                res = self.MI(Y, x.iloc[:, i])
                mic_value.append([x.columns[i], res])
        mic_value = sorted(mic_value, key=lambda x: x[1])
        return mic_value

    # 递归特征消除法
    def wrapper_way(self, data, label):
        k = self.wrapper_k
        # k 为要保留的数据feature个数
        label = str(label)
        label_data = data[label]
        col = [x for x in data.columns if x != label]
        train_data = data[col]
        res = pd.DataFrame(
            RFE(estimator=LogisticRegression(), n_features_to_select=k).fit_transform(train_data, label_data))
        res_c = []
        for i in range(res.shape[1]):
            for j in range(data.shape[1]):
                if (res.iloc[:, i] - data.iloc[:, j]).sum() == 0:
                    res_c.append(data.columns[j])
        res.columns = res_c
        return res

    # l1/l2正则方法
    def embedded_way(self, data, label):
        way = self.way
        C_0 = self.C_0
        label = str(label)
        label_data = data[label]
        col = [x for x in data.columns if x != label]
        train_data = data[col]
        res = pd.DataFrame(
            SelectFromModel(LogisticRegression(penalty=way, C=C_0)).fit_transform(train_data, label_data))
        res_c = []
        for i in range(res.shape[1]):
            for j in range(data.shape[1]):
                if (res.iloc[:, i] - data.iloc[:, j]).sum() == 0:
                    res_c.append(data.columns[j])
        res.columns = res_c
        return res

    # 基于树模型特征选择
    def tree_way(self, data, label):
        label = str(label)
        label_data = data[label]
        col = [x for x in data.columns if x != label]
        train_data = data[col]
        res = pd.DataFrame(SelectFromModel(GradientBoostingClassifier()).fit_transform(train_data, label_data))
        res_c = []
        for i in range(res.shape[1]):
            for j in range(data.shape[1]):
                if (res.iloc[:, i] - data.iloc[:, j]).sum() == 0:
                    res_c.append(data.columns[j])
        res.columns = res_c
        return res

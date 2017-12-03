from sklearn.feature_selection import VarianceThreshold
import numpy as np
import pandas as pd
import sys
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import GradientBoostingClassifier

# 方差选择法
def var_filter(data, k=None):
    var_data = data.var().sort_values()
    if k is not None:
        new_data = VarianceThreshold(threshold=k).fit_transform(data)
        return var_data, new_data
    else:
        return var_data


# 线性相关系数衡量
def pearson_value(data, label, k=None):
    label = str(label)
    # k为想删除的feature个数
    Y = data[label]
    x = data[[x for x in data.columns if x != label]]
    res = []
    for i in range(x.shape[1]):
        data_res = np.c_[Y, x.iloc[:, i]].T
        cor_value = np.abs(np.corrcoef(data_res)[0, 1])
        res.append([label, x.columns[i], cor_value])
    res = sorted(np.array(res), key=lambda x: x[2])
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
def vif_test(data, label, k=None):
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
def MI(X, Y):
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


def mic_entroy(data, label, k=None):
    # mic_value值越小，两者相关性越弱
    label = str(label)
    # k为想删除的feature个数
    x = data[[x for x in data.columns if x != label]]
    Y = data[label]
    mic_value = []
    for i in range(x.shape[1]):
        if len(set(x.iloc[:, i])) <= 10:
            res = MI(Y, x.iloc[:, i])
            mic_value.append([x.columns[i], res])
        mic_value = sorted(mic_value, key=lambda x: x[1])
    return mic_value


# 递归特征消除法
def wrapper_way(data, label, k=3):
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
def embedded_way(data, label, way='l2', C_0=0.1):
    label = str(label)
    label_data = data[label]
    col = [x for x in data.columns if x != label]
    train_data = data[col]
    res = pd.DataFrame(SelectFromModel(LogisticRegression(penalty=way, C=C_0)).fit_transform(train_data, label_data))
    res_c = []
    for i in range(res.shape[1]):
        for j in range(data.shape[1]):
            if (res.iloc[:, i] - data.iloc[:, j]).sum() == 0:
                res_c.append(data.columns[j])
    res.columns = res_c
    return res


# 基于树模型特征选择
def tree_way(data,label):
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


if __name__ == '__main__':
    data_all = pd.read_table('/Users/slade/Documents/Yoho/personal-code/machine-learning/data/data_all.txt')
    # ...

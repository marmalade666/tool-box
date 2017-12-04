# -*- coding:utf-8 -*-
import pandas as pd
import random as rd
import numpy as np
import math as ma


class sample_s(object):
    def __init__(self):
        ''''this is my pleasure'''

    def group_sample(self, data_set, label, percent=0.1):
        # 分层抽样
        # data_set:数据集
        # label:分层变量
        # percent:抽样占比
        # q:每次抽取是否随机,null为随机
        # 抽样根据目标列分层，自动将样本数较多的样本分层按percent抽样，得到目标列样本较多的特征欠抽样数据
        x = data_set
        y = label
        z = percent
        diff_case = pd.DataFrame(x[y]).drop_duplicates([y])
        result = []
        result = pd.DataFrame(result)
        for i in range(len(diff_case)):
            k = np.array(diff_case)[i]
            data_set = x[x[y] == k[0]]
            nrow_nb = data_set.iloc[:, 0].count()
            data_set.index = range(nrow_nb)
            index_id = rd.sample(range(nrow_nb), int(nrow_nb * z))
            result = pd.concat([result, data_set.iloc[index_id, :]], axis=0)
        new_data = pd.Series(result['label']).value_counts()
        new_data = pd.DataFrame(new_data)
        new_data.columns = ['cnt']
        k1 = pd.DataFrame(new_data.index)
        k2 = new_data['cnt']
        new_data = pd.concat([k1, k2], axis=1)
        new_data.columns = ['id', 'cnt']
        max_cnt = max(new_data['cnt'])
        k3 = new_data[new_data['cnt'] == max_cnt]['id']
        result = result[result[y] == k3[0]]
        return result

    def under_sample(self, data_set, label, percent=0.1, q=1):
        # 欠抽样
        # data_set:数据集
        # label:抽样标签
        # percent:抽样占比
        # q:每次抽取是否随机
        # 抽样根据目标列分层，自动将样本数较多的样本按percent抽样，得到目标列样本较多特征的欠抽样数据
        x = data_set
        y = label
        z = percent
        diff_case = pd.DataFrame(pd.Series(x[y]).value_counts())
        diff_case.columns = ['cnt']
        k1 = pd.DataFrame(diff_case.index)
        k2 = diff_case['cnt']
        diff_case = pd.concat([k1, k2], axis=1)
        diff_case.columns = ['id', 'cnt']
        max_cnt = max(diff_case['cnt'])
        k3 = diff_case[diff_case['cnt'] == max_cnt]['id']
        new_data = x[x[y] == k3[0]].sample(frac=z, random_state=q, axis=0)
        return new_data

    def combine_sample(self, data_set, label, number, percent=0.35, q=1):
        # 组合抽样
        # data_set:数据集
        # label:目标列
        # number:计划抽取多类及少类样本和
        # percent：少类样本占比
        # q:每次抽取是否随机
        # 设定总的期待样本数量，及少类样本占比，采取多类样本欠抽样，少类样本过抽样的组合形式
        x = data_set
        y = label
        n = number
        p = percent
        diff_case = pd.DataFrame(pd.Series(x[y]).value_counts())
        diff_case.columns = ['cnt']
        k1 = pd.DataFrame(diff_case.index)
        k2 = diff_case['cnt']
        diff_case = pd.concat([k1, k2], axis=1)
        diff_case.columns = ['id', 'cnt']
        max_cnt = max(diff_case['cnt'])
        k3 = diff_case[diff_case['cnt'] == max_cnt]['id']
        k4 = diff_case[diff_case['cnt'] != max_cnt]['id']
        n1 = p * n
        n2 = n - n1
        fre1 = n2 / float(x[x[y] == k3[0]]['label'].count())
        fre2 = n1 / float(x[x[y] == k4[1]]['label'].count())
        fre3 = ma.modf(fre2)
        new_data1 = x[x[y] == k3[0]].sample(frac=fre1, random_state=q, axis=0)
        new_data2 = x[x[y] == k4[1]].sample(frac=fre3[0], random_state=q, axis=0)
        test_data = pd.DataFrame([])
        if int(fre3[1]) > 0:
            i = 0
            while i < (int(fre3[1])):
                data = x[x[y] == k4[1]]
                test_data = pd.concat([test_data, data], axis=0)
                i += 1
        result = pd.concat([new_data1, new_data2, test_data], axis=0)
        return result

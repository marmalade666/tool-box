import pandas as pd
import numpy as np
import matplotlib as mat


class Dealing(object):
    def __init__(self):
        ''''this is my pleasure , slade sal!'''

    def Key_Dealing(self, data_input, key_value=0.95):
        data_union = []
        data_union = pd.DataFrame(data_union)
        x = data_input
        y = key_value
        for i in range(len(x.columns)):
            data1 = x.iloc[:, i].dropna(how='any')
            key = data1.quantile(y)
            data2 = x.iloc[:, i]
            data2 = data2.fillna(value=key)
            data2[data2 > key] = key
            data_union = pd.concat([data_union, data2], axis=1)
        return data_union

    def Value_Dealing(self, data_input, Value):
        data_union = []
        data_union = pd.DataFrame(data_union)
        x = data_input
        y = Value
        for i in range(len(x.columns)):
            key = y
            data2 = x.iloc[:, i]
            data2 = data2.fillna(value=key)
            data2[data2 > key] = key
            data_union = pd.concat([data_union, data2], axis=1)
        return data_union

    def Value_Mode(self, data_input, key_value=0.95):
        data_union = []
        data_union = pd.DataFrame(data_union)
        x = data_input
        y = key_value
        for i in range(len(x.columns)):
            data1 = x.iloc[:, i].dropna(how='any')
            key = data1.value_counts().argmax()
            data2 = x.iloc[:, i].dropna(how='any')
            key1 = data2.quantile(y)
            data3 = x.iloc[:, i].dropna(how='any')
            data3[data3 > key1] = key1
            data3 = data3.fillna(value=key)
            data_union = pd.concat([data_union, data3], axis=1)
        return data_union

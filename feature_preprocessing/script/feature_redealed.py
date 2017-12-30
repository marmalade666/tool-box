'''
    __author__ = 'sladesal'
    __time__ = '20171128'
    __bolg__ = 'www.shataowei.com'
'''
from __future__ import division
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import NearestNeighbors

def change_data_format(data):
    # 以下预处理都是基于dataframe格式进行的
    data_new = pd.DataFrame(data)
    return data_new


# 去除空值过多的feature
def nan_remove(data, rate_base=0.4):
    all_cnt = data.shape[0]
    avaiable_index = []
    # 针对每一列feature统计nan的个数，个数大于全量样本的rate_base的认为是异常feature，进行剔除
    for i in range(data.shape[1]):
        rate = np.isnan(np.array(data.iloc[:, i])).sum() / all_cnt
        if rate <= rate_base:
            avaiable_index.append(i)
    data_available = data.iloc[:, avaiable_index]
    return data_available, avaiable_index


# 离群点盖帽
def outlier_remove(data, limit_value=10, method='box', percentile_limit_set=90, changed_feature_box=[]):
    # limit_value是最小处理样本个数set，当独立样本大于limit_value我们认为非可onehot字段
    feature_cnt = data.shape[1]
    feature_change = []
    if method == 'box':
        for i in range(feature_cnt):
            if len(pd.DataFrame(data.iloc[:, i]).drop_duplicates()) >= limit_value:
                q1 = np.percentile(np.array(data.iloc[:, i]), 25)
                q3 = np.percentile(np.array(data.iloc[:, i]), 75)
                # q3+3/2*qi为上截距点，详细百度分箱图
                top = q3 + 1.5 * (q3 - q1)
                data.iloc[:, i][data.iloc[:, i] > top] = top
                feature_change.append(i)
        return data, feature_change
    if method == 'self_def':
        # 快速截断
        if len(changed_feature_box) == 0:
            # 当方法选择为自定义，且没有定义changed_feature_box则全量数据全部按照percentile_limit_set的分位点大小进行截断
            for i in range(feature_cnt):
                if len(pd.DataFrame(data.iloc[:, i]).drop_duplicates()) >= limit_value:
                    q_limit = np.percentile(np.array(data.iloc[:, i]), percentile_limit_set)
                    data.iloc[:, i][data.iloc[:, i] > q_limit] = q_limit
                    feature_change.append(i)
        else:
            # 如果定义了changed_feature_box，则将changed_feature_box里面的按照box方法，changed_feature_box的feature index按照percentile_limit_set的分位点大小进行截断
            for i in range(feature_cnt):
                if len(pd.DataFrame(data.iloc[:, i]).drop_duplicates()) >= limit_value:
                    if i in changed_feature_box:
                        q1 = np.percentile(np.array(data.iloc[:, i]), 25)
                        q3 = np.percentile(np.array(data.iloc[:, i]), 75)
                        # q3+3/2*qi为上截距点，详细百度分箱图
                        top = q3 + 1.5 * (q3 - q1)
                        data.iloc[:, i][data.iloc[:, i] > top] = top
                        feature_change.append(i)
                    else:
                        q_limit = np.percentile(np.array(data.iloc[:, i]), percentile_limit_set)
                        data.iloc[:, i][data.iloc[:, i] > q_limit] = q_limit
                        feature_change.append(i)
            return data, feature_change


# 空feature填充
def nan_fill(data, limit_value=10, countinuous_dealed_method='mean'):
    feature_cnt = data.shape[1]
    normal_index = []
    continuous_feature_index = []
    class_feature_index = []
    continuous_feature_df = pd.DataFrame()
    class_feature_df = pd.DataFrame()
    # 当存在空值且每个feature下独立的样本数小于limit_value，我们认为是class feature采取one_hot_encoding；
    # 当存在空值且每个feature下独立的样本数大于limit_value，我们认为是continuous feature采取mean,min,max方式
    for i in range(feature_cnt):
        if np.isnan(np.array(data.iloc[:, i])).sum() > 0:
            if len(pd.DataFrame(data.iloc[:, i]).drop_duplicates()) >= limit_value:
                if countinuous_dealed_method == 'mean':
                    continuous_feature_df = pd.concat(
                        [continuous_feature_df, data.iloc[:, i].fillna(data.iloc[:, i].mean())], axis=1)
                    continuous_feature_index.append(i)
                elif countinuous_dealed_method == 'max':
                    continuous_feature_df = pd.concat(
                        [continuous_feature_df, data.iloc[:, i].fillna(data.iloc[:, i].max())], axis=1)
                    continuous_feature_index.append(i)
                elif countinuous_dealed_method == 'min':
                    continuous_feature_df = pd.concat(
                        [continuous_feature_df, data.iloc[:, i].fillna(data.iloc[:, i].min())], axis=1)
                    continuous_feature_index.append(i)
            elif len(pd.DataFrame(data.iloc[:, i]).drop_duplicates()) > 0 and len(
                    pd.DataFrame(data.iloc[:, i]).drop_duplicates()) < limit_value:
                class_feature_df = pd.concat(
                    [class_feature_df, pd.get_dummies(data.iloc[:, i], prefix=data.columns[i])], axis=1)
                class_feature_index.append(i)
        else:
            normal_index.append(i)
    data_update = pd.concat([data.iloc[:, normal_index], continuous_feature_df, class_feature_df], axis=1)
    return data_update


# onehotencoding
def ohe(data, limit_value=10):
    feature_cnt = data.shape[1]
    class_index = []
    class_df = pd.DataFrame()
    normal_index = []
    # limit_value以下的均认为是class feature，进行ohe过程
    for i in range(feature_cnt):
        if len(pd.DataFrame(data.iloc[:, i]).drop_duplicates()) < limit_value:
            class_index.append(i)
            class_df = pd.concat([class_df, pd.get_dummies(data.iloc[:, i], prefix=data.columns[i])], axis=1)
        else:
            normal_index.append(i)
    data_update = pd.concat([data.iloc[:, normal_index], class_df], axis=1)
    return data_update


# smote unbalance dataset
def smote(data, tag_index=None, max_amount=0, std_rate=5, kneighbor=5, kdistinctvalue=10, method='mean'):
    try:
        data = pd.DataFrame(data)
    except:
        raise ValueError
    case_state = data.iloc[:, tag_index].groupby(data.iloc[:, tag_index]).count()
    case_rate = max(case_state) / min(case_state)
    location = []
    if case_rate < 5:
        print('不需要smote过程')
        return data
    else:
        # 拆分不同大小的数据集合
        less_data = np.array(
            data[data.iloc[:, tag_index] == np.array(case_state[case_state == min(case_state)].index)[0]])
        more_data = np.array(
            data[data.iloc[:, tag_index] == np.array(case_state[case_state == max(case_state)].index)[0]])
        # 找出每个少量数据中每条数据k个邻居
        neighbors = NearestNeighbors(n_neighbors=kneighbor).fit(less_data)
        for i in range(len(less_data)):
            point = less_data[i, :]
            location_set = neighbors.kneighbors([less_data[i]], return_distance=False)[0]
            location.append(location_set)
        # 确定需要将少量数据补充到上限额度
        # 判断有没有设定生成数据个数，如果没有按照std_rate(预期正负样本比)比例生成
        if max_amount > 0:
            amount = max_amount
        else:
            amount = int(max(case_state) / std_rate)
        # 初始化，判断连续还是分类变量采取不同的生成逻辑
        times = 0
        continue_index = []  # 连续变量
        class_index = []  # 分类变量
        for i in range(less_data.shape[1]):
            if len(pd.DataFrame(less_data[:, i]).drop_duplicates()) > kdistinctvalue:
                continue_index.append(i)
            else:
                class_index.append(i)
        case_update = list()
        location_transform = np.array(location)
        while times < amount:
            # 连续变量取附近k个点的重心，认为少数样本的附近也是少数样本
            new_case = []
            pool = np.random.permutation(len(location))[1]
            neighbor_group = location_transform[pool]
            if method == 'mean':
                new_case1 = less_data[list(neighbor_group), :][:, continue_index].mean(axis=0)
            # 连续样本的附近点向量上的点也是异常点
            if method == 'random':
                away_index = np.random.permutation(len(neighbor_group) - 1)[1]
                neighbor_group_removeorigin = neighbor_group[1:][away_index]
                new_case1 = less_data[pool][continue_index] + np.random.rand() * (
                    less_data[pool][continue_index] - less_data[neighbor_group_removeorigin][continue_index])
            # 分类变量取mode
            new_case2 = np.array(pd.DataFrame(less_data[neighbor_group, :][:, class_index]).mode().iloc[0, :])
            new_case = list(new_case1) + list(new_case2)
            if times == 0:
                case_update = new_case
            else:
                case_update = np.c_[case_update, new_case]
            print('已经生成了%s条新数据，完成百分之%.2f' % (times, times * 100 / amount))
            times = times + 1
        less_origin_data = np.hstack((less_data[:, continue_index], less_data[:, class_index]))
        more_origin_data = np.hstack((more_data[:, continue_index], more_data[:, class_index]))
        data_res = np.vstack((more_origin_data, less_origin_data, np.array(case_update.T)))
        label_columns = [0] * more_origin_data.shape[0] + [1] * (
        less_origin_data.shape[0] + np.array(case_update.T).shape[0])
        data_res = pd.DataFrame(data_res)
    return data_res


# 数据分列
def reload(data):
    feature = pd.concat([data.iloc[:, :2], data.iloc[:, 4:]], axis=1)
    tag = data.iloc[:, 3]
    return feature, tag


# 数据切割
def split_data(feature, tag):
    X_train, X_test, y_train, y_test = train_test_split(feature, tag, test_size=0.33, random_state=42)
    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    data_all = pd.read_table('/Users/slade/Documents/Yoho/personal-code/machine-learning/data/data_all.txt')
    print('数据读取完成！')
    # 更改数据格式
    data_all = change_data_format(data_all)
    # 删除电话号码列
    data_all = data_all.iloc[:, 1:]
    data_all, data_avaiable_index = nan_remove(data_all)
    print('空值列处理完毕！')
    data_all, _ = outlier_remove(data_all)
    print('异常点处理完成！')
    data_all = nan_fill(data_all)
    print('空值填充完成！')
    data_all = ohe(data_all)
    print('onehotencoding 完成！')
    data_all = smote(data_all,tag_index=1)
    print('smote过程完成！')
    feature, tag = reload(data_all)
    X_train, X_test, y_train, y_test = split_data(feature, tag)
    print('数据预处理完成！')

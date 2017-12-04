import pandas as pd
import sys
import numpy as np

sys.path.append('/Users/slade/Documents/GitHub/python_tools/feature_ preprocessing/script')
from feature_filter import feature_filter

# load_data
path = '/Users/slade/Documents/GitHub/python_tools/feature_ preprocessing/data/feature_data.txt'
train_data = pd.read_table(path)
# 把null的数据去掉
index_col = [0]
index_col.extend(list(range(6, train_data.shape[1])))
train_data = train_data.iloc[:, index_col]

# 方差,干掉方差小于100的列
model = feature_filter(k_var=100)
model.var_filter(train_data)
# output1：每个方差的个数
# Out[5]:
# status                    2.167657e-01
# clct_qty_all              6.322301e-01
# rencetly_tot_qty_unpay    1.428411e+00
# sum_time                  1.486031e+00
# rencetly_clct_qty         2.576003e+00
# tot_qty_unpay_all         6.719284e+00
# rencetly_cart2_qty        6.810065e+00
# rencetly_cart1_qty        7.749835e+00
# rencetly_pv_qty           8.624140e+00
# pv_qty_search_all         9.851803e+00
# rencetly_pv_qty_search    1.053492e+01
# cart2_qty_all             2.145267e+01
# cart1_qty_all             9.725881e+01
# pv_qty_all                5.048676e+02
# between_time              5.329928e+03
# sum_pay                   1.281202e+10
# max_pay                   2.897641e+10

# output2:选出方差大于k_var的列
# Out[5]:
# dtype: float64, array([[  5.38622000e+03,   0.00000000e+00,   1.59000000e+02,
#            2.20000000e+01],
#         [  6.67600000e+02,   0.00000000e+00,   1.11000000e+02,
#            1.60000000e+01],
#         [  2.00000000e-02,   0.00000000e+00,   3.66000000e+02,
#            3.00000000e+00],
#         ...,
#         [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
#            0.00000000e+00],
#         [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
#            0.00000000e+00],
#         [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
#            0.00000000e+00]]))


# 线性相关系数衡量，pearson_value_k：你想删除掉的feature个数
model = feature_filter(pearson_value_k=3)
model.pearson_value(train_data, 'label')
# train_data为全量数据，'label'为其中因变量的列名
# output1:因变量和每个自变量直接的相关性降序
# Out[5]:
# [array(['label', 'rencetly_cart2_qty', '0.232779867368'],
#         dtype='<U22'),
#   array(['label', 'rencetly_tot_qty_unpay', '0.132289957283'],
#         dtype='<U22'),
#   array(['label', 'cart2_qty_all', '0.109110557883'],
#         dtype='<U22'),
#   array(['label', 'pv_qty_all', '0.102172469365'],
#         dtype='<U22'),
#   array(['label', 'rencetly_clct_qty', '0.0855905395068'],
#         dtype='<U22'),
#   array(['label', 'rencetly_pv_qty_search', '0.0778524743801'],
#         dtype='<U22'),
#   array(['label', 'rencetly_cart1_qty', '0.0745756346545'],
#         dtype='<U22'),
#   array(['label', 'clct_qty_all', '0.0667812613778'],
#         dtype='<U22'),
#   array(['label', 'pv_qty_search_all', '0.0528009483462'],
#         dtype='<U22'),
#   array(['label', 'rencetly_pv_qty', '0.0494240337632'],
#         dtype='<U22'),
#   array(['label', 'cart1_qty_all', '0.044781538479'],
#         dtype='<U22'),
#   array(['label', 'status', '0.0319021916765'],
#         dtype='<U22'),
#   array(['label', 'sum_time', '0.0208099396432'],
#         dtype='<U22'),
#   array(['label', 'tot_qty_unpay_all', '0.0145425871927'],
#         dtype='<U22'),
#   array(['label', 'between_time', '0.00200541775233'],
#         dtype='<U22'),
#   array(['label', 'max_pay', '0.000903654641191'],
#         dtype='<U22'),
#   array(['label', 'sum_pay', '0.000209367052814'],
#         dtype='<U22')]

# output2:保留的feature
# Out[5]:
# ['rencetly_cart2_qty',
#  'rencetly_tot_qty_unpay',
#  'cart2_qty_all',
#  'pv_qty_all',
#  'rencetly_clct_qty',
#  'rencetly_pv_qty_search',
#  'rencetly_cart1_qty',
#  'clct_qty_all',
#  'pv_qty_search_all',
#  'rencetly_pv_qty',
#  'cart1_qty_all',
#  'status',
#  'sum_time',
#  'tot_qty_unpay_all']

# 共线性检验，vif_k想要剔除的feature个数
model = feature_filter(vif_k=3)
model.vif_test(train_data, 'label')

# output1:
# Out[5]:任意两个自变量自己的相关程度
# ([['max_pay', 'cart2_qty_all', 2.374688253745391e-05],
#   ['sum_pay', 'pv_qty_search_all', 6.8059403959146441e-05],
#   ['sum_pay', 'rencetly_tot_qty_unpay', 0.00026664809716079403],
#   ['max_pay', 'cart1_qty_all', 0.00041717936120951634],
#   ['sum_pay', 'rencetly_cart1_qty', 0.00042607850129303847],
#   ['sum_pay', 'cart1_qty_all', 0.00070337806659011347],
#   ['sum_pay', 'rencetly_pv_qty_search', 0.00072269860524510796],
#   ['sum_pay', 'rencetly_clct_qty', 0.00074971849101914691],
#   ['max_pay', 'between_time', 0.00078863133474504042],
#     ...])

# output2：保留的feature
# Out[5]:
# ['status',
#   'max_pay',
#   'sum_time',
#   'between_time',
#   'pv_qty_all',
#   'clct_qty_all',
#   'cart1_qty_all',
#   'tot_qty_unpay_all',
#   'rencetly_pv_qty',
#   'rencetly_pv_qty_search',
#   'rencetly_clct_qty',
#   'rencetly_cart2_qty',
#   'rencetly_cart1_qty',
#   'rencetly_tot_qty_unpay'])

# Mutual Information检验
model = feature_filter()
model.mic_entroy(train_data, 'label')
# output:这边只对满足要求的分类变量进行计算，连续变量跳过
# [['status', 0.00072322360641714854]]

# 递归特征消除法
model = feature_filter(wrapper_k=3)
model.wrapper_way(train_data, 'label')
# output:wrapper_k为要保留的feature个数，输出结果为保留下来的数据
# Out[5]:
#         rencetly_clct_qty  rencetly_cart2_qty  rencetly_tot_qty_unpay
# 0                     7.0                 7.0                     7.0
# 1                     7.0                 7.0                     7.0
# 2                     7.0                 7.0                     7.0
# 3                     7.0                 7.0                     7.0
# 4                     7.0                 7.0                     7.0
# 5                     7.0                 7.0                     7.0

# 正则方法
model = feature_filter()
model.embedded_way(train_data, 'label')
# output:正则化后保留下来的数据
#         rencetly_pv_qty  rencetly_pv_qty_search  rencetly_clct_qty  \
# 0                   1.0                     7.0                7.0
# 1                   1.0                     1.0                7.0

# 基于树模型特征选择
model = feature_filter()
model.tree_way(train_data, 'label')
# output:树模型选择后保留下来的数据
# Out[5]:
#          max_pay  between_time  pv_qty_all  cart2_qty_all  cart1_qty_all  \
# 0        5386.22         159.0        22.0            0.0           24.0
# 1         667.60         111.0        16.0            0.0            7.0
# 2           0.02         366.0         3.0            0.0            0.0
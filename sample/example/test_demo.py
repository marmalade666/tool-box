import pandas as pd
import numpy as np
import sys

sys.path.append('/Users/slade/Documents/GitHub/python_tools/sample')
from sample import sample_s

sps = sample_s()
path = '/Users/slade/Documents/GitHub/python_tools/sample/data/sample_demo_data.txt'
data = pd.read_table(path)

group_sample_data = sps.group_sample(data,'label')
#产出结果如下：
# Out[7]:
#         label   max_pay   sum_pay  sum_time  between_time  pv_qty_all  \
# 20127       0   6090.00  11739.00         2             7           1
# 50125       0      0.00      0.00         0             0           0
# 50137       0   6247.00   9353.40         6             2           0
# 110116      0      0.00      0.00         0             0           1
# 64542       0      0.00      0.00         0             0           5

under_sample_data = sps.under_sample(data,'label')
#产出结果如下：
# Out[9]:
#         label  max_pay  sum_pay  sum_time  between_time  pv_qty_all  \
# 89033       0      0.0      0.0         0             0           2
# 51955       0      0.0      0.0         0             0           0
# 31767       0   3499.0      0.0         0           160           1
# 19339       0    234.0    234.0         1             1          17
# 127765      0   6999.0   6999.0         1            25          76

combine_sample_data = sps.combine_sample(data,'label',number=100, percent=0.35, q=1)
#number为组合抽样的正负样本和
#产出结果如下：
# Out[16]:
#         label  max_pay  sum_pay  sum_time  between_time  pv_qty_all  \
# 89033       0      0.0      0.0         0             0           2
# 51955       0      0.0      0.0         0             0           0
# 31767       0   3499.0      0.0         0           160           1
# 19339       0    234.0    234.0         1             1          17
# 127765      0   6999.0   6999.0         1            25          76
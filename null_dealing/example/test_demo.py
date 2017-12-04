import pandas as pd
import numpy as np
import sys

sys.path.append('/Users/slade/Documents/GitHub/python_tools/null_dealing/script')
from null_dealing import Dealing

path = '/Users/slade/Documents/GitHub/python_tools/null_dealing/data/null_data_demo.txt'
data = pd.read_table(path)

model = Dealing()

#以0.95分位数填充
model.Key_Dealing(data.iloc[:,:2])
#output:
# Out[8]:
#         label  diff_rgst
# 0           0     2037.0
# 1           0     2037.0
# 2           0     2037.0
# 3           1     2037.0

#以固定值key_value填充
model.Value_Dealing(data.iloc[:,:2],Value=10)
#output:
# Out[9]:
#         label  diff_rgst
# 0           0       10.0
# 1           0       10.0
# 2           0       10.0
# 3           1       10.0

##以众数填充
model.Value_Mode(data.iloc[:,:2])
#output:
# Out[2]:
#         label  diff_rgst
# 0           0     2037.0
# 1           0     2037.0
# 2           0     2037.0
# 3           1     2037.0
# 4           0     2037.0
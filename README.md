### python_tools
有一些python常用的小工具，实现后存放在这边

**其中feature_preprocessing、null_dealing、sample已经封装打包到pypi上了欢迎使用:[data_preprocessing](https://pypi.python.org/pypi?:action=display&name=data_preprocessing&version=0.0.2)**

安装：pip install data_preprocessing

使用:
```
from data_preprocessing import data_preprocessing
# sample 模块
sample = data_preprocessing.sample()
sample.方法名 #即可

# null_dealing 模块
null_dealing = data_preprocessing.null_dealing()
null_dealing.方法名 #即可

# feature_filter 模块
feature_filter = data_preprocessing.feature_filter()
feature_filter.方法名 #即可
```

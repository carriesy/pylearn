# -*- coding:utf-8 -*-
import sys

reload(sys)
sys.setdefaultencoding('utf-8')
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
import matplotlib as mpl
import matplotlib.pyplot as plt

if __name__ == '__main__':
    pd.set_option('display.width', 300)
    path = 'data/car.data'
    data = pd.read_csv(path, header=None)
    n_columns = len(data.columns)
    columns = ['buy', 'maintain', 'doors', 'persons', 'boot', 'safety', 'accept']
    new_columns = dict(zip(np.arange(n_columns), columns))
    data.rename(columns=new_columns, inplace=True)
    print data.head()
    # one_hot编码
    x = pd.DataFrame()
    for col in columns[:-1]:
        t = pd.get_dummies(data[col])
        t = t.rename(columns=lambda x: col + '_' + str(x))
        x = pd.concat((x, t), axis=1)
    print u'输出x前五行', x.head()
    print x.columns

    y = np.array(pd.Categorical(data['accept']).codes)

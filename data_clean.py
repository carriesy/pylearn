# coding: utf-8

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

if __name__=='__main__':
    #read data
    pd.set_option('display.width',300)
    data=pd.read_csv('data/tel.csv',skipinitialspace=True,thousands=',')
    print u'原始数据:\n',data.head(10)

    #字符串映射为数值
    le=LabelEncoder()
    for col in data.columns:
        data[col]=le.fit_transform(data[col])

    #年龄分组
    bins=[-1,6,12,18,24,35,50,70]
    pd.cut(data['age'],bins=bins,labels=np.arange(len(bins)-1))

    #取对数,归一化
    columns_log = ['income', 'tollten', 'longmon', 'tollmon', 'equipmon', 'cardmon',
                   'wiremon', 'longten', 'tollten', 'equipten', 'cardten', 'wireten', ]
    mms=MinMaxScaler()
    for col in columns_log:
        data[col]=np.log(data[col]-data[col].min()+1)
    #one_hot编码
    columns_one_hot = ['region', 'age', 'address', 'ed', 'reside', 'custcat']
    for col in columns_one_hot:
        data=data.join(pd.get_dummies(data[col],prefix= col))

    data.drop(columns_one_hot, axis=1,inplace=True)

    columns=list(data.columns)
    columns.remove('churn')
    x=data[columns]
    y=data['churn']

    print u'清洗后的数据\n',x.head(10)
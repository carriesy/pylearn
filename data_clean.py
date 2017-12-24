# coding: utf-8

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

if __name__ == '__main__':
    # read data
    pd.set_option('display.width', 300)
    data = pd.read_csv('data/tel.csv', skipinitialspace=True, thousands=',')
    print u'原始数据:\n', data.head(10)

    # 字符串映射为数值
    le = LabelEncoder()
    for col in data.columns:
        data[col] = le.fit_transform(data[col])

    # 年龄分组
    bins = [-1, 6, 12, 18, 24, 35, 50, 70]
    pd.cut(data['age'], bins=bins, labels=np.arange(len(bins) - 1))

    # 取对数,归一化
    columns_log = ['income', 'tollten', 'longmon', 'tollmon', 'equipmon', 'cardmon',
                   'wiremon', 'longten', 'tollten', 'equipten', 'cardten', 'wireten', ]
    mms = MinMaxScaler()
    for col in columns_log:
        data[col] = np.log(data[col] - data[col].min() + 1)
    # one_hot编码
    columns_one_hot = ['region', 'age', 'address', 'ed', 'reside', 'custcat']
    for col in columns_one_hot:
        data = data.join(pd.get_dummies(data[col], prefix=col))

    data.drop(columns_one_hot, axis=1, inplace=True)

    columns = list(data.columns)
    columns.remove('churn')
    x = data[columns]
    y = data['churn']

    print u'清洗后的数据\n', x.head(10)

    # 特征选择
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, random_state=0)
    clf = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=12, min_samples_split=5, oob_score=True
                                 , class_weight={0: 1, 1: 1 / y_train.mean()})
    clf.fit(x_train, y_train)
    importances_fearture = pd.DataFrame(data={'fearture': x.columns,
                                              'importance': clf.feature_importances_})
    importances_fearture.sort_values(by='importance', axis=0, ascending=False, inplace=True)
    importances_fearture['importance_cum'] = importances_fearture['importance'].cumsum()
    print u'特征重要度\n', importances_fearture
    select_fearture = importances_fearture.loc[importances_fearture['importance_cum'] < 0.95, 'fearture']

    # 重新组织数据
    x_train = x_train[select_fearture]
    x_test = x_test[select_fearture]

    # 训练模型
    clf.fit(x_train, y_train)
    y_hat = clf.predict(x_train)
    # 评价模型
    print u'oob', clf.oob_score_
    print u'训练及准确率', accuracy_score(y_train, y_hat)
    print u'训练集召回率', recall_score(y_train, y_hat)
    print u'训练集F1', f1_score(y_train, y_hat)

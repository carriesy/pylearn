# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, SelectPercentile, chi2
from sklearn.linear_model import LogisticRegressionCV
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.manifold import TSNE
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

if __name__ == '__main__':
    # 读取数据
    stype = 'pca'
    pd.set_option('display.width', 200)
    data = pd.read_csv('data/iris.data', header=None)
    columns = np.array([u'花萼长度', u'花萼宽度', u'花瓣长度', u'花瓣宽度', u'类型'])
    data.rename(columns=dict(zip(np.arange(5), columns)), inplace=True)
    data[u'类型'] = pd.Categorical(data[u'类型']).codes
    x = data[columns[:-1]]
    y = data[columns[-1]]
    print data.head(5)

    # pca降维  #卡方检验选择特征
    if stype == 'pca':
        pca = PCA(n_components=2, whiten=True, random_state=0)
        x = pca.fit_transform(x)
        print u'各方向方差', pca.explained_variance_
        print u'方差所占比例', pca.explained_variance_ratio_
    else:

        fs = SelectKBest(chi2, k=2)
        fs.fit(x, y)
        idx = fs.get_support(indices=True)
        print 'get_support=', idx

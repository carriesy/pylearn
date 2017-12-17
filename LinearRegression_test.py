# /usr/bin/python
# -*- coding:utf-8 -*-

import sys

reload(sys)
sys.setdefaultencoding('utf-8')
from matplotlib.font_manager import FontManager, FontProperties
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import pprint
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


def getChineseFont():
    return FontProperties(fname='/System/Library/Fonts/PingFang.ttc')


if __name__ == "__main__":
    path = 'data/Advertising.csv'

    data = pd.read_csv(path)
    # print data
    pprint.pprint(data)
    x = data[['TV', 'Radio', 'Newspaper']]
    y = data[['Sales']]
    print x

    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1)

    model = LinearRegression()
    model.fit(x_train, y_train)
    y_hat = model.predict(x_test)
    mse = np.average((y_hat - np.array(y_test)) ** 2)
    print 'mse is'
    print mse

    print 'score is'
    print model.score(x_test, y_test)

    mpl.rcParams['font.sans-serif'] = [u'simHei']
    mpl.rcParams['axes.unicode_minus'] = False

    plt.figure(facecolor='w', figsize=(9, 10))
    plt.plot(data['TV'], y, 'ro', label='TV')
    plt.plot(data['Radio'], y, 'g^', label='Radio')
    plt.plot(data['Newspaper'], y, 'mv', label='Newspaper')
    plt.legend(loc='lower right')
    plt.xlabel(u'广告花费', fontproperties=getChineseFont(), fontsize=16)
    plt.ylabel(u'销售额', fontproperties=getChineseFont(), fontsize=16)
    plt.title(u'销售额与广告投入对比数据', fontproperties=getChineseFont(), fontsize=20)
    plt.grid()
    plt.show()

    plt.figure(facecolor='w')
    plt.subplot(311)
    plt.plot(data['TV'], y, 'ro')
    plt.title('TV')
    plt.grid()
    plt.subplot(312)
    plt.plot(data['Radio'], y, 'b*')
    plt.title('Radio')
    plt.grid()
    plt.subplot(313)
    plt.plot(data['Newspaper'], y, 'g^')
    plt.title('Nespaper')
    plt.grid()
    fontproperties = getChineseFont()
    plt.show()

    plt.figure(facecolor='w')
    t = np.arange(len(x_test))
    plt.plot(t, y_hat, 'r-',label='hat')
    plt.plot(t, y_test, 'g-',label='true')
    plt.legend(loc='upper right')
    plt.title(u'线性回归预测销量', fontproperties=getChineseFont(), fontsize=20)
    plt.show()

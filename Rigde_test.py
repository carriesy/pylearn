#!/usr/bin/python
# -*- coding:utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf-8')#解决macos系统的字符串问题

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import GridSearchCV

if __name__ == '__main__':
    path = 'data/Advertising.csv'
    data = pd.read_csv(path)
    x = data[['TV', 'Radio', 'Newspaper']]
    y = data[['Sales']]
    print x

    # model training
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1)
    model = Ridge()
    # model=Lasso()
    alpha_can = np.logspace(-3, 2, 10)
    np.set_printoptions(suppress=True)
    print 'alpha_can = ', alpha_can
    lasso_model = GridSearchCV(model, param_grid={'alpha': alpha_can}, cv=5)
    lasso_model.fit(x_train, y_train)
    print '超参数：\n', lasso_model.best_params_  # 描述了已取得最佳结果的参数的组合

    #anasis
    y_hat = lasso_model.predict(x_test)
    MSE = np.average((y_hat - np.array(y_test)) ** 2)
    RMSE = np.sqrt(MSE)
    print 'mse', MSE
    print'rmse', RMSE

    # picture
    t = np.arange(len(x_test))
    plt.figure(facecolor='w')
    plt.plot(t, y_hat, label='predict')
    plt.plot(t, y_test, label='true')
    plt.legend(loc='best')
    plt.title('lr_model_predict')
    plt.grid()
    plt.show()

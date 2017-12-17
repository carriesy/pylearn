import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import pprint
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

if __name__ == "__main__":
    path = 'Advertising.csv'

    data = pd.read_csv(path)
    # print data
    pprint.pprint(data)
    x = data[['TV','Radio','Newspaper']]
    y = data[['Sales']]
    print x

    x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.8,random_state=1)

    model=LinearRegression()
    model.fit(x_train,y_train)
    y_hat=model.predict(x_test)
    mse= np.average((y_hat-np.array(y_test))**2)
    print 'mse is'
    print mse

    print 'score is'
    print model.score(x_test,y_test)

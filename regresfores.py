# -*- coding: utf-8 -*-
"""
Created on Sun Jul  8 11:21:43 2018

@author: therock
"""
#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as mp

#get the datatset
dataset = pd.read_csv('Position_Salaries.csv')

#form the dependent and independent variables
X=dataset.iloc[:,[1]].values
Y=dataset.iloc[:,[2]].values

from sklearn.ensemble import  RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=340,random_state=0)
regressor.fit(X,Y)
ypred=regressor.predict(6.5)

xgrid=np.arange(min(X),max(X),0.01)
xgrid=xgrid.reshape((len(xgrid),1))
mp.scatter(X,Y,color='red')
mp.plot(xgrid,regressor.predict(xgrid),color='blue')
mp.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

train = pd.read_csv('train_vehicles.csv', nrows=8000)
#test = pd.read_csv('test_vehicles.csv', nrows=2000)

trainX = train.drop(['price'], axis=1)
trainY = train['price']

print("Number of features: " +str(len(train.columns)))
print("Number of rows: "+str(len(train)))
print("\n")  

from sklearn.metrics import mean_squared_error

def crossVal(model, X, Y, cv=10):
    Xtemp = X
    Ytemp = Y
    dataX = []
    dataY = []
    fittedList = []
    mseList = []
    
    #split into CV mini datsets
    for i in range(0, cv):
        Xsam = Xtemp.sample(int(len(Xtemp)/(cv-i)))
        Ysam = Ytemp.loc[Xsam.index]
        dataX.append(Xsam)
        dataY.append(Ysam)
        Xtemp = Xtemp.drop(Xsam.index)
        Ytemp = Ytemp.drop(Ysam.index)
    
    #cross val with CV models
    for i in range(0, cv):
        trainX = pd.concat(dataX[:i] + dataX[i+1:], axis=0)
        trainY = pd.concat(dataY[:i] + dataY[i+1:], axis=0)
        testX = dataX[i]
        testY = dataY[i]
        fitModel = model.fit(trainX, trainY)
        fittedList.append(fitModel)
        testPredict = fitModel.predict(testX)
        mse = mean_squared_error(testY, testPredict)
        mseList.append(mse)

    return fittedList, mseList


modelsList, mseList = crossVal(RandomForestRegressor(), trainX, trainY, cv=10)

mean = np.asarray(mseList).mean()
std = np.asarray(mseList).std()
var = np.asarray(mseList).var()
print(mean)
print(std)
print(var)
print("Default Parameters:")

from sklearn.model_selection import RandomizedSearchCV

#setup grid of parameters to change

# Sample selection method
bootstrap = [True, False]

#Criterion (mean squared error or mean absolute error)
criterion = ['mse', 'mae']

#Max levels in tree
max_depth = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None]

# Number of features to consider for best split
max_features = ['auto', 'sqrt', 'log2']

# Minimum number of samples to split
min_samples_split = [2, 5, 10]

# Minimum number of samples at node
min_samples_leaf = [1, 2, 4]

# Number of trees
n_estimators = [100, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap,
               'criterion': criterion}

rfr = RandomForestRegressor()

rfr_random = RandomizedSearchCV(estimator = rfr, param_distributions = random_grid, n_iter = 100, cv = 10, verbose=2, n_jobs = -1)

rfr_random.fit(trainX, trainY)

import pickle

pickle.dump(rfr_random, open( "rfr_random", "wb" ) )
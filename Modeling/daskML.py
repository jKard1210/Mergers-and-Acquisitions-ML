import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


import dask.dataframe as dd
from dask.distributed import Client

def runRF():
    client = Client()
    from dask_ml.datasets import make_classification

    df = dd.read_csv("train5.csv",assume_missing=True,sample=640000000,blocksize="10MB")
    df = df.fillna(0).fillna(0)
    for column in df.columns:
        if '.' in column:
            df = df.drop(column,axis=1)
    y_train = df['acquiredNew']
    print(y_train)
    X_train = df.drop('acquiredNew',axis=1)
    X_train = X_train.drop('SIC',axis=1)
    X_train = X_train.drop('CIK',axis=1)
    df2 = dd.read_csv("test5.csv",assume_missing=True,sample=640000000,blocksize="10MB")
    df2 = df2.fillna(0).fillna(0)
    for column in df2.columns:
        if '.' in column:
            df2 = df2.drop(column,axis=1)

    y_test = df2['acquiredNew']
    X_test = df2.drop('acquiredNew',axis=1)
    X_test = X_test.drop('SIC',axis=1)
    X_test = X_test.drop('CIK',axis=1)
    x_test_tickers = X_test['ticker'].values.compute()
    x_test_dates = X_test['date'].values.compute()
    print(x_test_tickers[0])

    np.savetxt("x_test_tickers.csv",x_test_tickers,delimiter=",",fmt='%s')
    np.savetxt("x_test_dates.csv",x_test_dates,delimiter=",",fmt='%s')
    print("GOOD")

    for column in X_train.columns:
        if 'ticker' in column or 'date' in column:
            X_train = X_train.drop(column,axis=1)
            X_test = X_test.drop(column,axis=1)

    X_train = X_train.to_dask_array()
    X_test = X_test.values.compute()
    y_train = y_train.to_dask_array()
    y_test = y_test.values.compute()

    np.savetxt("y_test.csv", y_test, delimiter=",")

    from dask_ml.wrappers import Incremental
    from sklearn.linear_model import SGDClassifier
    from sklearn.neural_network import MLPClassifier
    from dask_ml.wrappers import ParallelPostFit
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=100, max_depth=4,random_state=0)
    clf.fit(X_train,y_train)
    np.savetxt("featureimportance.csv",clf.feature_importances_)
    predictions = clf.predict(X_test)
    np.savetxt("predictions.csv",predictions)
    np.savetxt("predictionsProba.csv",clf.predict_proba(X_test))

def runMLP():
    client = Client()
    from dask_ml.datasets import make_classification

    df = dd.read_csv("train5.csv",assume_missing=True,sample=640000000,blocksize="10MB")
    df = df.fillna(0).fillna(0)
    for column in df.columns:
        if '.' in column:
            df = df.drop(column,axis=1)
    y_train = df['acquiredNew']
    print(y_train)
    X_train = df.drop('acquiredNew',axis=1)
    X_train = X_train.drop('SIC',axis=1)
    X_train = X_train.drop('CIK',axis=1)
    df2 = dd.read_csv("test5.csv",assume_missing=True,sample=640000000,blocksize="10MB")
    df2 = df2.fillna(0).fillna(0)
    for column in df2.columns:
        if '.' in column:
            df2 = df2.drop(column,axis=1)

    y_test = df2['acquiredNew']
    X_test = df2.drop('acquiredNew',axis=1)
    X_test = X_test.drop('SIC',axis=1)
    X_test = X_test.drop('CIK',axis=1)
    x_test_tickers = X_test['ticker'].values.compute()
    x_test_dates = X_test['date'].values.compute()
    print(x_test_tickers[0])

    np.savetxt("x_test_tickers.csv",x_test_tickers,delimiter=",",fmt='%s')
    np.savetxt("x_test_dates.csv",x_test_dates,delimiter=",",fmt='%s')
    print("GOOD")

    for column in X_train.columns:
        if 'ticker' in column or 'date' in column:
            X_train = X_train.drop(column,axis=1)
            X_test = X_test.drop(column,axis=1)

    X_train = X_train.to_dask_array()
    X_test = X_test.values.compute()
    y_train = y_train.to_dask_array()
    y_test = y_test.values.compute()

    np.savetxt("y_test.csv", y_test, delimiter=",")
    
    from dask_ml.wrappers import Incremental
    from sklearn.linear_model import SGDClassifier
    from sklearn.neural_network import MLPClassifier
    from dask_ml.wrappers import ParallelPostFit
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier

    est  = MLPClassifier(solver='adam',activation='relu', alpha = 0.0001,random_state=10,max_iter=2000)
    inc = Incremental(est)
    print("WORKING")
    for _ in range(100):
        inc.fit(X_train, y_train, classes=[0,1])
        for j in range(0, 20):
            print("FITTED")
        predictions = inc.predict(X_test)
        print("Number Positive: ", np.sum(predictions))
        print("TOTAL: ", len(predictions))
        np.savetxt("predictions.csv",predictions)
        np.savetxt("predictionsProba.csv",inc.predict_proba(X_test))
        print('Score:', inc.score(X_train, y_train))
        print('Score:', inc.score(X_test, y_test))



if __name__ == '__main__':
    runRF()
    runMLP()

import pandas as pd
import numpy as np
from sklearn import svm as svmr
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from pdb import set_trace
import csv


def normalize(matrix):
    return (matrix - matrix.min(0)) / matrix.ptp(0)

def get_data(filename, x_names, t_names):
    with open(filename) as csvfile:
        df = pd.read_csv(csvfile)
        # print('num samples: {}'.format(df.shape))
    together = df[x_names + t_names]
    together = together.dropna()  # TODO: maybe impute some values later?
    # print('num samples after dropna: {}'.format(together.shape))
    
    xs = together[x_names]
    ts = together[t_names]
    
    xs = np.array(xs)
    ts = np.array(ts)
    
    return normalize(xs), ts
    
def train(clf, X_train, y_train, X_test, y_test):
    clf.fit(X_train, y_train)
    pred_train = clf.predict(X_train)
    L1_tr = np.average(np.abs(np.subtract(pred_train, y_train)))
    L2_tr = np.average(np.square(np.subtract(pred_train, y_train)))
    
    pred_test = clf.predict(X_test)
    L1_ts = np.average(np.abs(np.subtract(pred_test, y_test)))
    L2_ts = np.average(np.square(np.subtract(pred_test, y_test)))
    
    with open('pred.csv', 'a') as csvfile:
        w = csv.writer(csvfile)
        w.writerow(pred_test)
    
    return L1_tr, L2_tr, L1_ts, L2_ts
    
def rf(X_train, y_train, X_test, y_test):
    forest = RandomForestRegressor(n_estimators=50)
    L1_tr, L2_tr, L1_ts, L2_ts = train(forest, X_train, y_train, X_test, y_test)
    print('tr rf:\tL1: {}\tL2: {}'.format(round(L1_tr, 3), round(L2_tr, 3)))
    return L1_ts, L2_ts
    
def gbr(X_train, y_train, X_test, y_test):
    regressor = GradientBoostingRegressor()
    L1_tr, L2_tr, L1_ts, L2_ts = train(regressor, X_train, y_train, X_test, y_test)
    print('tr gbr:\tL1: {}\tL2: {}'.format(round(L1_tr, 3), round(L2_tr, 3)))
    return L1_ts, L2_ts
    
def svm(X_train, y_train, X_test, y_test):
    classifier = svmr.SVR(gamma='scale')
    L1_tr, L2_tr, L1_ts, L2_ts = train(classifier, X_train, y_train, X_test, y_test)
    print('tr svm:\tL1: {}\tL2: {}'.format(round(L1_tr, 3), round(L2_tr, 3)))
    return L1_ts, L2_ts

    
def run(datafile):   
    t_names = ['doc', 'no3', 'tn', 'tp']
    x_names = ['area', 'elev', 'forest', 'wetland', 'urban', 'ag', 'roads', 'pop']  # TODO: removed shrub and precip because if together everything has NA values
    
    xs, ts = get_data(datafile, x_names, t_names)
    
    for i in range(len(t_names)):
        print('Predicting {}'.format(t_names[i]))
        X_train, X_test, y_train, y_test = train_test_split(xs, ts[:,i], test_size=.25)
    
        with open('pred.csv', 'w') as csvfile:
            w = csv.writer(csvfile)
            w.writerow(y_test)
    
        for fun in [rf, gbr, svm]:
            L1, L2 = fun(X_train, y_train, X_test, y_test)
            print('ts {}:\tL1: {}\tL2: {}'.format(str(fun).split()[1], round(L1, 3), round(L2, 3)))
    
if __name__ == '__main__':
    import sys
    filename = sys.argv[1]
    run(filename)
    

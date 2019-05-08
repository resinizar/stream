import pandas as pd
import numpy as np
from sklearn import svm as svmr
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import csv



def train(clf, X_train, y_train, X_test, y_test, get_imp=False):
    clf.fit(X_train, y_train)
    pred_train = clf.predict(X_train)
    L1_tr = np.average(np.abs(np.subtract(pred_train, y_train)))
    L2_tr = np.average(np.square(np.subtract(pred_train, y_train)))
    
    pred_test = clf.predict(X_test)
    L1_ts = np.average(np.abs(np.subtract(pred_test, y_test)))
    L2_ts = np.average(np.square(np.subtract(pred_test, y_test)))
    
    with open('./output/pred.csv', 'a') as csvfile:
        w = csv.writer(csvfile)
        w.writerow(pred_test)

    imp = clf.feature_importances_ if get_imp else None
    
    return L1_tr, L2_tr, L1_ts, L2_ts, imp


def rf(X_train, y_train, X_test, y_test, get_imp=False):
    forest = RandomForestRegressor(n_estimators=50)
    L1_tr, L2_tr, L1_ts, L2_ts, imp = train(forest, X_train, y_train, X_test, y_test, get_imp)
    return L1_tr, L2_tr, L1_ts, L2_ts, imp
    

def gbr(X_train, y_train, X_test, y_test, get_imp=False):
    regressor = GradientBoostingRegressor()
    L1_tr, L2_tr, L1_ts, L2_ts, imp = train(regressor, X_train, y_train, X_test, y_test, get_imp)
    return L1_tr, L2_tr, L1_ts, L2_ts, imp
    

def svm(X_train, y_train, X_test, y_test, get_imp=False):
    classifier = svmr.SVR(gamma='scale')
    L1_tr, L2_tr, L1_ts, L2_ts, imp = train(classifier, X_train, y_train, X_test, y_test, False)
    return L1_tr, L2_tr, L1_ts, L2_ts, imp


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
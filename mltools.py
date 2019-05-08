import pandas as pd
import numpy as np
from sklearn import svm as svmr
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import csv
from pdb import set_trace
import matplotlib.pyplot as plt



def train(clf, X_train, y_train, X_test, y_test, get_imp=False, writer=None):
    clf.fit(X_train, y_train)
    pred_train = clf.predict(X_train)
    L1_tr = np.average(np.abs(np.subtract(pred_train, y_train)))
    L2_tr = np.average(np.square(np.subtract(pred_train, y_train)))
    
    pred_test = clf.predict(X_test)
    L1_ts = np.average(np.abs(np.subtract(pred_test, y_test)))
    L2_ts = np.average(np.square(np.subtract(pred_test, y_test)))
    
    if writer:
        writer.writerow(pred_test)

    imp = clf.feature_importances_ if get_imp else None
    
    return L1_tr, L2_tr, L1_ts, L2_ts, imp


def rf(X_train, y_train, X_test, y_test, get_imp=False, writer=None):
    forest = RandomForestRegressor(n_estimators=5)
    L1_tr, L2_tr, L1_ts, L2_ts, imp = train(forest, X_train, y_train, X_test, y_test, get_imp, writer)
    return L1_tr, L2_tr, L1_ts, L2_ts, imp
    

def gbr(X_train, y_train, X_test, y_test, get_imp=False, writer=None):
    regressor = GradientBoostingRegressor(n_estimators=10)
    L1_tr, L2_tr, L1_ts, L2_ts, imp = train(regressor, X_train, y_train, X_test, y_test, get_imp, writer)
    return L1_tr, L2_tr, L1_ts, L2_ts, imp
    

def svm(X_train, y_train, X_test, y_test, get_imp=False, writer=None):
    classifier = svmr.SVR(gamma='scale')
    L1_tr, L2_tr, L1_ts, L2_ts, imp = train(classifier, X_train, y_train, X_test, y_test, False, writer)
    return L1_tr, L2_tr, L1_ts, L2_ts, imp


def normalize(matrix):
    return (matrix - matrix.min(0)) / matrix.ptp(0)


# def get_data(filename, x_col_names, t_col_names):
#     with open(filename) as csvfile:
#         df = pd.read_csv(csvfile)
#         # print('num samples: {}'.format(df.shape))
#     together = df[x_col_names + t_col_names]
#     together = together.dropna()  # TODO: maybe impute some values later?
#     # print('num samples after dropna: {}'.format(together.shape))
    
#     xs = together[x_col_names]
#     ts = together[t_col_names]
    
#     xs = np.array(xs)
#     ts = np.array(ts)
    
#     return normalize(xs), ts


def get_NLA_data():
    with open('./data/NLA07_mldata.csv', 'r') as data_file:
        df = pd.read_csv(data_file)
        df = df.drop(['sort', 'year', 'siteid', 'lat', 'long'], axis=1)  # drop columns I don't need
        df = df.dropna()  # drop any rows with unknown values

    xs = df.iloc[:, 4:]
    x_names = df.columns.values[4:]
    ts = df.iloc[:, 0:4]
    t_names = df.columns.values[0:4]

    xs = np.array(xs)
    ts = np.array(ts)

    return normalize(xs), ts, x_names, t_names


def get_NRSA_data():
    with open('./data/NRSA08_mldata.csv', 'r') as data_file:
        df = pd.read_csv(data_file)
        df = df.drop(['sort', 'year', 'siteid', 'lat', 'long', 'epareg', 'strorder'], axis=1)  # drop columns I don't need
        df = df.dropna()  # drop any rows with unknown values

    xs = df.iloc[:, 4:]
    x_names = df.columns.values[4:]
    ts = df.iloc[:, 0:4]
    t_names = df.columns.values[0:4]

    xs = np.array(xs)
    ts = np.array(ts)

    return normalize(xs), ts, x_names, t_names

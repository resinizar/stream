import numpy as np
from sklearn import svm as svmr
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor



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
    forest = RandomForestRegressor(n_estimators=10)
    L1_tr, L2_tr, L1_ts, L2_ts, imp = train(forest, X_train, y_train, X_test, y_test, get_imp, writer)
    return L1_tr, L2_tr, L1_ts, L2_ts, imp
    

def gbr(X_train, y_train, X_test, y_test, get_imp=False, writer=None):
    regressor = GradientBoostingRegressor(n_estimators=50)
    L1_tr, L2_tr, L1_ts, L2_ts, imp = train(regressor, X_train, y_train, X_test, y_test, get_imp, writer)
    return L1_tr, L2_tr, L1_ts, L2_ts, imp
    

def svm(X_train, y_train, X_test, y_test, get_imp=False, writer=None):
    classifier = svmr.SVR(gamma='scale')
    L1_tr, L2_tr, L1_ts, L2_ts, _ = train(classifier, X_train, y_train, X_test, y_test, False, writer)
    return L1_tr, L2_tr, L1_ts, L2_ts, None


def rd(X_train, y_train, X_test, y_test, get_imp=False, writer=None):
    mean = np.average(y_train)
    stdev = np.std(y_train)
    pred_train = np.random.normal(mean, stdev, len(y_train))
    pred_test = np.random.normal(mean, stdev, len(y_test))

    if writer:
        writer.writerow(pred_test)

    L1_tr = np.average(np.abs(np.subtract(pred_train, y_train)))
    L2_tr = np.average(np.square(np.subtract(pred_train, y_train)))

    L1_ts = np.average(np.abs(np.subtract(pred_test, y_test)))
    L2_ts = np.average(np.square(np.subtract(pred_test, y_test)))

    return L1_tr, L2_tr, L1_ts, L2_ts, None

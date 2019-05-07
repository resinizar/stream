import csv
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from collections import Counter
from sklearn.model_selection import KFold
import pandas as pd
from pdb import set_trace

def normalize(matrix):
    return (matrix - matrix.min(0)) / matrix.ptp(0)

def get_data(datafile, x_names, t_names):
    with open(datafile) as csvfile:
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

def train(clf, xs_train, y_train, xs_test, y_test):
    clf.fit(xs_train, y_train)
    # pred_train = clf.predict(xs_train)
    # L1_tr = np.average(np.abs(np.subtract(pred_train, y_train)))
    # L2_tr = np.average(np.square(np.subtract(pred_train, y_train)))
    
    pred_test = clf.predict(xs_test)
    L1_ts = np.average(np.abs(np.subtract(pred_test, y_test)))
    L2_ts = np.average(np.square(np.subtract(pred_test, y_test)))
    
    # with open('pred.csv', 'a') as csvfile:
    #     w = csv.writer(csvfile)
    #     w.writerow(pred_test)
    
    imp = clf.feature_importances_
    
    return L1_ts, L2_ts, imp

def rf(xs_train, y_train, xs_test, y_test):
    forest = RandomForestRegressor(n_estimators=10)
    L1, L2, imp = train(forest, xs_train, y_train, xs_test, y_test)
    return L1, L2, imp


def gbr(xs_train, y_train, xs_test, y_test):
    regressor = GradientBoostingRegressor()
    L1, L2, imp = train(regressor, xs_train, y_train, xs_test, y_test)
    return L1, L2, imp


def run(filename):
    t_names = ['doc', 'no3', 'tn', 'tp']
    x_names = ['area', 'elev', 'forest', 'wetland', 'urban', 'ag', 'roads', 'pop']  # TODO: removed shrub and precip because if together everything has NA values
    
    xs, ts = get_data(filename, x_names, t_names)
    
    for i in range(len(t_names)):  
        rf_imps = []
        gb_imps = []  
        for _ in range(20):
            kf = KFold(n_splits=5, shuffle=True)
            for train_index, test_index in kf.split(xs):
                xs_train, xs_test = xs[train_index], xs[test_index]
                y_train, y_test = ts[train_index, i], ts[test_index, i]
                _, _, rf_imp = rf(xs_train, y_train, xs_test, y_test)
                _, _, gb_imp = gbr(xs_train, y_train, xs_test, y_test)
                rf_inds = np.argpartition(rf_imp, -2)[-2:]
                gb_inds = np.argpartition(gb_imp, -2)[-2:]
                rf_imps.extend(np.array(x_names)[rf_inds])
                gb_imps.extend(np.array(x_names)[gb_inds])
        rf_count = Counter(rf_imps)
        gb_count = Counter(gb_imps)
                
        rf_count = sorted(rf_count.items())
        gb_count = sorted(gb_count.items())
    
        print('-----------')
        print('target: {}'.format(t_names[i]))
        print('-----------')
        for ftr_name, freq in rf_count:
            print('{}\t{}'.format(ftr_name, freq))
        print('---')
        for ftr_name, freq in gb_count:
            print('{}\t{}'.format(ftr_name, freq))
        
        

if __name__ == '__main__':
    import sys
    filename = sys.argv[1]
    run(filename)

import csv
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from collections import Counter
from sklearn.model_selection import KFold
from pdb import set_trace
from mltools import *



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
                _, _, _, _, rf_imp = rf(xs_train, y_train, xs_test, y_test, True)
                _, _, _, _, gb_imp = gbr(xs_train, y_train, xs_test, y_test, True)
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

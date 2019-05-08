import numpy as np
from sklearn.model_selection import KFold
from pdb import set_trace
import csv
from mltools import *
    

    
def run(datafile):
    models = [rf, gbr]
    t_names = ['doc', 'no3', 'tn', 'tp']
    x_names = ['area', 'elev', 'forest', 'wetland', 'urban', 'ag', 'roads', 'pop']  # TODO: removed shrub and precip because if together everything has NA values
    
    xs, ts = get_data(datafile, x_names, t_names)
    
    for t_ind in range(len(t_names)):
        print('Predicting {}'.format(t_names[t_ind]))

        tr_L1s = [[] for _ in range(len(models))]
        ts_L1s = [[] for _ in range(len(models))]
        tr_L2s = [[] for _ in range(len(models))]
        ts_L2s = [[] for _ in range(len(models))]
        kf = KFold(n_splits=5, shuffle=True)
        for train_index, test_index in kf.split(xs):
            xs_train, xs_test = xs[train_index], xs[test_index]
            y_train, y_test = ts[train_index, t_ind], ts[test_index, t_ind]

            # with open('./output/pred.csv', 'w') as csvfile:
            #     w = csv.writer(csvfile)
            #     w.writerow(y_test)

            for m_ind, model in enumerate(models):
                L1_tr, L2_tr, L1_ts, L2_ts, _ = model(xs_train, y_train, xs_test, y_test)
                tr_L1s[m_ind].append(L1_tr)
                ts_L1s[m_ind].append(L1_ts)
                tr_L2s[m_ind].append(L2_tr)
                ts_L2s[m_ind].append(L2_ts)
                print('tr {}:\tL1: {}\tL2: {}'.format(str(model).split()[1], round(L1_tr, 3), round(L2_tr, 3)))
                print('ts {}:\tL1: {}\tL2: {}'.format(str(model).split()[1], round(L1_ts, 3), round(L2_ts, 3)))

        with open('./output/res_{}.csv'.format(t_names[t_ind]), 'w') as csvfile:
            w = csv.writer(csvfile)
            for m_ind, model in enumerate(models):
                w.writerow([str(model).split()[1]])
                w.writerow(['avg train L1', sum(tr_L1s[m_ind])/len(tr_L1s[m_ind]), 
                            'avg test L1', sum(ts_L1s[m_ind])/len(ts_L1s[m_ind])])
                w.writerow(['avg train L2', sum(tr_L2s[m_ind])/len(tr_L2s[m_ind]), 
                            'avg test L2', sum(ts_L2s[m_ind])/len(ts_L2s[m_ind])])
    
        
    

if __name__ == '__main__':
    import sys
    filename = sys.argv[1]
    run(filename)
    

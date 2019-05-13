import numpy as np
from sklearn.model_selection import KFold
import csv
from mltools import *
import os
import data    

    
def run(xs, ts, output_dir):
    models = [gbr, rf, svm, rd]
    k = 3
    results = np.zeros((len(ts.columns), len(models), k, 4), dtype=np.float64)

    csvfile = open(os.path.join(output_dir, 'predictions.csv'), 'w')
    pred_writer = csv.writer(csvfile)
    
    for t_ind in range(len(ts.columns)):
        print('Predicting {}'.format(ts.columns.values[t_ind]))

        pred_writer.writerow([ts.columns.values])
        
        kf = KFold(n_splits=k, shuffle=True)
        for f_ind, (train_index, test_index) in enumerate(kf.split(xs)):
            xs_train, xs_test = xs.iloc[train_index], xs.iloc[test_index]
            y_train, y_test = ts.iloc[train_index, t_ind], ts.iloc[test_index, t_ind]

            xs_train, xs_test = data.preprocess(xs_train, xs_test)
            # xs_train, y_train = data.remove_outliers(xs_train, y_train, xs.columns, num_stdev=3)

            pred_writer.writerow(y_test)
            for m_ind, model in enumerate(models):
                L1_tr, L2_tr, L1_ts, L2_ts, _ = model(xs_train, y_train, xs_test, y_test, False, pred_writer)
                
                results[t_ind, m_ind,f_ind] = L1_tr, L2_tr, L1_ts, L2_ts

                print('tr {}:\tL1: {}\tL2: {}'.format(str(model).split()[1], round(L1_tr, 3), round(L2_tr, 3)))
                print('ts {}:\tL1: {}\tL2: {}'.format(str(model).split()[1], round(L1_ts, 3), round(L2_ts, 3)))


    with open(os.path.join(output_dir, 'accuracy.csv'), 'w') as res_file:
        res_writer = csv.writer(res_file)
        for t_ind, t_name in enumerate(ts.columns.values):
            res_writer.writerow([t_name])
            for m_ind, model in enumerate(models):
                res_writer.writerow([str(model).split()[1]])
                L1_tr, L2_tr, L1_ts, L2_ts = np.average(results[t_ind, m_ind], axis=0)
                res_writer.writerow(['avg train L1', L1_tr, 'avg test L1', L1_ts])
                res_writer.writerow(['avg train L2', L2_tr, 'avg test L2', L2_ts])

    csvfile.close()
    

if __name__ == '__main__':
    import sys
    if sys.argv[1] == 'NRSA':
        xs, ts = data.get_NRSA_data(for_training=True)
    else: 
        xs, ts = data.get_NLA_data(for_training=True)
    output_dir = './NRSA_output' if sys.argv[1] == 'NRSA' else './NLA_output'

    run(xs, ts, output_dir)
    

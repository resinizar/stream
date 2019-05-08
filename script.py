import numpy as np
from sklearn.model_selection import train_test_split
from pdb import set_trace
import csv
from mltools import *
    

    
def run(datafile):   
    t_names = ['doc', 'no3', 'tn', 'tp']
    x_names = ['area', 'elev', 'forest', 'wetland', 'urban', 'ag', 'roads', 'pop']  # TODO: removed shrub and precip because if together everything has NA values
    
    xs, ts = get_data(datafile, x_names, t_names)
    
    for i in range(len(t_names)):
        print('Predicting {}'.format(t_names[i]))
        X_train, X_test, y_train, y_test = train_test_split(xs, ts[:,i], test_size=.25)
    
        with open('./output/pred.csv', 'w') as csvfile:
            w = csv.writer(csvfile)
            w.writerow(y_test)
    
        for fun in [rf, gbr, svm]:
            L1_tr, L2_tr, L1_ts, L2_ts, _ = fun(X_train, y_train, X_test, y_test)
            print('tr {}:\tL1: {}\tL2: {}'.format(str(fun).split()[1], round(L1_tr, 3), round(L2_tr, 3)))
            print('ts {}:\tL1: {}\tL2: {}'.format(str(fun).split()[1], round(L1_ts, 3), round(L2_ts, 3)))
    

if __name__ == '__main__':
    import sys
    filename = sys.argv[1]
    run(filename)
    

import csv
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from collections import Counter
from sklearn.model_selection import KFold
from pdb import set_trace
from mltools import *



def plot_imp(imp, x_names, ax):
    imp = 100 * (imp / np.sum(imp))
    sorted_ind = np.argsort(imp)
    ax.barh(x_names[sorted_ind], imp[sorted_ind], color='#635a4d')


def run():
    xs, ts, x_names, t_names = get_NRSA_data()

    f, axarr = plt.subplots(len(t_names), 2, sharex='col')
    f.set_size_inches(14, 20)
    axarr[0, 0].set_title('Random Forest')
    axarr[0, 1].set_title('Gradient Boosting')
    
    for i in range(len(t_names)): 
        axarr[i, 0].annotate(t_names[i],xy=(0, 0.5), xytext=(-axarr[i, 0].yaxis.labelpad-5,0), 
            xycoords=axarr[i, 0].yaxis.label, textcoords='offset points', 
            size='large', ha='right', va='center')

        rf_imps = np.zeros(len(x_names))
        gb_imps = np.zeros(len(x_names))
        for _ in range(20):
            kf = KFold(n_splits=5, shuffle=True)
            for train_index, test_index in kf.split(xs):
                xs_train, xs_test = xs[train_index], xs[test_index]
                y_train, y_test = ts[train_index, i], ts[test_index, i]
                _, _, _, _, rf_imp = rf(xs_train, y_train, xs_test, y_test, True)
                _, _, _, _, gb_imp = gbr(xs_train, y_train, xs_test, y_test, True)
                rf_imps += rf_imp
                gb_imps += gb_imp

        plot_imp(rf_imps, np.array(x_names), axarr[i, 0])
        plot_imp(gb_imps, np.array(x_names), axarr[i, 1])

    f.savefig('./output/graphs/ftr_imp.png')
        
        
if __name__ == '__main__':
    run()

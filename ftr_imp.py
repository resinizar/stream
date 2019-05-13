import numpy as np
from sklearn.model_selection import KFold
import mltools
import data
import matplotlib.pyplot as plt
import os



def plot_imp(imp, x_names, ax):
    imp = 100 * (imp / np.sum(imp))
    sorted_ind = np.argsort(imp)
    ax.barh(x_names[sorted_ind], imp[sorted_ind], color='#635a4d')


def run(xs, ts, output_dir):
    
    # set up plots
    f, axarr = plt.subplots(len(ts.columns.values), 2, sharex='col')
    f.set_size_inches(14, 20)
    axarr[0, 0].set_title('Random Forest')
    axarr[0, 1].set_title('Gradient Boosting')
    
    for i in range(len(ts.columns.values)):  # for each target

        # some labels
        axarr[i, 0].annotate(ts.columns.values[i],xy=(0, 0.5), xytext=(-axarr[i, 0].yaxis.labelpad-5,0), 
            xycoords=axarr[i, 0].yaxis.label, textcoords='offset points', 
            size='large', ha='right', va='center')

        rf_imps = np.zeros(len(xs.columns.values))
        gb_imps = np.zeros(len(xs.columns.values))
        for _ in range(20):  # perform 5-fold cross validation 20 times
            kf = KFold(n_splits=5, shuffle=True)
            for train_index, test_index in kf.split(xs):
                xs_train, xs_test = xs.iloc[train_index], xs.iloc[test_index]
                y_train, y_test = ts.iloc[train_index, i], ts.iloc[test_index, i]

                xs_train, xs_test = data.preprocess(xs_train, xs_test)
                # xs_train, y_train = data.remove_outliers(xs_train, y_train, xs.columns, num_stdev=3)
                _, _, _, _, rf_imp = mltools.rf(xs_train, y_train, xs_test, y_test, True)
                _, _, _, _, gb_imp = mltools.gbr(xs_train, y_train, xs_test, y_test, True)
                rf_imps += rf_imp
                gb_imps += gb_imp

        plot_imp(rf_imps, xs.columns.values, axarr[i, 0])
        plot_imp(gb_imps, xs.columns.values, axarr[i, 1])

    f.savefig(os.path.join(output_dir, 'ftr_imp_imputed_rem3.png'))
        
        
if __name__ == '__main__':
    import sys
    if sys.argv[1] == 'NRSA':
        xs, ts = data.get_NRSA_data(for_training=True)
    else: 
        xs, ts = data.get_NLA_data(for_training=True)
    output_dir = './NRSA_output' if sys.argv[1] == 'NRSA' else './NLA_output'
    run(xs, ts, output_dir)

from matplotlib import pyplot as plt
from pdpbox import pdp
from sklearn.model_selection import KFold
from mltools import *
import data
import os


def run(xs, ts, model, output_dir):
    x_names = xs.columns.values
    for target in ts.columns.values:
        logfile = open(os.path.join(output_dir, '{}_zlog.txt'.format(target)), 'w')
        print('Using model: {}'.format(model), file=logfile)

        kf = KFold(n_splits=3, shuffle=True)
        for f_ind, (train_index, test_index) in enumerate(kf.split(xs)):
            xs_train, xs_test = xs.iloc[train_index], xs.iloc[test_index]
            y_train, y_test = ts[target].iloc[train_index], ts[target].iloc[test_index]


            xs_train, xs_test = data.preprocess(xs_train, xs_test)
            # xs_train, y_train = data.remove_outliers(xs_train, y_train, xs.columns, num_stdev=3)
            model.fit(xs_train, y_train)
            pred_train = model.predict(xs_train)
            pred_test = model.predict(xs_test)
            L1_tr = np.average(np.abs(np.subtract(pred_train, y_train.values)))
            L1_ts = np.average(np.abs(np.subtract(pred_test, y_test.values)))
            print('fold {}: trL1: {} tsL1: {}'.format(f_ind+1, round(L1_tr, 3), round(L1_ts, 3)), file=logfile)

            imp = model.feature_importances_
            inds = np.argsort(imp)[::-1]  

            for ind in inds:
                plot = pdp.pdp_isolate(model=model, dataset=xs_test, model_features=x_names, feature=x_names[ind])

                f, ax = pdp.pdp_plot(plot, x_names[ind])
                plt.savefig(os.path.join(output_dir, '{}_{}_fold{}.png'.format(target, x_names[ind], str(f_ind+1))))
        logfile.close()


if __name__ == '__main__':
    import sys
    if sys.argv[1] == 'NRSA':
        xs, ts = data.get_NRSA_data(for_training=True)
    else: 
        xs, ts = data.get_NLA_data(for_training=True)
    output_dir = './NRSA_output/pdp' if sys.argv[1] == 'NRSA' else './NLA_output/pdp'

    model = GradientBoostingRegressor(n_estimators=50)
        
    run(xs, ts, model, output_dir)

from matplotlib import pyplot as plt
from pdpbox import pdp, get_dataset, info_plots
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from mltools import *


def run(datafile, model, target, logfile):
    print('Using model: {}'.format(model), file=logfile)
    with open(datafile, 'r') as data_file:
        df = pd.read_csv(data_file)
        df = df.drop(['sort', 'year', 'siteid', 'lat', 'long'], axis=1)  # drop columns I don't need
        len_before = len(df)
        df = df.dropna()  # drop any rows with unknown values
        print('Dropped {} rows for a total of {}'.format(len_before - len(df), len(df)), file=logfile)
        print('Target: {}'.format(target), file=logfile)

    xs = df.iloc[:, 4:]
    x_names = df.columns.values[4:]
    ts = df.iloc[:, 0:4]
    t_names = df.columns.values[0:4]


    kf = KFold(n_splits=3, shuffle=True)
    for f_ind, (train_index, test_index) in enumerate(kf.split(xs)):
        xs_train, xs_test = xs.iloc[train_index], xs.iloc[test_index]
        y_train, y_test = ts['doc'].iloc[train_index], ts['doc'].iloc[test_index]

        model.fit(xs_train, y_train)
        pred_train = model.predict(xs_train)
        pred_test = model.predict(xs_test)
        L1_tr = np.average(np.abs(np.subtract(pred_train, y_train.values)))
        L1_ts = np.average(np.abs(np.subtract(pred_test, y_test.values)))
        print('fold {}: trL1: {} tsL1: {}'.format(f_ind+1, round(L1_tr, 3), round(L1_ts, 3)), file=logfile)

        imp = model.feature_importances_
        inds = np.argsort(imp)[::-1]  

        for ind in inds:
            pdp_goals = pdp.pdp_isolate(model=model, dataset=xs_test, model_features=x_names, feature=x_names[ind])

            f, ax = pdp.pdp_plot(pdp_goals, x_names[ind])
            plt.savefig('./output/pdp/{}_{}_{}.png'.format(target, ind, str(f_ind+1)))


if __name__ == '__main__':
    import sys
    datafile = sys.argv[1]
    model = GradientBoostingRegressor(n_estimators=50)
    target = 'no3no2'
    logfilename = './output/pdp/{}_log.txt'.format(target)
    with open(logfilename, 'w') as logfile:
        run(datafile, model, target, logfile)

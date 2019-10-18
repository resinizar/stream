import os
import pandas as pd
from sklearn.impute import SimpleImputer
from copy import deepcopy as cpy
from sklearn.preprocessing import normalize
import numpy as np



data_path = './data'
NRSA_targets = ['doc', 'no3', 'tn', 'tp']
NLA_targets = ['doc', 'no3no2', 'tn', 'tp']
NRSA2_targets = ['DOC', 'NTL_UG_L', 'PTL', 'NTL_COND', 'PTL_COND']


def get_NLA_data(for_training=False):
    with open(os.path.join(data_path, 'NLA07_mldata.csv'), 'r') as data_file:
        df = pd.read_csv(data_file)

    if for_training:
        # orig_nsamples, orig_nftrs = df.shape
        df = df.dropna(subset=NLA_targets)  # drop rows where no target
        df = df.drop(['sort', 'siteid', 'year'], axis=1)
        return df.drop(NLA_targets, axis=1), df[NLA_targets]
    else:
        return df


def get_NRSA_data(for_training=False):
    with open(os.path.join(data_path, 'NRSA08_mldata.csv'), 'r') as data_file:
        df = pd.read_csv(data_file)

    if for_training:
        # orig_nsamples, orig_nftrs = df.shape
        df = df.dropna(subset=NRSA_targets)  # drop rows where no target
        df = df.drop(['sort', 'siteid', 'year', 'epareg', 'strorder'], axis=1)
        return df.drop(NRSA_targets, axis=1), df[NRSA_targets]
    else:
        return df


def get_NRSA2_data(for_training=False):
    with open(os.path.join(data_path, 'NRSA1314_mldata.csv'), 'r') as data_file:
        df = pd.read_csv(data_file)

    if for_training:
        orig_nsamples, orig_nftrs = df.shape
        df = df.dropna(subset=NRSA2_targets)  # drop any rows that do not have a target
        df = df.drop(['SITE_ID', 'MISS_BASIN_NM', 'FS_EW', 'MICX_RESULT', 'MICX_RL', 'MERCURY_FLAG', 'MICX_MDL'], axis=1)  # drop unneeded columns & ones with over half missing vals
        # NOTE: MICX_MDL removed because all vals were 0.1 or NaN and was messing with normalization, could be converted to binary
        df = df.select_dtypes(include=[np.number])  # only include numeric data
        return df.drop(NRSA2_targets[:3], axis=1), df[NRSA2_targets[:3]]  

    else:
        return df


def preprocess(train, valid):

    # impute with median
    imputer = SimpleImputer(strategy='median')
    imp_train = pd.DataFrame(imputer.fit_transform(train))
    imp_train.columns = train.columns
    imp_train.index = train.index
    imp_valid = pd.DataFrame(imputer.transform(valid))
    imp_valid.columns = valid.columns
    imp_valid.index = valid.index

    imp_train_norm = ((imp_train - imp_train.min()) / (imp_train.max() - imp_train.min()))  # norm train 0-1
    imp_valid_norm = ((imp_valid - imp_train.min()) / (imp_train.max() - imp_train.min()))  # norm valid according to train



    return imp_train_norm, imp_valid_norm


def remove_outliers(xs, ts, cols, num_stdev=6):
    rem_xs = cpy(xs)
    rem_ts = cpy(ts)
    for col in cols:
        mean, stdev = rem_xs[col].mean(), rem_xs[col].std()
        outliers = rem_xs.loc[rem_xs[col] > mean + num_stdev*stdev, col]

        if not outliers.empty:
            rem_xs = rem_xs.drop(outliers.index)
            rem_ts = rem_ts.drop(outliers.index)
    print('Removed {} outliers for a new total of {}'.format(xs.shape[0]-rem_xs.shape[0], rem_xs.shape[0]))
    return rem_xs, rem_ts

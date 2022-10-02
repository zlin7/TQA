import os.path
import numpy as np
import pandas as pd

from _settings import WORKSPACE, DATA_PATH
from data.online_dataset import get_split_idx, DatasetWrapperFull

COVID_ROOT = os.path.join(DATA_PATH, 'COVID')

def get_normalized_daily_counts(startdate='2022-03-01', enddate='2022-03-31'):
    #Download the data from https://coronavirus.data.gov.uk/
    df = pd.read_csv(os.path.join(COVID_ROOT, "ltla_2022-04-14.csv"))
    df = df.pivot_table(values=['newCasesBySpecimenDate'], index='date', columns='areaCode').fillna(0.)
    tdf = df.rolling(window=365, min_periods=365).mean().dropna().shift(1)
    rate = df.reindex(tdf.index) / tdf
    return rate.loc[startdate:enddate]

def get_splits(conformal=True, seed=7, horizon=30, n_train=200, n_calibration=100, n_test=80):
    data = np.expand_dims(get_normalized_daily_counts().T.values, 2)
    Xs, Ys = data[:, :-1], data[:, 1:]
    train_idx, calibration_idx, train_calibration_idx, test_idx = get_split_idx(n_train, n_calibration, n_test, seed=seed)

    if conformal:
        kwargs = {"max_full_length": horizon}
        train_dataset = DatasetWrapperFull(Xs[train_idx], Ys[train_idx], **kwargs)
        calibration_dataset = DatasetWrapperFull(Xs[calibration_idx], Ys[calibration_idx], **kwargs)
    else:
        kwargs = {"max_full_length": horizon}
        train_dataset = DatasetWrapperFull(Xs[train_calibration_idx], Ys[train_calibration_idx], **kwargs)
        calibration_dataset = None
    test_dataset = DatasetWrapperFull(Xs[test_idx], Ys[test_idx], **kwargs)
    return train_dataset, calibration_dataset, test_dataset


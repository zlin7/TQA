import numpy as np
import pandas as pd
import os
from _settings import DATA_PATH, GEFCom_NAME
from data.online_dataset import DatasetWrapperFull

GEFC_ROOT = os.path.join(DATA_PATH, GEFCom_NAME)

from datetime import datetime

def read_data(sanity_check=False):
    df = pd.read_csv(os.path.join(GEFC_ROOT, 'L1-train.csv'))
    dr = pd.date_range('20010101', '20101001')
    df['time'] = pd.Index([datetime(d.year, d.month, d.day, hour=i) for d in dr for i in range(24)][1:-23])
    if sanity_check:
        _sanity_check = df['time'].map(lambda x: f"{x.month}{x.day}{x.year} {x.hour}:00")
        assert _sanity_check.eq(df['TIMESTAMP']).all()
        assert df['time'].sort_values().eq(df['time']).all()
    df = df.drop(['TIMESTAMP', 'ZONEID'], axis=1)
    df['time'] = pd.Index([datetime(d.year, d.month, d.day, hour=i) for d in dr for i in range(24)][0:-24])
    df['date'] = df['time'].map(lambda x: datetime.strftime(x, "%Y%m%d"))

    #checked: since 20050101, load is always populated.
    return df.set_index('time')

#2099 days of load data
def get_gefc_data(conformal=True, horizon=24, n_train=1198, n_calibration=200, n_test=700,
                  random_split=False, seed=None,
                      **kwargs):
    df = read_data(sanity_check=False).dropna().drop('date', axis=1) #faster

    assert len(df) == (n_calibration + n_test + n_train) * 24 + (horizon)
    #normalize
    _mean = df.iloc[:(n_train*24)].mean()
    _std = df.iloc[:(n_train * 24)].std()
    df = (df - _mean) / _std
    #print(f"Reducing by a scale of: {_std['LOAD']}")

    df['pred_LOAD'] = df['LOAD'].shift(1)
    Y = np.expand_dims(df['LOAD'].iloc[1:].values, 1)
    X = df.reindex(columns=['pred_LOAD'] + ['w%d'%(i+1) for i in range(25)]).iloc[1:].values

    Xs = []
    Ys = []
    for y_st in range(horizon - 1, len(df) - 1, horizon):
        Ys.append(Y[y_st:y_st+horizon])
        Xs.append(X[y_st: y_st + horizon])

    assert seed is None or (random_split and seed is not None)
    if random_split:
        perm = np.random.RandomState(seed=seed).permutation(n_train + n_calibration + n_test)
        Xs = [Xs[i] for i in perm]
        Ys = [Ys[i] for i in perm]

    if conformal:
        kwargs = {"max_full_length": horizon}
        train_dataset = DatasetWrapperFull(Xs[:n_train], Ys[:n_train], **kwargs)
        calibration_dataset = DatasetWrapperFull(Xs[n_train:n_train+n_calibration], Ys[n_train:n_train+n_calibration], **kwargs)
    else:
        kwargs = {"max_full_length": horizon}
        train_dataset = DatasetWrapperFull(Xs[:n_train+n_calibration], Ys[:n_train+n_calibration], **kwargs)
        calibration_dataset = None
    test_dataset = DatasetWrapperFull(Xs[n_train+n_calibration:], Ys[n_train+n_calibration:], **kwargs)

    return train_dataset, calibration_dataset, test_dataset



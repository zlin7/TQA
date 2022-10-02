import pandas as pd
import torch

from _settings import WORKSPACE, DATA_PATH, MIMIC_NAME
import os, tqdm
from data.online_dataset import DatasetWrapperFull, get_split_idx

MIMIC_ROOT = os.path.join(DATA_PATH, 'MIMIC')
"""
--SQL Queries 


SELECT distinct p1.subject_id
INTO mimiciii.target_patients_20220420
FROM mimiciii.patients as p1
inner join mimiciii.prescriptions as t3
    on t3.subject_id = p1.subject_id
where  t3.drug_name_generic = 'Levofloxacin'; -- restrict our study to the patients on antibiotics


--chartevents.csv
select c1.*
INTO mimiciii.chartevents_target_patients_20220420
from mimiciii.chartevents as c1
    inner join mimiciii.target_patients_20220420 as p1
        on c1.subject_id = p1.subject_id
    where c1.itemid in (1538,225690,1525,220615,8368,220180,223835,3420,807,225664,211,220045,813,220545,814,220228,52,220181,828,227457,1535,227442,618,220210,220277,646,51,220179,223761,678,1542,220546,763,224639); --select the relevant events/measurements


--patients.csv
select t.subject_id, t.gender, t.dob, t.dod, t.dod_hosp, t.dod_ssn, min(a.admittime) as first_admit
from mimiciii.patients as t
inner join mimiciii.target_patients_20220420 as p1
    on t.subject_id = p1.subject_id
inner join mimiciii.admissions a on p1.subject_id = a.subject_id
group by t.subject_id, t.gender, t.dob, t.dod, t.dod_hosp, t.dod_ssn;

--antibiotic.csv
select t3.*
from mimiciii.prescriptions as t3
inner join mimiciii.target_patients_20220420 as p1
    on t3.subject_id = p1.subject_id
where  t3.drug_name_generic = 'Levofloxacin';


"""

def to_date(x):
    return x[:10]


def _get_antibiotic_df():
    antibiotics = pd.read_csv(os.path.join(MIMIC_ROOT, 'antibiotic.csv'))
    antibiotics = antibiotics[antibiotics['dose_unit_rx'] == 'mg'].dropna(subset=['startdate', 'enddate'])
    for c in ['startdate', 'enddate']: antibiotics[c] = antibiotics[c].map(to_date).map(pd.to_datetime)
    antibiotics = antibiotics.reindex(columns=['startdate', 'enddate', 'subject_id', 'dose_val_rx'])
    return antibiotics

def _process_mimic_data(autoregressive=False, Y_col ='WBC high'):
    summ_cols = ['Bilirubin', 'Creatinine', 'Hematocrit', 'Hemoglobin', 'Platelet', 'Potassium', 'WBC']
    mean_cols = ['Diastolic blood pressure', 'Glucose', 'Mean blood pressure', 'SpO2', 'Systolic blood pressure',
                 'Temperature', 'Weight']
    mean_cols += ['Heart Rate', 'Respiratory Rate']

    df = pd.read_csv(os.path.join(MIMIC_ROOT, 'processed.csv'))
    df['date'] = df['date'].map(pd.to_datetime)
    df = df.reindex(columns=[f"{c} high" for c in summ_cols] + [f"{c} low" for c in summ_cols] + mean_cols + ['date', 'subject_id', 'age'])

    adf = _get_antibiotic_df()

    X = {}
    Y = {}
    L = {}
    for subject_id, tdf in tqdm.tqdm(df.groupby('subject_id')):
        tdf = tdf.set_index('date').sort_index().drop('subject_id', axis=1)
        x = tdf[Y_col].iloc[:-1] if autoregressive else tdf.iloc[:-1]
        y = pd.Series(tdf[Y_col].iloc[1:].values, index=x.index)
        msk = ~pd.isnull(y)
        x, y = x[msk], y[msk]
        if len(y) == 0: continue

        #since antibiotic is an intervention, we use the prediction date to make the X
        tadf = adf[adf['subject_id'] == subject_id]
        x['Antibiotics'] = 0.
        for dt in y.index:
            for idx in tadf.index:
                if tadf.loc[idx, 'startdate'] <= dt < tadf.loc[idx, 'enddate']:
                    x.loc[dt, 'Antibiotics'] += tadf.loc[idx, 'dose_val_rx']
        X[subject_id], Y[subject_id], L[subject_id] = x, y, len(y)
    return X, Y, pd.Series(L)


def process_mimic_data(autoregressive=False, Y_col ='WBC high'):
    cache_path = os.path.join(WORKSPACE, 'processed_data', MIMIC_NAME, f'{autoregressive}_{Y_col}.pkl')
    if not os.path.isfile(cache_path):
        if not os.path.isdir(os.path.dirname(cache_path)): os.makedirs(os.path.dirname(cache_path))
        res = _process_mimic_data(autoregressive, Y_col)
        pd.to_pickle(res, cache_path)
    return pd.read_pickle(cache_path)


def normalize_data(X, mean=None, std=None):
    ret_mean_std = False
    if mean is None:
        assert std is None
        ret_mean_std = True
        _tX = pd.concat(X, ignore_index=True).describe()
        mean, std = _tX.loc['mean'], _tX.loc['std']
    try:
        mean['Antibiotics'] = 0
    except:
        assert isinstance(mean, float)

    ret = []
    for x in X:
        if len(x.shape) == 1:
            assert len(x) == x.count()
            x = torch.tensor(((x - mean) / std).values, dtype=torch.float).unsqueeze(1)
        else:
            x.iloc[0] = x.iloc[0].fillna(mean)
            x = x.fillna(method='ffill')
            x = torch.tensor(((x - mean) / std).values, dtype=torch.float)
        ret.append(x)
    return (ret, mean, std) if ret_mean_std else ret

def get_splits(conformal=True, seed=7, horizon=30,
               #n_train=267, n_calibration=167, n_test=100,
               n_train=192, n_calibration=100, n_test=100,
               autoregressive = False, Y_col='WBC high'):
    X, Y, L = process_mimic_data(autoregressive, Y_col) # X and Y are already aligned - so there is no informtion leak
    #ipdb.set_trace()
    L = L[L >= horizon]
    train_idx, calibration_idx, train_calibration_idx, test_idx = get_split_idx(n_train, n_calibration, n_test, seed=seed, idx=L.index)

    if conformal:
        X_train, _mean, _std = normalize_data([X[k].iloc[:horizon] for k in train_idx])
        Y_train, _mean_Y, _std_Y = normalize_data([Y[k].iloc[:horizon] for k in train_idx])
        X_calibration = normalize_data([X[k].iloc[:horizon] for k in calibration_idx], _mean, _std)
        Y_calibration = normalize_data([Y[k].iloc[:horizon] for k in calibration_idx], _mean_Y, _std_Y)
    else:
        raise NotImplementedError()
    X_test = normalize_data([X[k].iloc[:horizon] for k in test_idx], _mean, _std)
    Y_test = normalize_data([Y[k].iloc[:horizon] for k in test_idx], _mean_Y, _std_Y)
    kwargs = {"max_full_length": horizon}
    train_dataset = DatasetWrapperFull(X_train, Y_train, **kwargs)
    calibration_dataset = DatasetWrapperFull(X_calibration, Y_calibration, **kwargs) if conformal else None
    test_dataset = DatasetWrapperFull(X_test, Y_test, **kwargs)
    return train_dataset, calibration_dataset, test_dataset

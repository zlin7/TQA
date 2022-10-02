import gzip
import os
import pathlib
import pickle

import numpy as np
from scipy.signal import resample

from _settings import WORKSPACE, DATA_PATH
from data.online_dataset import get_split_idx, DatasetWrapperFull

EEG_ROOT = os.path.join(DATA_PATH, 'EEG')
eeg_root_train = os.path.join(EEG_ROOT, 'SMNI_CMI_TRAIN')
eeg_root_test = os.path.join(EEG_ROOT, 'SMNI_CMI_TEST')


def parse_eeg_file(filename, ):
    with gzip.open(filename, "rb") as f:
        chans = {}
        for line in f:
            tokens = line.decode("ascii").split()
            if tokens[0] != "#":
                if tokens[1] not in chans.keys():
                    chans[tokens[1]] = []
                chans[tokens[1]].append(float(tokens[3]))
        chan_arrays = []
        for chan in chans.values():
            chan_arrays.append(chan)
    return chan_arrays


def get_raw_eeg_data(split="train", include_alcoholic_class=False, cached=True):
    if split == "train":
        root = eeg_root_train
    else:
        root = eeg_root_test
    filepath = os.path.join(EEG_ROOT, f"eeg_by_chans_{split}.pkl")
    if cached and os.path.exists(filepath):
        with open(filepath, "rb") as f:
            dataset = pickle.load(f)
    else:
        dataset = []
        for folder in os.listdir(root):
            if folder != "README" and (include_alcoholic_class or folder[3] == "c"):
                subfolder = os.path.join(root, folder)
                for filename in os.listdir(subfolder):
                    f = os.path.join(subfolder, filename)
                    if ".gz" in pathlib.Path(f).suffixes:
                        chan_arrays = parse_eeg_file(f)
                        dataset.append(chan_arrays)
        with open(filepath, "wb") as f:
            pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)

    return dataset


def get_downsampled_sequences(split, total_length):
    raw_data = np.asarray(get_raw_eeg_data(split))
    return resample(raw_data, total_length, axis=2)


def get_splits(conformal=True, seed=7, horizon=63, n_train=300, n_calibration=100, n_test=200, channel=0):
    #random_split means test set is being permuted as well
    data_train = get_downsampled_sequences('train', horizon + 1)[:, channel]
    data_cal_test = get_downsampled_sequences('test', horizon + 1)[:, channel]
    data_train = data_train[data_train.std(1) > 0] #remove the meaningless sequences
    data_cal_test = data_cal_test[data_cal_test.std(1) > 0]  # remove the meaningless sequences
    assert len(data_cal_test) == (n_test + n_calibration) and len(data_train) == n_train
    data_cal_test, data_train = np.expand_dims(data_cal_test, 2), np.expand_dims(data_train, 2)
    _, _, test_idx, calibration_idx = get_split_idx(n_test - 1, 1, n_calibration, seed=seed, idx=np.arange(len(data_cal_test)))
    if conformal:
        data_test = data_cal_test[test_idx]
        data_cal = data_cal_test[calibration_idx]
        _mean, _std = data_train.mean(), data_train.std()
        data_train = (data_train - _mean) / _std
        data_cal = (data_cal - _mean) / _std
        data_test = (data_test - _mean) / _std
        kwargs = {"max_full_length": horizon}
        calibration_dataset = DatasetWrapperFull(data_cal[:, :-1], data_cal[:, 1:], **kwargs)
    else:
        raise NotImplementedError()
    train_dataset = DatasetWrapperFull(data_train[:, :-1], data_train[:, 1:], **kwargs)
    test_dataset = DatasetWrapperFull(data_test[:, :-1], data_test[:, 1:], **kwargs)

    return train_dataset, calibration_dataset, test_dataset

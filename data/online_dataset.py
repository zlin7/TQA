import torch
from torch.utils.data import Dataset
import numpy as np

from _settings import COVID_NAME, MIMIC_NAME, GEFCom_NAME, EEG_NAME

HORIZON_LENGTHS = {MIMIC_NAME: 30, GEFCom_NAME: 24, COVID_NAME: 30, EEG_NAME: 63}
DEFAULT_NTESTS = {MIMIC_NAME: 100, COVID_NAME: 80, GEFCom_NAME: 700, EEG_NAME:200}
DEFAULT_NVALIDS  = {MIMIC_NAME: 100, COVID_NAME: 100, GEFCom_NAME: 200, EEG_NAME:100}

def pre_pad(x, max_length):
    lx = len(x)
    return torch.cat([torch.zeros([max_length - lx, x.shape[1]], dtype=torch.float), x],0)

def _to_device(data_or_model, device):
    if isinstance(device, tuple) or isinstance(device, list):
        device = device[0]
    def _to_device(d):
        try:
            return d.to(device)
        except: #if device is a list/tuple, we don't do anything as this should be dataparalle. (hacky, I know)
            return d
    if isinstance(data_or_model, tuple) or isinstance(data_or_model, list):
        return tuple([_to_device(x) for x in data_or_model])
    return _to_device(data_or_model)

class DatasetWrapperFull(Dataset):
    def __init__(self, X, Y, max_full_length):
        # Aligned at the end (aka pre-padding)
        super(DatasetWrapperFull, self).__init__()
        self.max_full_length = max_full_length
        self.X, self.Y = X, Y

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.Y[idx]
        if not (isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor)):
            x = torch.tensor(x, dtype=torch.float)
            y = torch.tensor(y, dtype=torch.float)
        lx, ly = len(x), len(y)
        x, y = pre_pad(x, self.max_full_length), pre_pad(y, self.max_full_length)
        return x, y, lx, ly

    @classmethod
    def _sep_data(cls, res):
        return res


def get_split_idx(n_train, n_calibration, n_test, seed, idx=None):
    total = n_train + n_calibration + n_test
    if idx is None:
        idx = np.arange(total)
    else:
        assert len(idx) == total
    perm = np.random.RandomState(seed=seed).permutation(n_train + n_calibration + n_test)
    train_idx = idx[perm[:n_train]]
    calibration_idx = idx[perm[n_train: n_train + n_calibration]]
    train_calibration_idx = idx[perm[: n_train + n_calibration]]
    test_idx = idx[perm[n_train + n_calibration:]]
    return train_idx, calibration_idx, train_calibration_idx, test_idx


def get_horizon(dataset):
    return HORIZON_LENGTHS[dataset.split("-")[0]]

def get_default_ntest(dataset):
    return DEFAULT_NTESTS[dataset.split("-")[0]]

def get_default_ncal(dataset):
    return DEFAULT_NVALIDS[dataset.split("-")[0]]

def get_default_data(dataset, conformal=True, seed=0, **kwargs):
    assert conformal
    if dataset.startswith(EEG_NAME):
        import data.preprocessing.eeg as eeg
        return eeg.get_splits(conformal=conformal, seed=seed, **kwargs)
    if dataset == MIMIC_NAME:
        import data.preprocessing.mimic as mimic
        return mimic.get_splits(conformal=conformal, seed=seed, **kwargs)
    if dataset.startswith(GEFCom_NAME):
        import data.preprocessing.gefc as gefc
        if dataset == f"{GEFCom_NAME}-R":
            return gefc.get_gefc_data(conformal=conformal, seed=seed, random_split=True, **kwargs)
        else:
            return gefc.get_gefc_data(conformal=conformal, seed=None, **kwargs)
    if dataset.startswith(COVID_NAME):
        import data.preprocessing.covid as covid
        return covid.get_splits(conformal=conformal, seed=seed, **kwargs)


if __name__ == "__main__":
    pass



import os.path

import torch
import numpy as np, os


def _torch_rank(x,dim=1):
    return torch.argsort(torch.argsort(x, dim),dim) / (x.shape[dim] - 1)

def inverse_nonconformity_L1(score, pred, t):
    return [pred - score, pred + score]

def quantile_regression_EWMA(_data, beta=0.8):
    w = 1. / torch.pow(beta, torch.arange(len(_data), device=_data.device))
    pred = torch.zeros_like(_data)
    for t in range(1, len(_data)):
        wt = w[-t:]
        wt = wt / wt.sum()
        pred[t] = torch.matmul(wt, _data[:t])
    return pred


class _PI_Constructor(torch.nn.Module):
    def __init__(self, base_model_path=None, **kwargs):
        super(_PI_Constructor, self).__init__()
        self.base_model_path = base_model_path
        assert os.path.isfile(self.base_model_path)
        self.base_model = torch.load(base_model_path, map_location=kwargs.get('device'))
        for param in self.base_model.parameters(): param.requires_grad = False

        self.kwargs = kwargs

        #some optional stuff
        self._update_cal_loc = 0 #if we want to update the calibration residuals in an online fashion


    def fit(self):
        raise NotImplementedError()

    def calibrate(self, calibration_dataset: torch.utils.data.Dataset, batch_size=32, device=None):
        self.base_model.eval()
        calibration_loader = torch.utils.data.DataLoader(calibration_dataset, batch_size=batch_size, shuffle=False)
        preds, ys = [], []
        with torch.no_grad():
            for calibration_example in calibration_loader:
                calibration_example = [_.to(device) for _ in calibration_example]
                sequences, targets, lengths_input, lengths_target = calibration_dataset._sep_data(calibration_example)
                out = self.base_model(sequences)
                preds.append(out)
                ys.append(targets)
        self.calibration_preds = torch.nn.Parameter(torch.cat(preds).float(), requires_grad=False)
        self.calibration_truths = torch.nn.Parameter(torch.cat(ys).float(), requires_grad=False)
        return

    def get_nonconformity_scores(self, cal_pred, cal_y, test_pred, test_y):
        #The most common nonconformity score
        return (cal_pred - cal_y).abs(), (test_pred - test_y).abs()

    def get_adjusted_q(self, cal_pred, cal_y, test_pred, test_y, alpha, **kwargs):
        raise NotImplementedError()

    def predict(self, x, y, alpha=0.05, **kwargs):
        raise NotImplementedError()


def mask_scores(scores, scores_len):
    for i in range(len(scores_len)):
        scores[i, scores_len[i]:] = 0
    return scores

def adapt_by_error_t(pred, Y, cal_scores, cal_scores_len=None, gamma=0.005, alpha=0.1,
                     *,
                     scores=None,
                     rev_func=None,
                     two_sided=True,
                     #mapping scores back to lower and upper bound. Take argument (score, pred, t)
                     ):
    if scores is None and rev_func is None:
        scores = (pred - Y).abs()
        rev_func = inverse_nonconformity_L1
    else:
        assert not (scores is None or rev_func is None)
    assert len(pred.shape) == len(Y.shape) == 1
    L = len(Y)
    device = pred.device
    if cal_scores_len is not None:
        cal_scores = mask_scores(cal_scores, cal_scores_len)
        _sidx = torch.argsort(cal_scores_len, descending=True)
        cal_scores, cal_scores_len = cal_scores[_sidx], cal_scores_len[_sidx]

        ns = []
        qs = []
        for t in range(L):
            n = (cal_scores_len > t).int().sum().item()
            qs.append(torch.concat([torch.sort(cal_scores[:n, t], descending=False)[0], torch.ones(1, device=device) * torch.inf]))
            ns.append(n)
    else:
        ns = [cal_scores.shape[0]] * L
        qs = []
        for t in range(L):
            qs.append(torch.concat([torch.sort(cal_scores[:, t], descending=False)[0], torch.ones(1, device=device) * torch.inf]))

    def Q(a, t):
        q = 1-a
        vs = qs[t]
        n = ns[t]
        loc = torch.ceil(q * (n)).long().clip(0, len(vs) - 1)
        return vs[loc]

    a_ts = torch.ones(L + 1, device=device) * alpha
    err_ts = torch.empty(L, dtype=torch.float, device=device)
    w_ts = torch.empty(L, dtype=torch.float, device=device)

    pred_pis = []
    for t in range(L):
        w_ts[t] = Q(a_ts[t], t) #Get the current adjusted nonconformity score
        if gamma > 0:
            s_t = scores[t]  # get the actual nonconformity score
            err_ts[t] = (s_t > w_ts[t]).int()  # check if it's violated
            if (two_sided and (a_ts[t] > 1 or a_ts[t] < 0)) or ((not two_sided) and (a_ts[t] > 1)):
                a_ts[t + 1] = a_ts[t] + gamma * (alpha - a_ts[t])  # a_{t+1}
            else:
                a_ts[t + 1] = a_ts[t] + gamma * (alpha - err_ts[t])  # a_{t+1}
        pred_pis.append(rev_func(w_ts[t], pred[t], t))
    return torch.tensor(pred_pis, device=device)


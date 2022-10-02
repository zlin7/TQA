import torch
import numpy as np

from models.tqa.utils import _PI_Constructor
def scores_to_intervals(pred, Y, cal_scores, scales, alpha=0.1):
    # mapping scores back to lower and upper bound.
    assert len(pred.shape) == len(Y.shape) == 1
    N, L = cal_scores.shape
    device = pred.device
    qs = torch.concat([torch.sort(cal_scores, 0)[0], torch.ones([1, L], device=device) * torch.inf], 0)
    qloc = max(0, min(int(np.ceil((1-alpha) * N)), N))
    w_ts = qs[qloc, :]
    return torch.stack([pred - w_ts * scales, pred + w_ts * scales], 1)

class CFRNN(_PI_Constructor):
    def __init__(self, base_model_path=None, **kwargs):
        super(CFRNN, self).__init__(base_model_path, **kwargs)

    def predict(self, x, y, alpha=0.05, state=None, gamma=0.05, update_cal=False, **kwargs):
        pred = self.base_model(x, state)
        assert pred.shape[2] == y.shape[2] == 1, "Currently only supports scalar regression"
        ret = []
        for b in range(y.shape[0]):
            calibration_scores, scores = self.get_nonconformity_scores(self.calibration_preds, self.calibration_truths,
                                                                       pred[b], y[b])
            res = scores_to_intervals(pred[b, :, 0], y[b, :, 0], calibration_scores[:, :, 0], 1, alpha=alpha)
            ret.append(res.T.unsqueeze(-1))
            if update_cal:
                self.calibration_preds[self._update_cal_loc] = pred[b]
                self.calibration_truths[self._update_cal_loc] = y[b]
                self._update_cal_loc = (self._update_cal_loc + 1) % len(self.calibration_preds)
        ret = torch.stack(ret)
        return pred, ret

import torch

from .utils import _PI_Constructor, adapt_by_error_t, inverse_nonconformity_L1

class TQA_E(_PI_Constructor):
    def __init__(self, base_model_path=None, **kwargs):
        super(TQA_E, self).__init__(base_model_path, **kwargs)

    def predict(self, x, y, alpha=0.05, state=None, gamma=0.05, update_cal=False, two_sided=True, **kwargs):
        pred = self.base_model(x, state)
        assert pred.shape[2] == y.shape[2] == 1, "Currently only supports scalar regression"
        ret = []
        for b in range(y.shape[0]):
            calibration_scores, scores = self.get_nonconformity_scores(self.calibration_preds, self.calibration_truths,
                                                                       pred[b], y[b])

            ret.append(adapt_by_error_t(pred[b, :, 0], y[b, :, 0], calibration_scores[:, :, 0], alpha=alpha, gamma=gamma,
                                        scores=scores[:, 0], rev_func=inverse_nonconformity_L1, two_sided=two_sided).T.unsqueeze(-1))
            if update_cal:
                self.calibration_preds[self._update_cal_loc] = pred[b]
                self.calibration_truths[self._update_cal_loc] = y[b]
                self._update_cal_loc = (self._update_cal_loc + 1) % len(self.calibration_preds)
        ret = torch.stack(ret)
        return pred, ret

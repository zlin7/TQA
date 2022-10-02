import torch
import numpy as np

class CQRNN(torch.nn.Module):

    def __init__( self, base_model_path, alpha=0.1, **kwargs):
        super(CQRNN, self).__init__()
        self.qrnn = torch.load(base_model_path)
        self.alpha = alpha

        self._update_cal_loc = 0  # if we want to update the calibration residuals in an online fashion


    def calibrate(self, calibration_dataset: torch.utils.data.Dataset, batch_size=32, device=None):
        """
        Computes the nonconformity scores for the calibration dataset.
        """
        self.qrnn.eval()
        calibration_loader = torch.utils.data.DataLoader(calibration_dataset, batch_size=batch_size, shuffle=False)
        preds, ys = [], []
        with torch.no_grad():
            for calibration_example in calibration_loader:
                calibration_example = [_.to(device) for _ in calibration_example]
                sequences, targets, lengths_input, lengths_target = calibration_dataset._sep_data(calibration_example)
                out = self.qrnn(sequences)
                preds.append(out)
                ys.append(targets)
        self.calibration_preds = torch.nn.Parameter(torch.cat(preds), requires_grad=False)
        self.calibration_truths = torch.nn.Parameter(torch.cat(ys), requires_grad=False)

    def get_nonconformity_scores(self, cal_pred, cal_y, test_pred, test_y):
        cs = torch.maximum(cal_pred[:, 0] - cal_y, cal_y -  cal_pred[:, 1] )
        ts = torch.maximum(test_pred[0] - test_y, test_y -  test_pred[1] )
        return cs, ts


    def predict(self, x, y, alpha=None, state=None, update_cal=False, **kwargs):
        if alpha is None: alpha = self.alpha

        N = len(self.calibration_preds)
        q = np.ceil((N+1) * (1-alpha)) / N
        base_pi = self.qrnn(x) #(B, 2, L, 1)
        pred = base_pi.mean(1)
        assert pred.shape[2] == y.shape[2] == 1, "Currently only supports scalar regression"

        ret = torch.clone(base_pi)
        for b in range(y.shape[0]):
            calibration_scores, scores = self.get_nonconformity_scores(self.calibration_preds, self.calibration_truths, base_pi[b], y[b])
            width = torch.quantile(calibration_scores, q, dim=0)
            ret[b, 0] -= width
            ret[b, 1] += width
            if update_cal:
                self.calibration_preds[self._update_cal_loc] = base_pi[b]
                self.calibration_truths[self._update_cal_loc] = y[b]
                self._update_cal_loc = (self._update_cal_loc + 1) % len(self.calibration_preds)
        return pred, ret
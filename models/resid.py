import torch
import numpy as np

class LASplit(torch.nn.Module):
    def __init__( self, base_model_path, alpha=0.1, **kwargs):
        super(LASplit, self).__init__()
        self.madrnn = torch.load(base_model_path, map_location=kwargs.get('device'))
        self.alpha = alpha

        self._update_cal_loc = 0  # if we want to update the calibration residuals in an online fashion


    def calibrate(self, calibration_dataset: torch.utils.data.Dataset, batch_size=32, device=None):
        """
        Computes the nonconformity scores for the calibration dataset.
        """
        self.madrnn.eval()
        calibration_loader = torch.utils.data.DataLoader(calibration_dataset, batch_size=batch_size, shuffle=False)
        preds, ys = [], []
        with torch.no_grad():
            for calibration_example in calibration_loader:
                calibration_example = [_.to(device) for _ in calibration_example]
                sequences, targets, lengths_input, lengths_target = calibration_dataset._sep_data(calibration_example)
                out = self.madrnn(sequences, resid_only=False)
                preds.append(out)
                ys.append(targets)
        self.calibration_preds = torch.nn.Parameter(torch.cat(preds), requires_grad=False)
        self.calibration_truths = torch.nn.Parameter(torch.cat(ys), requires_grad=False)

        msk = self.calibration_preds[:, 0] > 0
        self._min_resid = self.calibration_preds[:, 0][msk].mean()
        self.calibration_preds[:, 0] = self.calibration_preds[:, 0].clip(self._min_resid)

    def get_nonconformity_scores(self, cal_pred, cal_y, test_pred, test_y):
        cs = (cal_pred[:, 1] - cal_y).abs() / cal_pred[:, 0]
        ts = (test_pred[1] - test_y).abs() / test_pred[0]
        return cs, ts


    def predict(self, x, y, alpha=None, state=None, update_cal=False, **kwargs):
        if alpha is None: alpha = self.alpha

        N = len(self.calibration_preds)
        q = np.ceil((N+1) * (1-alpha)) / N

        resid_yhat = self.madrnn(x, resid_only=False) #(B, 2, L, 1)
        pred = resid_yhat[:, 1]
        resid_yhat[:, 0] = resid_yhat[:, 0].clip(self._min_resid)
        assert pred.shape[2] == y.shape[2] == 1, "Currently only supports scalar regression"

        ret = torch.clone(resid_yhat)
        for b in range(y.shape[0]):
            calibration_scores, scores = self.get_nonconformity_scores(self.calibration_preds, self.calibration_truths, resid_yhat[b], y[b])
            width = torch.quantile(calibration_scores, q, dim=0) * resid_yhat[b, 0]
            ret[b, 0] = pred[b] - width
            ret[b, 1] = pred[b] + width
            if update_cal:
                self.calibration_preds[self._update_cal_loc] = resid_yhat[b]
                self.calibration_truths[self._update_cal_loc] = y[b]
                self._update_cal_loc = (self._update_cal_loc + 1) % len(self.calibration_preds)
        return pred, ret
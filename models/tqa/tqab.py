import torch

from .utils import _PI_Constructor, quantile_regression_EWMA, _torch_rank



class QuantileRegressionVariants:
    # return a range in [0, 1]
    @classmethod
    def rank_first(cls, pred, y, beta, **kwargs):
        base_scores = (pred - y).abs()
        base_rank = torch.argsort(torch.argsort(base_scores, 1),1) / (base_scores.shape[1] - 1)  - 0.5 #get the normalized ranking
        pred_rank = quantile_regression_EWMA(base_rank, beta=beta)
        return pred_rank + 0.5

    @classmethod
    def scale_first(cls, pred, y, beta, **kwargs):
        base_scores = (pred - y).abs()
        pred_aresid = quantile_regression_EWMA(base_scores, beta=beta)
        pred_rank = _torch_rank(pred_aresid, 1) - 0.5
        return pred_rank + 0.5

class BudgetingVariants:
    @classmethod
    def aggressive(cls, test_pred_rank, alpha, max_adj, N):
        assert alpha < 0.5, "The following logic might not make sense for alpha > 0.5 - will need to check later"
        test_pred_rank = test_pred_rank - 0.5 #[0,1] -> [-0.5, 0.5]
        q = (N+1) * (1-alpha) / N
        max_adj = (max_adj - q) / alpha
        qb = q + max_adj * alpha * (2 * test_pred_rank)
        return qb
    @classmethod
    def conservative(cls, test_pred_rank, alpha, max_adj, N):
        assert alpha < 0.5, "The following logic might not make sense for alpha > 0.5 - will need to check later"

        q = (N + 1) * (1 - alpha) / N
        max_adj = (max_adj - q) / alpha

        qb = torch.ones_like(test_pred_rank) * q
        delta = test_pred_rank - (1 - alpha)
        adj_up_msk = test_pred_rank > (1 - alpha)
        qb[~adj_up_msk] += ((alpha / (1 - alpha)) ** 2) * delta[~adj_up_msk] * max_adj
        qb[adj_up_msk] += delta[adj_up_msk] * max_adj

        qb[0] = q  # sanify check
        return qb


class TQA_B(_PI_Constructor):
    def __init__(self, base_model_path=None, **kwargs):
        super(TQA_B, self).__init__(base_model_path, **kwargs)

    def get_nonconformity_scores(self, cal_pred, cal_y, test_pred, test_y, beta=0.8):
        raise NotImplementedError()
    def get_adjusted_q(self, cal_pred, cal_y, test_pred, test_y, alpha,
                       beta=0.8, max_adj=0.99,
                       **kwargs
                       ):
        pred = torch.cat([cal_pred, test_pred.unsqueeze(0)], 0).squeeze(2).T
        y = torch.cat([cal_y, test_y.unsqueeze(0)], 0).squeeze(2).T
        # Using the scale-based prediction for rank
        test_pred_rank = QuantileRegressionVariants.scale_first(pred, y, beta=beta)[:, -1] # in [0, 1]
        # Use the conservative adjustment
        return BudgetingVariants.conservative(test_pred_rank, alpha, max_adj, len(cal_y))

    def predict(self, x, y, alpha=0.05, state=None, gamma=0, update_cal=False, censor=False, **kwargs):
        #assert  len(x.shape) == 3, "(batch, length, features)"
        device = y.device

        pred = self.base_model(x, state)
        B, L = pred.shape[0], pred.shape[1]
        assert pred.shape[2] == y.shape[2] == 1, "Currently only supports scalar regression"

        calibration_scores = (self.calibration_preds - self.calibration_truths).abs().sort(0)[0] #(n_cal, seq_len, 1)

        ret = torch.zeros(B, 2, L, device=device)

        qs = torch.zeros(B, L, device=device)
        if update_cal:

            for b in range(B):
                qs[b] = self.get_adjusted_q(self.calibration_preds, self.calibration_truths, pred[b], y[b], alpha=alpha, **kwargs)
                for t in range(L):
                    w = torch.quantile(calibration_scores[:, t, 0], qs[b, t])
                    ret[b, 0, t] = pred[b, t, 0] - w
                    ret[b, 1, t] = pred[b, t, 0] + w
                #update the residuals etc
                self.calibration_preds[self._update_cal_loc] = pred[b]
                self.calibration_truths[self._update_cal_loc] = y[b]
                self._update_cal_loc = (self._update_cal_loc + 1) % len(self.calibration_preds)
        else:
            qs = torch.zeros(B, L, device=device)
            for b in range(B):
                qs[b] = self.get_adjusted_q(self.calibration_preds, self.calibration_truths, pred[b], y[b], alpha=alpha, **kwargs)

            for t in range(L):
                w = torch.quantile(calibration_scores[:, t, 0], qs[:, t])
                ret[:, 0, t] = pred[:, t, 0] - w
                ret[:, 1, t] = pred[:, t, 0] + w
        ret = ret.unsqueeze(3)
        return pred, ret#, qs.cpu().numpy()


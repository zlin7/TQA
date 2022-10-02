import torch
from .tqab import TQA_B, QuantileRegressionVariants, BudgetingVariants


class TQA_B_AR(TQA_B):
    def __init__(self, base_model_path=None, **kwargs):
        super(TQA_B_AR, self).__init__(base_model_path, **kwargs)

    def get_adjusted_q(self, cal_pred, cal_y, test_pred, test_y, alpha,
                       beta=0.8, max_adj=0.99,
                       **kwargs
                       ):
        pred = torch.cat([cal_pred, test_pred.unsqueeze(0)], 0).squeeze(2).T
        y = torch.cat([cal_y, test_y.unsqueeze(0)], 0).squeeze(2).T
        # Using the rank-based prediction for rank
        test_pred_rank = QuantileRegressionVariants.rank_first(pred, y, beta=beta)[:, -1] # in [0, 1]
        # Use the aggressive adjustment
        return BudgetingVariants.aggressive(test_pred_rank, alpha, max_adj, len(cal_y))

class TQA_B_AS(TQA_B):
    def __init__(self, base_model_path=None, **kwargs):
        super(TQA_B_AS, self).__init__(base_model_path, **kwargs)

    def get_adjusted_q(self, cal_pred, cal_y, test_pred, test_y, alpha,
                       beta=0.8, max_adj=0.99,
                       **kwargs
                       ):
        pred = torch.cat([cal_pred, test_pred.unsqueeze(0)], 0).squeeze(2).T
        y = torch.cat([cal_y, test_y.unsqueeze(0)], 0).squeeze(2).T
        # Using the scale-based prediction for rank
        test_pred_rank = QuantileRegressionVariants.scale_first(pred, y, beta=beta)[:, -1] # in [0, 1]
        # Use the aggressive adjustment
        return BudgetingVariants.aggressive(test_pred_rank, alpha, max_adj, len(cal_y))


class TQA_B_CR(TQA_B_AR):
    def __init__(self, base_model_path=None, **kwargs):
        super(TQA_B_AR, self).__init__(base_model_path, **kwargs)

    def get_adjusted_q(self, cal_pred, cal_y, test_pred, test_y, alpha,
                       beta=0.8, max_adj=0.99,
                       **kwargs
                       ):
        pred = torch.cat([cal_pred, test_pred.unsqueeze(0)], 0).squeeze(2).T
        y = torch.cat([cal_y, test_y.unsqueeze(0)], 0).squeeze(2).T
        # Using the rank-based prediction for rank
        test_pred_rank = QuantileRegressionVariants.rank_first(pred, y, beta=beta)[:, -1] # in [0, 1]
        # Use the conservative adjustment
        return BudgetingVariants.conservative(test_pred_rank, alpha, max_adj, len(cal_y))

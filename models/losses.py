import torch



def quantile_loss(output, target, masks, alpha):
    single_loss = masks * quantile_noreduction_loss(output, target, alpha)
    loss = torch.mean(torch.sum(single_loss, dim=1) / torch.sum(masks, dim=1))

    return loss

def quantile_noreduction_loss(output, target,alpha):
    return ((output - target) * (output >= target) * alpha + (target - output) * (output < target) * (1 - alpha))

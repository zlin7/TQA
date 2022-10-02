import os.path

import ipdb
import torch
import tqdm

from models.training_utils import fit_mean, fit_resid, get_lengths_mask


class PointEstimator(torch.nn.Module):
    def __init__(self):
        super(PointEstimator, self).__init__()
        self.path = None

    def forward(self, x, state=None, use_horizon=True):
        raise NotImplementedError()


    def fit(self, train_dataset, batch_size, epochs, lr, val_dataset=None, device='cuda:0', **kwargs):
        print(f"Unused fitkwargs={kwargs}")
        return fit_mean(self, train_dataset, batch_size, epochs, lr, val_dataset, device)

class ResidEstimator(torch.nn.Module):
    def __init__(self, base_model_path):
        super(ResidEstimator, self).__init__()
        self.base_model = torch.load(base_model_path).eval()
        for p in self.base_model.parameters():
            p.requires_grad = False #This is important - otherwise it could be degenerate
        self.path = None
    def forward(self, x, state=None, use_horizon=True):
        raise NotImplementedError()


    def fit(self, train_dataset, batch_size, epochs, lr, val_dataset=None, device='cuda:0', **kwargs):
        print(f"Unused fitkwargs={kwargs}")
        return fit_resid(self, train_dataset, batch_size, epochs, lr, val_dataset, device)


class QuantileEstimator(torch.nn.Module):
    def __init__(self, coverage=0.9):
        super(QuantileEstimator, self).__init__()
        self.tail_alpha = (1 - coverage) / 2.
        self.path = None

    def forward(self, x, state=None, use_horizon=True):
        raise NotImplementedError()

    def fit(self, train_dataset, batch_size, epochs, lr,
            device='cuda:0', **kwargs):
        from models.losses import quantile_loss
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.train()
        criterion = lambda o, y, msk: quantile_loss(o[:, 0], y, msk, 1-self.tail_alpha) + quantile_loss(o[:, 1], y, msk, self.tail_alpha)
        for epoch in tqdm.tqdm(range(epochs), desc='Training..'):
            train_loss = 0.0

            for sequences, targets, lengths_input, lengths_target in train_loader:

                optimizer.zero_grad()
                valid_sequences = sequences * get_lengths_mask(sequences, lengths_input)
                out = self(valid_sequences.to(device))
                valid_out = out * get_lengths_mask(out, lengths_target, None)

                targets = targets.to(device)
                msk = get_lengths_mask(targets, lengths_target)
                loss = criterion(valid_out, targets, msk)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            if epoch % (epochs // 20) == 0:
                print("Epoch: ", epoch, "| train loss: %.4f" % (train_loss / len(train_loader)))

        if self.path is not None:
            torch.save(self, self.path)
        return

class QuantileLinearRegression(QuantileEstimator):
    def __init__(self, input_size=1, output_size=1, horizon=1, path=None, coverage=0.9):
        super(QuantileLinearRegression, self).__init__(coverage)
        assert output_size == 1
        self.input_size = input_size
        self.horizon = horizon
        self.output_size = output_size
        self.path = path

        #Should have (input_size * horizon) by output_size as the weights..
        self.lins = torch.nn.ModuleList([torch.nn.Linear(in_features=(t+1) * input_size, out_features=output_size * 2) for t in range(horizon)])

    def forward(self, x, state=None, use_horizon=True):
        B = x.shape[0]
        qhat = []
        for t in range(self.horizon):
            xt = x[:, :t+1].view(B, -1)
            qhat.append(self.lins[t](xt))
        qhat = torch.stack(qhat, 1).unsqueeze(-1).permute(0,2,1,3)
        return qhat


class ResidLinearRegression(ResidEstimator):
    def __init__(self, base_model_path, path=None):
        super(ResidLinearRegression, self).__init__(base_model_path)

        input_size = self.input_size = self.base_model.input_size
        horizon = self.horizon = self.base_model.horizon
        output_size = self.output_size = self.base_model.output_size
        self.path = path

        #Should have (input_size * horizon) by output_size as the weights..
        self.lins = torch.nn.ModuleList([torch.nn.Linear(in_features=(t+1) * input_size, out_features=output_size) for t in range(horizon)])

    def forward(self, x, state=None, use_horizon=True, resid_only=False):
        B = x.shape[0]
        resid_hat = []
        for t in range(self.horizon):
            xt = x[:, :t+1].view(B, -1)
            resid_hat.append(self.lins[t](xt))
        resid_hat = torch.stack(resid_hat, 1)
        if resid_only: return resid_hat
        yhat = self.base_model(x, state, use_horizon)
        return torch.stack([resid_hat, yhat], 1)

class LinearRegression(PointEstimator):
    def __init__(self, input_size=1, output_size=1, horizon=1, path=None):
        super(LinearRegression, self).__init__()
        # input_size indicates the number of features in the time series
        # input_size=1 for univariate series.
        self.input_size = input_size
        self.horizon = horizon
        self.output_size = output_size
        self.path = path

        #Should have (input_size * horizon) by output_size as the weights..
        self.lins = torch.nn.ModuleList([torch.nn.Linear(in_features=(t+1) * input_size, out_features=output_size) for t in range(horizon)])

    def forward(self, x, state=None, use_horizon=True):
        B = x.shape[0]
        yhat = []
        for t in range(self.horizon):
            xt = x[:, :t+1].view(B, -1)
            yhat.append(self.lins[t](xt))
        return torch.stack(yhat, 1)

import torch

from models.training_utils import fit_mean, fit_resid, get_lengths_mask


class MyRNN(torch.nn.Module):
    def __init__(self, embedding_size, input_size=1, output_size=1, horizon=1, rnn_mode="LSTM", path=None):
        super(MyRNN, self).__init__()
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.horizon = horizon
        self.output_size = output_size
        self.path = path

        self.rnn_mode = rnn_mode
        assert self.rnn_mode == 'LSTM'
        self.forecaster_rnn = torch.nn.LSTM(input_size=input_size, hidden_size=embedding_size, batch_first=True)
        self.forecaster_out = torch.nn.Linear(embedding_size, output_size)

    def forward(self, x, state=None, use_horizon=True):
        # [batch, horizon, output_size]
        o_n, (h_n, c_n) = self.forecaster_rnn(x.float(), state)
        if isinstance(use_horizon, bool):
            if use_horizon:
                out = self.forecaster_out(o_n)[:, -self.horizon:, :]
            else:
                out = self.forecaster_out(o_n)[:, :, :]
        else:
            assert isinstance(use_horizon, int)
            out = self.forecaster_out(o_n)[:, -use_horizon:, :]
        return out

    def fit(self, train_dataset, batch_size, epochs, lr, val_dataset=None, device='cuda:0', **kwargs):
        print(f"Unused fitkwargs={kwargs}")
        return fit_mean(self, train_dataset, batch_size, epochs, lr, val_dataset, device)


class ResidRNN(torch.nn.Module):

    def __init__(self, base_rnn_path, path=None):
        super(ResidRNN, self).__init__()
        self.rnn = torch.load(base_rnn_path).eval()
        for p in self.rnn.parameters():
            p.requires_grad = False  # This is important - otherwise it could be degenerate

        input_size = self.input_size = self.rnn.input_size
        embedding_size = self.embedding_size = self.rnn.embedding_size
        horizon = self.horizon = self.rnn.horizon
        output_size = self.output_size = self.rnn.output_size
        self.path = path

        self.rnn_mode = self.rnn.rnn_mode
        assert self.rnn_mode == 'LSTM'
        self.forecaster_rnn = torch.nn.LSTM(input_size=input_size, hidden_size=embedding_size, batch_first=True)
        self.forecaster_out = torch.nn.Linear(embedding_size, output_size)

    def forward(self, x, state=None, use_horizon=True, resid_only=False):
        if state is not None:
            h_0, c_0 = state
        else:
            h_0 = None
        o_n, (h_n, c_n) = self.forecaster_rnn(x.float(), state)
        if isinstance(use_horizon, bool):
            if use_horizon:
                out = self.forecaster_out(o_n)[:, -self.horizon:, :]
            else:
                out = self.forecaster_out(o_n)[:, :, :]
        else:
            assert isinstance(use_horizon, int)
            out = self.forecaster_out(o_n)[:, -use_horizon:, :]
        if resid_only: return out
        yhat = self.rnn(x, state, use_horizon)
        out = torch.stack([out, yhat], 1)
        return out

    def fit(self, train_dataset, batch_size, epochs, lr, val_dataset=None, device='cuda:0', **kwargs):
        print(f"Unused fitkwargs={kwargs}")
        return fit_resid(self, train_dataset, batch_size, epochs, lr, val_dataset, device)
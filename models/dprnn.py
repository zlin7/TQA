# Adapted from https://github.com/kamilest/conformal-rnn/blob/master/models/dprnn.py
import torch
from scipy import stats as st
from torch import nn

import tqdm
from models.training_utils import get_lengths_mask


class DPRNN(nn.Module):
    def __init__(
        self,
        rnn_mode="LSTM",
        epochs=5,
        batch_size=150,
        max_steps=50,
        input_size=1,
        lr=0.01,
        output_size=1,
        embedding_size=20,
        n_layers=1,
        coverage=0.9,
        dropout_prob=0.5,
        **kwargs
    ):

        super(DPRNN, self).__init__()

        self.EPOCH = epochs
        self.BATCH_SIZE = batch_size
        self.MAX_STEPS = max_steps
        self.INPUT_SIZE = input_size
        self.LR = lr
        self.OUTPUT_SIZE = output_size
        self.HIDDEN_UNITS = embedding_size
        self.NUM_LAYERS = n_layers

        self.rnn_mode = rnn_mode
        assert self.rnn_mode == 'LSTM'
        self.dropout_prob = dropout_prob
        self.dropout = nn.Dropout(p=dropout_prob)

        self.rnn = nn.LSTM(
                input_size=self.INPUT_SIZE,
                hidden_size=self.HIDDEN_UNITS,
                num_layers=self.NUM_LAYERS,
                batch_first=True,
                dropout=self.dropout_prob,
            )
        self.out = nn.Linear(self.HIDDEN_UNITS, self.OUTPUT_SIZE)

    def forward(self, x):
        self.train() #So we always sample. If we only have 1 layer in LSTM this does not make any difference though.

        r_out, (h_n, h_c) = self.rnn(x, None)  # None represents zero

        # choose r_out at the last time step
        out = self.out(self.dropout(r_out))
        return out

    def fit(self, train_dataset, batch_size, device='cuda:0', **kwargs):

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.parameters(), lr=self.LR)  # optimize all rnn parameters
        self.train()
        criterion = torch.nn.MSELoss(reduction='none')
        for epoch in tqdm.tqdm(range(self.EPOCH), desc='Training..'):
            train_loss = 0.0


            for sequences, targets, lengths_input, lengths_target in train_loader:

                optimizer.zero_grad()
                valid_sequences = sequences * get_lengths_mask(sequences, lengths_input)
                out = self(valid_sequences.to(device))
                valid_out = out * get_lengths_mask(out, lengths_target, None)

                targets = targets.to(device)
                valid_targets = targets * get_lengths_mask(targets, lengths_target)
                loss = criterion(valid_out, valid_targets).mean()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            if epoch % (self.EPOCH // 20) == 0:
                print("Epoch: ", epoch, "| train loss: %.4f" % (train_loss / len(train_loader)))
        return

    def predict(self, X, y=None, num_samples=100, alpha=0.05, **kwargs):
        z_critical = st.norm.ppf((1 - alpha) + (alpha) / 2)

        predictions = []

        for idx in range(num_samples):
            predicts_ = self(X)
            predictions.append(predicts_.detach())
        predictions = torch.stack(predictions, 0)
        pred_mean = predictions.mean(0)
        pred_std = predictions.std(0)
        pi = torch.stack([pred_mean - pred_std * z_critical , pred_mean + pred_std * z_critical], 1)
        return pred_mean, pi

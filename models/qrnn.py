# Adapted from https://github.com/kamilest/conformal-rnn/blob/master/models/qrnn.py
import torch

from models.losses import quantile_loss
from models.training_utils import get_lengths_mask
import tqdm

torch.manual_seed(1)


class QRNN(torch.nn.Module):
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
        **kwargs
    ):

        super(QRNN, self).__init__()

        self.EPOCH = epochs
        self.BATCH_SIZE = batch_size
        self.MAX_STEPS = max_steps
        self.INPUT_SIZE = input_size
        self.LR = lr
        self.OUTPUT_SIZE = output_size
        self.HIDDEN_UNITS = embedding_size
        self.NUM_LAYERS = n_layers
        self.tail_alpha = (1 - coverage) / 2.
        self.rnn_mode = rnn_mode

        rnn_dict = {
            "RNN": torch.nn.RNN(
                input_size=self.INPUT_SIZE, hidden_size=self.HIDDEN_UNITS, num_layers=self.NUM_LAYERS, batch_first=True,
            ),
            "LSTM": torch.nn.LSTM(
                input_size=self.INPUT_SIZE, hidden_size=self.HIDDEN_UNITS, num_layers=self.NUM_LAYERS, batch_first=True,
            ),
            "GRU": torch.nn.GRU(
                input_size=self.INPUT_SIZE, hidden_size=self.HIDDEN_UNITS, num_layers=self.NUM_LAYERS, batch_first=True,
            ),
        }

        self.rnn = rnn_dict[self.rnn_mode]
        self.out = torch.nn.Linear(self.HIDDEN_UNITS, 2 * self.OUTPUT_SIZE)

    def forward(self, x, state=None):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)

        if self.rnn_mode == "LSTM":
            r_out, (h_n, h_c) = self.rnn(x, state)  # None represents zero
            # initial hidden state
        else:
            r_out, h_n = self.rnn(x, state)
        # choose r_out at the last time step
        out = self.out(r_out)

        return out.unsqueeze(-1).permute(0,2,1,3)

    def fit(self, train_dataset, batch_size, device='cuda:0', **kwargs):
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.parameters(), lr=self.LR)  # optimize all rnn parameters
        self.train()
        criterion = lambda o, y, msk: quantile_loss(o[:, 0], y, msk, 1-self.tail_alpha) + quantile_loss(o[:, 1], y, msk, self.tail_alpha)
        for epoch in tqdm.tqdm(range(self.EPOCH), desc='Training..'):
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
            if epoch % (self.EPOCH // 20) == 0:
                print("Epoch: ", epoch, "| train loss: %.4f" % (train_loss / len(train_loader)))
        return

    def predict(self, X, y=None, alpha=None, **kwargs):
        if alpha is None: alpha = self.tail_alpha * 2
        assert abs(alpha - self.tail_alpha * 2) < 1e-5
        pi =  self(X) #->(Batch, 2, L,  1)
        return pi.mean(1), pi


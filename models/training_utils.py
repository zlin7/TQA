import torch
import tqdm

def get_lengths_mask(sequences, lengths, horizon=None):
    lengths_mask = torch.zeros_like(sequences, device=sequences.device)
    for i, l in enumerate(lengths):
        if horizon is not None: l = min(l, horizon)
        lengths_mask[i, -l:, :] = 1
    return lengths_mask

def fit_mean(self, train_dataset, batch_size, epochs, lr, val_dataset=None, device='cuda:0'):
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(self.parameters(), lr=lr)
    criterion = torch.nn.MSELoss(reduction='none')


    self.train()
    for epoch in tqdm.tqdm(range(epochs), desc='Training..'):
        train_loss = 0.0

        for sequences, targets, lengths_input, lengths_target in train_loader:
            optimizer.zero_grad()
            valid_sequences = sequences * get_lengths_mask(sequences, lengths_input)
            out = self(valid_sequences.to(device), use_horizon=targets.shape[1]) #predict the part that we have labels for
            valid_out = out * get_lengths_mask(out, lengths_target, None) #Remove horizon in training

            targets = targets.to(device)
            valid_targets = targets * get_lengths_mask(targets, lengths_target)
            loss_full = criterion(valid_out, valid_targets)
            loss = loss_full.mean()
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        mean_train_loss = train_loss / len(train_loader)
        if epoch % (epochs // 20) == 0:
            print("Epoch: {}\tTrain loss: {}".format(epoch, mean_train_loss))
            if val_dataset is not None:
                self.eval()
                val_loss = 0.
                val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
                with torch.no_grad():
                    for sequences, targets, lengths_input, lengths_target in val_loader:
                        valid_sequences = sequences * get_lengths_mask(sequences, lengths_input)
                        out = self(valid_sequences.to(device), use_horizon=targets.shape[1])
                        valid_out = out * get_lengths_mask(out, lengths_target, None)  # Remove horizon in training
                        targets = targets.to(device)
                        valid_targets = targets * get_lengths_mask(targets, lengths_target)
                        loss_full = criterion(valid_out, valid_targets)
                        val_loss += loss_full.mean().item()
                    print(f"Epoch: {epoch}\t Valid Loss: {val_loss / len(val_loader)}")
                self.train()


    if self.path is not None:
        torch.save(self, self.path)

def fit_resid(self, train_dataset, batch_size, epochs, lr, val_dataset=None, device='cuda:0'):
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(self.parameters(), lr=lr)
    criterion = torch.nn.MSELoss(reduction='none')

    self.train()
    for epoch in tqdm.tqdm(range(epochs), desc='Training..'):
        train_loss = 0.0

        for sequences, targets, lengths_input, lengths_target in train_loader:
            optimizer.zero_grad()
            valid_sequences = sequences * get_lengths_mask(sequences, lengths_input)
            out = self(valid_sequences.to(device), use_horizon=targets.shape[1]) #predict the part that we have labels for
            valid_out = out * get_lengths_mask(out, lengths_target, None) #Remove horizon in training
            targets = (targets.to(device) - valid_out[:, 1]).abs()
            valid_out = valid_out[:, 0]

            valid_targets = targets * get_lengths_mask(targets, lengths_target)
            loss_full = criterion(valid_out, valid_targets)

            loss = loss_full.mean()
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        mean_train_loss = train_loss / len(train_loader)
        if epoch % (epochs // 20) == 0:
            print("Epoch: {}\tTrain loss: {}".format(epoch, mean_train_loss))
            if val_dataset is not None:
                self.eval()
                val_loss = 0.
                val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
                with torch.no_grad():
                    for sequences, targets, lengths_input, lengths_target in val_loader:
                        valid_sequences = sequences * get_lengths_mask(sequences, lengths_input)
                        out = self(valid_sequences.to(device), use_horizon=targets.shape[1])
                        valid_out = out * get_lengths_mask(out, lengths_target, None)  # Remove horizon in training

                        targets = (targets.to(device) - valid_out[:, 1]).abs()
                        valid_out = valid_out[:, 0]

                        valid_targets = targets * get_lengths_mask(targets, lengths_target)
                        loss_full = criterion(valid_out, valid_targets)
                        val_loss += loss_full.mean().item()
                    print(f"Epoch: {epoch}\t Valid Loss: {val_loss / len(val_loader)}")
                self.train()

    if self.path is not None:
        torch.save(self, self.path)

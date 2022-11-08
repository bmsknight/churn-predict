import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

import src.constants as const


class ChurnDataset(Dataset):

    def __init__(self, x, y):
        self.x = torch.tensor(np.array(x), dtype=torch.float32)
        y = torch.tensor(np.array(y), dtype=torch.float32)
        self.y = y.unsqueeze(-1)
        self.length = self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return self.length


class SimpleFeedForwardNetwork(nn.Module):

    def __init__(self, in_features):
        super(SimpleFeedForwardNetwork, self).__init__()
        self.input_layer = nn.Linear(in_features=in_features, out_features=32)
        self.hidden_layer_1 = nn.Linear(in_features=32, out_features=64)
        self.hidden_layer_2 = nn.Linear(in_features=64, out_features=32)
        self.output_layer = nn.Linear(in_features=32, out_features=1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        output = self.input_layer(x)
        output = self.relu(output)
        output = self.hidden_layer_1(output)
        output = self.relu(output)
        output = self.hidden_layer_2(output)
        output = self.relu(output)
        output = self.output_layer(output)
        output = self.sigmoid(output)
        return output


class Learner:
    def __init__(self, network, loss, optim, num_epochs, device,
                 model_save_path=const.TEMPORARY_MODEL_SAVE_PATH):

        self.network = network.to(device)
        self.loss = loss
        self.optimizer = optim
        self.num_epochs = num_epochs
        self.device = device

        self.best_loss = float('inf')
        self.model_save_path = model_save_path

    def _train_step(self, data_loader):
        self.network.train()
        train_loss = 0
        for x, y in data_loader:
            x = x.to(self.device)
            y = y.to(self.device)

            prediction = self.network(x)
            loss = self.loss(prediction, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item() * x.shape[0]
        return train_loss / len(data_loader.dataset)

    def _validation_step(self, data_loader):
        self.network.eval()
        with torch.no_grad():
            val_loss = 0
            for x, y in data_loader:
                x = x.to(self.device)
                y = y.to(self.device)

                prediction = self.network(x)
                loss = self.loss(prediction, y)
                val_loss += loss.item() * x.shape[0]
        return val_loss / len(data_loader.dataset)

    def train(self, train_loader, val_loader=None):
        history = {}
        for epoch in range(self.num_epochs):
            train_loss = self._train_step(train_loader)
            loss_log = {"epoch": epoch, "train_loss": train_loss}
            if val_loader:
                val_loss = self._validation_step(val_loader)
                print(f"Epoch: {epoch}, \t Training loss: {train_loss}, \t Val loss: {val_loss}")
                loss_log["val_loss"] = val_loss
                self._save_best_model(val_loss=val_loss)
            else:
                print(f"Epoch: {epoch}, \t Training loss: {train_loss}")
            history[epoch] = loss_log
        return history

    def predict(self, test_loader):
        full_predictions = torch.empty(0)
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(self.device)
                prediction = self.network(x)
                prediction = prediction.to("cpu")
                full_predictions = torch.cat([full_predictions, prediction], dim=0)
        full_predictions = (full_predictions > 0.5).float()
        return full_predictions

    def _save_best_model(self, val_loss):
        if val_loss < self.best_loss:
            torch.save(self.network.state_dict(), self.model_save_path)
            print("Saving best model")
            self.best_loss = val_loss

    def load_best_model(self):
        self.network.load_state_dict(torch.load(self.model_save_path))
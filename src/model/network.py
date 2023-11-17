import torch
import torch.nn as nn


class Network(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=config.model.input_size,
            hidden_size=config.model.hidden_size,
            num_layers=config.model.num_layers,
            dropout=config.model.dropout,
            batch_first=True
        )
        self.classifier = nn.Linear(config.model.hidden_size, config.model.n_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.lstm.flatten_parameters()
        _, (hidden, _) = self.lstm(x)
        out = hidden[-1]
        out = self.classifier(out)
        out = self.sigmoid(out)
        return out

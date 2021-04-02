import torch.nn as nn
import torch
from copy import deepcopy


class IMUCLSBaseline(nn.Module):
    def __init__(self, config):

        super(IMUCLSBaseline, self).__init__()

        input_dim = config.get("input_dim")
        feature_dim = config.get("transformer_dim")
        window_size = config.get("window_size")

        self.conv1 = nn.Sequential(nn.Conv1d(input_dim, feature_dim, kernel_size=1), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv1d(feature_dim, feature_dim, kernel_size=1), nn.ReLU())

        self.dropout = nn.Dropout(config.get("baseline_dropout"))
        self.maxpool = nn.MaxPool1d(2) # Collapse T time steps to T/2
        self.fc1 = nn.Linear(window_size*(feature_dim//2), feature_dim, nn.ReLU())
        self.fc2 = nn.Linear(feature_dim,  config.get("num_classes"))
        self.log_softmax = nn.LogSoftmax(dim=1)

        # init
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, data):
        """
        Forward pass
        :param x:  B X M x T tensor reprensting a batch of size B of  M sensors (measurements) X T time steps (e.g. 128 x 6 x 100)
        :return: B X N weight for each mode per sample
        """
        x = data.get('imu').transpose(1, 2)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.dropout(x)
        x = self.maxpool(x) # return B X C/2 x M
        x = x.view(x.size(0), -1) # B X C/2*M
        x = self.fc1(x)
        x = self.log_softmax(self.fc2(x))
        return x # B X N

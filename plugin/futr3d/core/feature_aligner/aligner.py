import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureAligner(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=128, output_dim=128, mode="train"):
        super(FeatureAligner, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.fc3 = nn.Linear(output_dim, output_dim)
        self.fc4 = nn.Linear(output_dim, output_dim)
        self.fc5 = nn.Linear(hidden_dim, output_dim)
        self.fc6 = nn.Linear(output_dim, output_dim)
        self.fc7 = nn.Linear(output_dim, output_dim)
        self.fc8 = nn.Linear(output_dim, output_dim)

        assert mode in ["train", "test"]

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = self.fc8(x)
        return x



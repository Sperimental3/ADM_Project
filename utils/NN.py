# definition of the two types of networks used

from torch import nn


class Strong(nn.Module):
    def __init__(self, input_channel_dimension, output_dimension):
        super().__init__()

        self.input_channel_dimension = input_channel_dimension
        self.output_dimension = output_dimension

        # Structure of the net
        self.net = nn.Sequential(nn.Conv2d(self.input_channel_dimension, 32, (3, 3), stride=1, padding=1), nn.ReLU(),
                                 nn.Conv2d(32, 32, (3, 3), stride=1, padding=1), nn.ReLU(),
                                 nn.MaxPool2d((2, 2), stride=2),
                                 nn.Conv2d(32, 32, (3, 3), stride=1, padding=1), nn.ReLU(),
                                 nn.Conv2d(32, 32, (3, 3), stride=1, padding=1), nn.ReLU(),
                                 nn.MaxPool2d((2, 2), stride=2),
                                 nn.Conv2d(32, 64, (3, 3), stride=1, padding=1), nn.ReLU(),
                                 nn.Conv2d(64, 64, (3, 3), stride=1, padding=1), nn.ReLU(),
                                 nn.MaxPool2d((2, 2), stride=2),
                                 nn.Conv2d(64, 128, (3, 3), stride=1, padding=1), nn.ReLU(),
                                 nn.Conv2d(128, 128, (3, 3), stride=1, padding=1), nn.ReLU(),
                                 nn.MaxPool2d((2, 2), stride=2),
                                 nn.Flatten(), nn.Linear(128 * 6 * 6, self.output_dimension)
                                 )

    def forward(self, x):
        m = nn.Sigmoid()

        out = m(self.net(x))

        return out


class Weak(nn.Module):
    def __init__(self, input_dimension, output_dimension):
        super().__init__()

        self.input_dimension = input_dimension
        self.output_dimension = output_dimension

        # Structure of the net
        self.net = nn.Sequential(nn.Flatten(),
                                 nn.Linear(self.input_dimension, 32), nn.ReLU(),
                                 nn.Linear(32, 32), nn.ReLU(),
                                 nn.Linear(32, self.output_dimension)
                                 )

    def forward(self, x):
        m = nn.Sigmoid()

        out = m(self.net(x))

        return out

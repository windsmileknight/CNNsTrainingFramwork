import torch.nn as nn


class LeNet5(nn.Module):
    def __init__(self,
                 in_channel,
                 width,
                 height,
                 class_num,
                 dropout):
        """
        input image size 32 × 32
        :param in_channel: input channel (RGB: 3).
        :param width: width of image
        :param height: height of image
        :param class_num: class number
        """
        super(LeNet5, self).__init__()
        self.dropout = dropout

        self.C1 = nn.Sequential(
            nn.Conv2d(in_channel, 6, kernel_size=(5, 5), stride=1),  # (32 × 32) --> (28 × 28)
            nn.ReLU()
        )
        self.S2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),  # (28 × 28) --> (14 × 14)
        )
        self.C3 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=(5, 5), stride=1),  # (14 × 14) --> (10 × 10)
            nn.ReLU()
        )
        self.S4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),  # (10 × 10) --> (5 × 5)
        )
        self.C5 = nn.Sequential(
            nn.Conv2d(16, 120, kernel_size=(5, 5), stride=1),  # (5 × 5) --> (1 × 1)
            nn.ReLU()
        )
        self.F6 = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, class_num)
        )

        self.LeNet5 = nn.Sequential(
            self.C1,
            self.S2,
            nn.Dropout(self.dropout),
            self.C3,
            self.S4,
            nn.Dropout(self.dropout),
            self.C5,
            self.F6
        )

    def forward(self, x):
        """
        :param x: :param x: (batch, in_channel, height, weight)
        """
        return self.LeNet5(x)

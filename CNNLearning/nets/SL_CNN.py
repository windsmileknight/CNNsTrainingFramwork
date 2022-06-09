import torch.nn as nn
import math


class SL_CNN(nn.Module):
    """
    Single layer - CNN
    """
    def __init__(self,
                 in_channel,
                 width,
                 height,
                 class_num
                 ):
        super(SL_CNN, self).__init__()
        self.in_channel = in_channel

        self.cnn = nn.Sequential(
            nn.Conv2d(self.in_channel,
                      1,
                      kernel_size=3,
                      stride=1
                      ),  # [batch, out_channel, height, weight]
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=3, stride=3),
            nn.Flatten(start_dim=1),  # [batch, weight * height * out_channel]
            nn.Linear((width - 2) * (height - 2) * 1, class_num)
        )

    def forward(self, x):
        """
        :param x: (batch, in_channel, height, weight)
        """
        return self.cnn(x)  # (batch, output)

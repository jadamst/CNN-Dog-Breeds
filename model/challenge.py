"""
EECS 445 - Introduction to Machine Learning
Winter 2021 - Project 2
Challenge
    Constructs a pytorch model for a convolutional neural network
    Usage: from model.challenge import Challenge
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt


class Challenge(nn.Module):
    def __init__(self):
        super().__init__()

        ## TODO: define each layer
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size = 3, stride = 2, padding = 2)
        self.pool = nn.AvgPool2d(kernel_size = 2, stride = 2)
        self.conv2 = nn.Conv2d(in_channels = 16, out_channels = 64, kernel_size = 3, stride = 2, padding = 2)
        self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 8, kernel_size = 3, stride = 2, padding = 2)
        self.fc_1 = nn.Linear(in_features = 32, out_features = 2)
        ##filter size 3x3
        ## more fc 
        ## add pooling, avg?

        self.init_weights()

    def init_weights(self):
        for conv in [self.conv1, self.conv2, self.conv3]:
            C_in = conv.weight.size(1)
            nn.init.normal_(conv.weight, 0.0, 1 / sqrt(5 * 5 * C_in))
            nn.init.constant_(conv.bias, 0.0)

        ## TODO: initialize the parameters for [self.fc1]
        nn.init.normal_(self.fc_1.weight, 0.0, 1 / sqrt(32))
        nn.init.constant_(self.fc_1.bias, 0.0) 
        ##

    def forward(self, x):
        N, C, H, W = x.shape

        ## TODO: forward pass
        x = (F.relu(self.conv1(x)))
        #print(x.shape)
        x = self.pool(x)
        #print(x.shape)
        x = (F.relu(self.conv2(x)))
        #print(x.shape)
        x = self.pool(x)
        x = (F.relu(self.conv3(x)))
        #print(x.shape)
        x = x.view(-1, 32)
        #x.size()
        x = self.fc_1(x)
        ##


        return x

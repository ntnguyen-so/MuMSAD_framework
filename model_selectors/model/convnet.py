import torch
import torch.nn as nn

import numpy as np


# Model used for CNN baseline
class ConvNet(nn.Module):
    def __init__(
        self,
        original_length,
        num_blocks=5,
        kernel_size=3,
        padding=1,
        original_dim=1,
        num_classes=12
    ):
        super(ConvNet, self).__init__()                
        
        self.num_class = num_classes
        self.kernel_size = kernel_size
        self.padding = padding
        # self.dilation = dilation
        # self.strides = strides
        self.layers = []

        dims = [original_dim]
        dims += list(2 ** np.arange(6, 6 + num_blocks))
        dims = [x if x <= 256 else 256 for x in dims]

        for i in range(num_blocks):
            self.layers.extend([
                nn.Conv1d(dims[i], dims[i+1], kernel_size=self.kernel_size, padding=self.padding),
                nn.BatchNorm1d(dims[i+1]),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
        self.layers.extend([
            nn.Conv1d(dims[-1], dims[-1], kernel_size=self.kernel_size, padding=self.padding),
            nn.ReLU(),
        ])
        self.layers = nn.Sequential(*self.layers)
                
        self.GAP = nn.AvgPool1d(kernel_size=original_length//original_dim, stride=original_length//original_dim, padding=0) #(original_length)
        # self.GAP = nn.AvgPool1d(original_length//2, stride=16, padding=0)
        
        self.fc1 = nn.Sequential(
            nn.Linear(dims[-1], num_classes),
            nn.LogSoftmax(dim=1) 
        )

        
    def forward(self, x):
        out = self.layers(x)
        
        out = self.GAP(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        
        return out

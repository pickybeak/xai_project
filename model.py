import torch
import torch.nn as nn

class ZbCNN(nn.Module):
    def __init__(self,
                 hidden_channels=[64, 128, 256],
                 maxpool_size=2,
                 kernel_size=5,
                 categories=7):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Conv2d(1, hidden_channels[0], kernel_size, bias=False),
            nn.ReLU(),
            nn.MaxPool2d((maxpool_size,maxpool_size)),
            nn.Conv2d(hidden_channels[0], hidden_channels[1], kernel_size, bias=False),
            nn.ReLU(),
            nn.MaxPool2d((maxpool_size,maxpool_size)),
            nn.Conv2d(hidden_channels[1], hidden_channels[2], kernel_size, bias=False)
        )
        self.flatten = nn.Flatten()
        # len(hidden_channels)+1 : plus 1 for quadrant pooling
        self.classifier = nn.Linear(hidden_channels[2]*2**(len(hidden_channels)+1), categories)
        
    def quadrant_pooling(self, x):
        B, C, H, W = x.size()
        assert H % 2 == 0 and W % 2 == 0
        x = x.view(B, C, H//2, 2, W//2, 2)
        x, _ = torch.max(x, dim=3)
        x, _ = torch.max(x, dim=4)
        return x
    
    def forward(self, x):
        if len(x.shape) < 4:
            x = x.unsqueeze(1)
        x = self.classifier(self.flatten(self.quadrant_pooling(self.net(x))))
        return x
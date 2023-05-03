import torch.nn as nn
import torch

class slow_r50(nn.Module):
    def __init__(self, num_classes=4, pretrained=False,
                 feature_extract=False, Encoder_CheckPoint=None):
        super().__init__()
        model = torch.hub.load("facebookresearch/pytorchvideo", model='slow_r50', pretrained=pretrained)
        model.blocks[5].proj = torch.nn.Linear(2048, 4)
        self.model = model

    def forward(self,x):
        # (videos, frames, C, H, W) ==> (videos, C, F, H, W)
        x = torch.transpose(x, 1, 2)
        out = self.model(x)
        return out

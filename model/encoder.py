from torch.functional import Tensor
import torchvision.models as models
import torch.nn as nn
class Encoder(nn.Module):
    def __init__(self,device):
        super(Encoder, self).__init__()
        print("Extracting feature vectors from the image")
        pretrained_model = models.vgg16(pretrained=True).to(device)
        self.conv_base = pretrained_model.features.to(device)
        # Freeze All layers as they will be used for inference
        for _, param in enumerate(self.conv_base.parameters()):  
            param.requires_grad = False

        # Flaten layer that flatten the dimensions 2 and 3 (H and W of the feature maps respectively)
        self.flat = nn.Flatten(2,3)
        self.relu = nn.ReLU()

    def forward(self, x):
        # (batch_size, feat_maps=512, H=7 , W=7) or (batch_size, 512, 14 , 14)
        features = self.conv_base(x)
        # (batch_size, 512, 7x7=49) or (batch_size, 512, 14x14=196)
        features = self.flat(features)
        # (batch_size, 49, 512) or (batch_size, 196, 512)                     
        return self.relu(features)
import torch
import torch.nn as nn
import torchvision.models as models

class ResNetCRNN(nn.Module):
    def __init__(self, num_classes: int, img_h: int = 32, nc: int = 1, nh: int = 256):
        super().__init__()
        
        # img_h and nc are kept for compatibility with the old interface, 
        # though ResNet usually handles varying input sizes (as long as stride aligns)
        input_channel = nc
        hidden_size = nh
        
        # 1. Backbone: ResNet18 (Modified stride)
        # Standard ResNet18 reduces size by 32x (stride 2 at layer 1,2,3,4, + stem)
        # We need output height=1, width=large enough.
        # Input: 32x160
        # Stem (conv1+maxpool): /4 -> 8x40
        # Layer1: /1 -> 8x40
        # Layer2: /2 -> 4x20
        # Layer3: /2 -> 2x10 (Too small for W!) -> Change stride to (2, 1) -> 2x20
        # Layer4: /2 -> 1x5 (Too small!) -> Change stride to (2, 1) -> 1x20
        
        # Actually standard CRNN output sequence length T is often W/4 or W/8.
        # If input W=160, T=20 is okay for 7 chars. T=40 is better.
        
        resnet = models.resnet18(weights=None)
        
        # Modify input conv to accept 1 channel (grayscale)
        if input_channel != 3:
            resnet.conv1 = nn.Conv2d(input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
            
        # We take layers up to layer4
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool # stride 2 -> 1/4
        
        self.layer1 = resnet.layer1 # stride 1 -> 1/4
        self.layer2 = resnet.layer2 # stride 2 -> 1/8
        
        # Modify stride of layer3 and layer4 to preserve width
        self.layer3 = resnet.layer3 # Standard stride 2. Change to (2, 1)
        self.layer3[0].conv1.stride = (2, 1)
        self.layer3[0].downsample[0].stride = (2, 1)
        
        self.layer4 = resnet.layer4 # Standard stride 2. Change to (2, 1)
        self.layer4[0].conv1.stride = (2, 1)
        self.layer4[0].downsample[0].stride = (2, 1)
        
        # Output shape analysis:
        # In: 32 x 160
        # conv1+pool: 8 x 40
        # layer1: 8 x 40
        # layer2: 4 x 20
        # layer3: 2 x 20 (H/2, W/1)
        # layer4: 1 x 20 (H/2, W/1)
        # Final: 512 channels, H=1, W=20.
        # W=20 might be slightly tight for 7 chars + blanks. 
        # Let's relax layer2 stride to (2, 1) as well -> W=40.
        
        self.layer2[0].conv1.stride = (2, 1)
        self.layer2[0].downsample[0].stride = (2, 1)
        # Now: layer2 -> 4x40. layer3 -> 2x40. layer4 -> 1x40. Perfect.
        
        # 2. RNN Head
        self.lstm = nn.LSTM(512, hidden_size, bidirectional=True, num_layers=2, batch_first=False)
        self.embedding = nn.Linear(hidden_size * 2, num_classes)
        
        self._init_weights()

    def _init_weights(self):
        # ResNet parts are already initialized if pretrained=False, but Kaiming is good.
        # LSTM needs init
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.kaiming_normal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
        
        nn.init.normal_(self.embedding.weight, 0, 0.01)
        nn.init.constant_(self.embedding.bias, 0)

    def forward(self, x):
        # x: [B, 1, 32, 160]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x) # [B, 64, 8, 40]
        
        x = self.layer1(x)  # [B, 64, 8, 40]
        x = self.layer2(x)  # [B, 128, 4, 40] (Modified stride)
        x = self.layer3(x)  # [B, 256, 2, 40] (Modified stride)
        x = self.layer4(x)  # [B, 512, 1, 40] (Modified stride)
        
        # Check height
        assert x.size(2) == 1, f"Expected H=1, got {x.size(2)}"
        x = x.squeeze(2) # [B, 512, 40]
        
        # [B, C, W] -> [W, B, C] for LSTM
        x = x.permute(2, 0, 1) 
        
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        
        # [W, B, H*2] -> [W, B, NumClasses]
        T, B, H = x.size()
        x = x.view(T * B, H)
        x = self.embedding(x)
        x = x.view(T, B, -1)
        
        return x

# Alias for compatibility
CRNN = ResNetCRNN

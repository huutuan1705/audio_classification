import torch
import torch.nn as nn

class n_conv(nn.Module):
    def __init__(self, in_channels, out_channels, N = 2):
        super(n_conv, self).__init__()
        model = []
        model.append(nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1,1)))
        model.append(nn.ReLU(True))
        for i in range(N-1):
            model.append(nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1,1)))
            model.append(nn.ReLU(True))
        
        model.append(nn.MaxPool2d(kernel_size=(2, 2), stride=(2,2)))
        self.conv = nn.Sequential(*model)
        
    def forward(self, x):
        return self.conv(x)
    
class VGG16(nn.Module):
    def __init__(self, in_channels=2, out_channels=3, init_weights=True):
        super(VGG16, self).__init__()
        self.conv1 = n_conv(in_channels,64)
        self.conv2 = n_conv(64,128)
        self.conv3 = n_conv(128,256,N=3)
        self.conv4 = n_conv(256,512,N=3)
        self.conv5 = n_conv(512,512,N=3)
        self.avgpool = nn.AdaptiveMaxPool2d((7,7))
        self.linear1 = nn.Linear(512*7*7, 4096)
        self.linear2 = nn.Linear(4096, 4096)
        self.relu = nn.ReLU(True)
        self.dropout = nn.Dropout(0.3)
        self.linear3 = nn.Linear(4096, out_channels)
        if init_weights:
            self._initialize_weights()
            
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
                
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.avgpool(x)
        
        x = torch.flatten(x,1)
        x = self.linear1(x)
        
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear3(x)
        return x
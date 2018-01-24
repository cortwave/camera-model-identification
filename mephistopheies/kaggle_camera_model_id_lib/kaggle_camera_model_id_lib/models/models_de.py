import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

import pretrainedmodels
from pretrainedmodels.models.inceptionresnetv2 import BasicConv2d, Mixed_5b, Block35, Block17, Block8
from pretrainedmodels.models.inceptionresnetv2 import Mixed_6a, Mixed_7a

from .models import resnets



class DANet(nn.Module):

    def __init__(self, model_fe, model_d, model_c):
        super(DANet, self).__init__()
        self.feature_exctractor = model_fe
        self.discrimitator = model_d
        self.classifier = model_c        
    
    def forward(self, x, mode='c'):
        x = self.feature_exctractor(x)
        if mode == 'd':
            x = x.view(x.shape[0],-1)
            return self.discrimitator(x)
        return self.classifier(x)


class ResNetFeatureExtractor(nn.Module):

    def __init__(self, block, layers, load_resnet='resnet18'):
        self.inplanes = 64
        super(ResNetFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)                
                
        if load_resnet is not None:
            resnet = resnets[load_resnet]()
            keys = set(self.state_dict().keys())
            state_dict = self.state_dict()
            state_dict.update(dict([(k, v) for (k, v) in resnet.state_dict().items() if k in keys]))
            self.load_state_dict(state_dict)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x
    
    
class AvgFcClassifier(nn.Module):

    def __init__(self, n_classes, load_resnet='resnet18'):
        super(AvgFcClassifier, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, n_classes)
        
    def forward(self, x):
        x = self.avgpool(x).squeeze()
        x = self.fc(x)
        return x
    

class AvgClassifier(nn.Module):

    def __init__(self, n_classes, in_planes):
        super(AvgClassifier, self).__init__()
        self.project = nn.Conv2d(            
            in_planes, n_classes, 
            kernel_size=(1, 1), stride=(1, 1), padding=(0, 0),
            bias=True)
        self.bn2 = nn.BatchNorm2d(n_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.project(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.avgpool(x)
        return x.squeeze()

    

class FCDiscriminator(nn.Module):

    def __init__(self):
        super(FCDiscriminator, self).__init__()
        self.net = model_d = nn.Sequential(
            nn.Linear(512*8*8, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
    def forward(self, x):
        return F.sigmoid(self.net(x))
    
    

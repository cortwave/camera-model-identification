import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

import pretrainedmodels
from pretrainedmodels.models.inceptionresnetv2 import BasicConv2d, Mixed_5b, Block35, Block17, Block8
from pretrainedmodels.models.inceptionresnetv2 import Mixed_6a, Mixed_7a


resnets = {
    'resnet18': lambda: models.resnet18(pretrained=True),
    'resnet34': lambda: models.resnet34(pretrained=True),
    'resnet50': lambda: models.resnet50(pretrained=True),
    'resnet101': lambda: models.resnet101(pretrained=True),
    'resnet152': lambda: models.resnet152(pretrained=True)
}


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'E_2b': [64, 64, 'M', 128, 128, 'M'],
    'E_3b': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M']
}

cfg_vgg_map = {
    'A': lambda: models.vgg11(pretrained=True),
    'B': lambda: models.vgg13(pretrained=True),
    'D': lambda: models.vgg16(pretrained=True),
    'E': lambda: models.vgg19(pretrained=True),
    'E_2b': lambda: models.vgg19(pretrained=True),
    'E_3b': lambda: models.vgg19(pretrained=True)
}

cfg_vgg_bn_map = {
    'A': lambda: models.vgg11_bn(pretrained=True),
    'B': lambda: models.vgg13_bn(pretrained=True),
    'D': lambda: models.vgg16_bn(pretrained=True),
    'E': lambda: models.vgg19_bn(pretrained=True),
    'E_2b': lambda: models.vgg19_bn(pretrained=True),
    'E_3b': lambda: models.vgg19_bn(pretrained=True)
}


def make_layers_vgg(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

    
class VggHead(nn.Module):

    def _transfer_vgg_head(self, vgg_key, load_vgg_bn=False, batch_norm=False):
        self.vgg = make_layers_vgg(cfg[vgg_key], batch_norm=batch_norm)
        
        if load_vgg_bn:
            if batch_norm:
                vgg = cfg_vgg_bn_map[vgg_key]()
            else:
                vgg = cfg_vgg_map[vgg_key]()
            keys = set(self.vgg.state_dict().keys())
            self.vgg.load_state_dict(dict([(k, v) for (k, v) in vgg.features.state_dict().items() if k in keys]))
            
    def _create_classifier(self, in_filters, num_classes):
        self.project = nn.Conv2d(            
            in_filters, num_classes, 
            kernel_size=(1, 1), stride=(1, 1), padding=(0, 0),
            bias=True)
        self.bn = nn.BatchNorm2d(num_classes)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
    
    def __init__(self, vgg_key, batch_norm=True, num_classes=10, load_vgg_bn=False):
        super(VggHead, self).__init__()
        self._transfer_vgg_head(load_vgg_bn=load_vgg_bn, vgg_key=vgg_key, batch_norm=batch_norm)
        self._create_classifier(cfg[vgg_key][-2], num_classes)
            
        
        

    def forward(self, x):
        x = self.vgg(x)
        x = self.project(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.shape[0], x.shape[1])
        return x
    
    
    
class StyleVggHead(VggHead):
    
    def _create_classifier(self, num_classes):
        self.fc1 = nn.Linear(8128, 4096)
        self.bn1 = nn.BatchNorm1d(4096)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(4096, 2048)
        self.bn2 = nn.BatchNorm1d(2048)
        self.fc3 = nn.Linear(2048, num_classes)
    
    def __init__(self, vgg_key, batch_norm=True, num_classes=10, load_vgg_bn=False):
        super(StyleVggHead, self).__init__(
            num_classes, 
            load_vgg_bn=load_vgg_bn, 
            vgg_key=vgg_key, 
            batch_norm=batch_norm)
        self._create_classifier(num_classes)
        

    def forward(self, x):
        x = self.vgg(x)
        
        (b, c, h, w) = x.shape
        x = x.view(b, c, w * h)
        x = x.bmm(x.transpose(1, 2)) / (c * h * w)
        
        x = torch.stack(
            [torch.cat([torch.diag(x[j, :], i) for i in range(1, x.shape[1])]) 
             for j in range(x.shape[0])])
        
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.fc3(x)
        
        return x
    


            
class IEEEfcn(nn.Module):
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
    
    def __init__(self, num_classes=10):
        super(IEEEfcn, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(4, 4), stride=1),
            nn.SELU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(32, 48, kernel_size=(5, 5), stride=1),
            nn.SELU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(48, 64, kernel_size=(5, 5), stride=1),
            nn.SELU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(64, 128, kernel_size=(5, 5), stride=1),
            nn.SELU(inplace=True),
            nn.Conv2d(128, num_classes, kernel_size=(1, 1), stride=1),
            nn.SELU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self._initialize_weights()
        

    def forward(self, x):
        x = self.net(x)        
        return x.squeeze()
    


class ResNetFC(nn.Module):

    def __init__(self, block, layers, num_classes=1000, load_resnet='resnet18'):
        self.inplanes = 64
        super(ResNetFC, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.project = nn.Conv2d(            
            512*block.expansion, num_classes, 
            kernel_size=(1, 1), stride=(1, 1), padding=(0, 0),
            bias=True)
        self.bn2 = nn.BatchNorm2d(num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
                
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

        x = self.project(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.avgpool(x)

        return x.squeeze()
    
class ResNetX(nn.Module):

    def __init__(self, block, layers, num_classes=1000, load_resnet='resnet18'):
        self.inplanes = 64
        super(ResNetX, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
               
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(512 * block.expansion, num_classes)
        

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
                
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

        x = self.avgpool(x)
        x = self.fc1(x.squeeze())

        return x

    
class InceptionResNetV2fc(nn.Module):

    def __init__(self, num_classes=1001, nun_block35=10, num_block17=20, num_block8=9):
        super(InceptionResNetV2fc, self).__init__()
        # Modules
        self.conv2d_1a = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.conv2d_2a = BasicConv2d(32, 32, kernel_size=3, stride=1)
        self.conv2d_2b = BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.maxpool_3a = nn.MaxPool2d(3, stride=2)
        self.conv2d_3b = BasicConv2d(64, 80, kernel_size=1, stride=1)
        self.conv2d_4a = BasicConv2d(80, 192, kernel_size=3, stride=1)
        self.maxpool_5a = nn.MaxPool2d(3, stride=2)
        self.mixed_5b = Mixed_5b()
        self.repeat = nn.Sequential(
            *[Block35(scale=0.17) for i in range(nun_block35)]
        )
        self.mixed_6a = Mixed_6a()
        self.repeat_1 = nn.Sequential(
            *[Block17(scale=0.10) for i in range(num_block17)]
        )
        self.mixed_7a = Mixed_7a()
        self.repeat_2 = nn.Sequential(
            *[Block8(scale=0.20) for i in range(num_block8)]
        )
        self.block8 = Block8(noReLU=True)
        self.conv2d_7b = BasicConv2d(2080, 1536, kernel_size=1, stride=1)
        
        self.project = nn.Conv2d(            
            1536, num_classes, 
            kernel_size=(1, 1), stride=(1, 1), padding=(0, 0),
            bias=True)
        self.bn2 = nn.BatchNorm2d(num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU(inplace=True)

    def features(self, input):
        x = self.conv2d_1a(input)
        x = self.conv2d_2a(x)
        x = self.conv2d_2b(x)
        x = self.maxpool_3a(x)
        x = self.conv2d_3b(x)
        x = self.conv2d_4a(x)
        x = self.maxpool_5a(x)
        x = self.mixed_5b(x)
        x = self.repeat(x)
        x = self.mixed_6a(x)
        x = self.repeat_1(x)
        x = self.mixed_7a(x)
        x = self.repeat_2(x)
        x = self.block8(x)
        x = self.conv2d_7b(x)
        return x

    def logits(self, features):
        x = self.project(features)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.avgpool(x)
        return x.squeeze()

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x
    

class InceptionResNetV2fcSmall(nn.Module):

    def __init__(self, num_classes=1001, nun_block35=10, num_block17=20):
        super(InceptionResNetV2fcSmall, self).__init__()
        # Modules
        self.conv2d_1a = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.conv2d_2a = BasicConv2d(32, 32, kernel_size=3, stride=1)
        self.conv2d_2b = BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.maxpool_3a = nn.MaxPool2d(3, stride=2)
        self.conv2d_3b = BasicConv2d(64, 80, kernel_size=1, stride=1)
        self.conv2d_4a = BasicConv2d(80, 192, kernel_size=3, stride=1)
        self.maxpool_5a = nn.MaxPool2d(3, stride=2)
        self.mixed_5b = Mixed_5b()
        self.repeat = nn.Sequential(
            *[Block35(scale=0.17) for i in range(nun_block35)]
        )
        self.mixed_6a = Mixed_6a()
        self.repeat_1 = nn.Sequential(
            *[Block17(scale=0.10) for i in range(num_block17)]
        )
                
        self.project = nn.Conv2d(            
            1088, num_classes, 
            kernel_size=(1, 1), stride=(1, 1), padding=(0, 0),
            bias=True)
        self.bn2 = nn.BatchNorm2d(num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU(inplace=True)

    def features(self, input):
        x = self.conv2d_1a(input)
        x = self.conv2d_2a(x)
        x = self.conv2d_2b(x)
        x = self.maxpool_3a(x)
        x = self.conv2d_3b(x)
        x = self.conv2d_4a(x)
        x = self.maxpool_5a(x)
        x = self.mixed_5b(x)
        x = self.repeat(x)
        x = self.mixed_6a(x)
        x = self.repeat_1(x)
        return x

    def logits(self, features):
        x = self.project(features)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.avgpool(x)
        return x.squeeze()

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x
    

class FatNet1(nn.Module):
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
    
    def __init__(self, num_classes=10):
        super(FatNet1, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 128, 11, stride=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 7, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, 5, stride=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(512, 1024, 3, stride=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(1024, 2048, 1, stride=1, bias=False),
            nn.BatchNorm2d(2048),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(2048, 4096, 3, stride=1, bias=False),
            nn.BatchNorm2d(4096),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 2048, bias=False),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 1024, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, num_classes, bias=True)
        )
        self._initialize_weights()
        

    def forward(self, x):
        x = self.features(x)
        x = x.squeeze()
        x = self.classifier(x)
        return x


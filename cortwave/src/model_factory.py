from torchvision.models import vgg19, squeezenet1_1, resnet152, resnet34, resnet50, resnet101, \
    densenet121, densenet161, densenet169, densenet201, resnet18
from inception.inception import inception_v3
from mobilenet.mobilenet import mobilenetv2
import torch.nn as nn
import torch
from widenet.widenet import Widenet
from dpn import model_factory as dpn_factory
from se_net.se_inception import SEInception3
import sys
sys.path.append('../pretrained-models.pytorch')
import pretrainedmodels


def get_model(num_classes, architecture):
    model = None
    if architecture == 'widenet':
        model = Widenet(num_classes).cuda()
    elif architecture == 'inceptionresnetv2':
        model = pretrainedmodels.__dict__[architecture](num_classes=num_classes, pretrained=False)
        features = nn.Sequential(
            model.conv2d_1a,
            model.conv2d_2a,
            model.conv2d_2b,
            model.maxpool_3a,
            model.conv2d_3b,
            model.conv2d_4a,
            model.maxpool_5a,
            model.mixed_5b,
            model.repeat,
            model.mixed_6a,
            model.repeat_1,
            model.mixed_7a,
            model.repeat_2,
            model.block8,
            model.conv2d_7b,
            model.avgpool_1a
        )
        classifier = nn.Linear(1536 + 1, num_classes)
        model = ManipModel(features, classifier).cuda()
    elif 'vgg' in architecture:
        if architecture == 'vgg19':
            model = vgg19(pretrained=True).cuda()
        if model is not None:
            model.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, num_classes),
            )
    elif 'inception_v3' in architecture:
        model = inception_v3(pretrained=False, aux_logits=False, num_classes=num_classes)
    elif "seinception" in architecture:
        model = SEInception3(num_classes=num_classes).cuda()
    elif "resnet" in architecture:
        if architecture == 'resnet18':
            model = resnet18(pretrained=True).cuda()
        elif architecture == 'resnet34':
            model = resnet34(pretrained=True).cuda()
        elif architecture == 'resnet50':
            model = resnet50(pretrained=True).cuda()
        elif architecture == 'resnet101':
            model = resnet101(pretrained=True).cuda()
        elif architecture == 'resnet152':
            model = resnet152(pretrained=True).cuda()
        if model is not None:
            features = nn.Sequential(
                model.conv1,
                model.bn1,
                model.relu,
                model.maxpool,
                model.layer1,
                model.layer2,
                model.layer3,
                model.layer4,
                nn.AdaptiveAvgPool2d(1)
            )
            classifier = nn.Linear(model.fc.in_features + 1, num_classes)
            model = ManipModel(features, classifier).cuda()
    elif "densenet" in architecture:
        if architecture == 'densenet121':
            model = densenet121(pretrained=True).cuda()
        elif architecture == "densenet161":
            model = densenet161(pretrained=True).cuda()
        elif architecture == "densenet169":
            model = densenet169(pretrained=True).cuda()
        elif architecture == "densenet201":
            model = densenet201(pretrained=True).cuda()
        if model is not None:
            features = nn.Sequential(
                model.features,
                nn.ReLU(inplace=True),
                nn.AvgPool2d(kernel_size=7)
            )
            classifier = nn.Linear(model.classifier.in_features + 1, num_classes)
            model = ManipModel(features, classifier).cuda()
    elif "squeezenet" in architecture:
        if architecture == "squeezenet1_1":
            model = squeezenet1_1(pretrained=True).cuda()
        if model is not None:
            final_conv = nn.Conv2d(512, num_classes, kernel_size=1)
            model.classifier = nn.Sequential(
                nn.Dropout(p=0.5),
                final_conv,
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1)
            )
            model.num_classes = num_classes
    elif "dpn" in architecture:
        if architecture == "dpn68":
            model = dpn_factory.create_model(architecture,
                                             num_classes=1000,
                                             pretrained=False,
                                             test_time_pool=False)
            model.classifier = nn.Conv2d(model.in_chs, num_classes, kernel_size=1, bias=True)
    elif architecture == "mobilenetv2":
        model = mobilenetv2(pretrained=False)
        features = model.features
        classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(model.last_channel + 1, num_classes),
        )
        model = ManipModel(features, classifier).cuda()
    if model is None:
        raise ValueError('Unknown architecture: ', architecture)
    return nn.DataParallel(model).cuda()


class ManipModel(nn.Module):
    def __init__(self, features, classifier):
        super(ManipModel, self).__init__()
        self.features = features
        self.classifier = classifier

    def forward(self, x, is_manip):
        features_result = self.features(x)
        features_result = features_result.view(x.size()[0], -1)
        classifier_input = torch.cat((features_result, is_manip), -1)
        out = self.classifier(classifier_input)
        return out

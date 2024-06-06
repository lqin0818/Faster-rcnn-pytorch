import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url

class VGG(nn.Module):
    def __init__(self, features, nc=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, nc),
        )
        if init_weights:
            self.initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)

        x = torch.flatten(x, 1)

        x = self.classifier(x)
        return x

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']

def make_layers(cfg, batch_norm = False):

    layers = []
    cin =3

    for m in cfg:
        if isinstance(m,int):
            layers.append(nn.Conv2d(cin,m,3,1,1))

            if batch_norm:
                layers.append(nn.BatchNorm2d(m))

            layers.append(nn.ReLU())
            cin = m
        else:
            layers.append(nn.MaxPool2d(2,2))

    return nn.Sequential(*layers)

def decom_vgg16(pretrained = False):
    model = VGG(make_layers(cfg))
    if pretrained:
        state_dict = load_state_dict_from_url("https://download.pytorch.org/models/vgg16-397923af.pth", model_dir="./model_data")
        model.load_state_dict(state_dict)
    #----------------------------------------------------------------------------#
    #   获取特征提取部分，最终获得一个37,37,1024的特征层
    #----------------------------------------------------------------------------#
    features = list(model.features)[:30]
    #----------------------------------------------------------------------------#
    #   获取分类部分，需要除去Dropout部分
    #----------------------------------------------------------------------------#
    classifier = list(model.classifier)
    del classifier[6]
    del classifier[5]
    del classifier[2]

    features = nn.Sequential(*features)
    classifier = nn.Sequential(*classifier)
    return features, classifier

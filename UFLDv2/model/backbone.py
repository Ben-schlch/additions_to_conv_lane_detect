import torch,pdb
import torchvision
import torch.nn.modules


class MobileNetV3(torch.nn.Module):
    def __init__(self, pretrained=False):
        super(MobileNetV3, self).__init__()
        self.model = torchvision.models.mobilenet_v3_small(weight= torchvision.models.MobileNet_V3_Small_Weights)
        self.red_conv = torch.nn.Conv2d(576, 24, 1)
        self.maxpool = torch.nn.MaxPool2d(1, stride=1)  # Das ergibt 0 Sinn ... nicht benutzen
        self.model.last_layer = torch.nn.Linear(27648, 1152)

    def forward(self, x):
        x = self.model.features(x)
        # x = self.model.avgpool(x)
        return x, x, x

class MobileNetV2(torch.nn.Module):
    def __init__(self, pretrained=False):
        super(MobileNetV2, self).__init__()
        self.model = torchvision.models.mobilenet_v2(pretrained=pretrained).features

    def forward(self, x):
        x = self.model.forward(x)
        #print(type(x))
        x = self.model.last_layer(x)
        x = list(self.model.children())[-1]
        #print(type(x))
        return x, x, x.view(-1, 24, 6, 8)
        #return self.model(x)


class vgg16bn(torch.nn.Module):
    def __init__(self,pretrained = False):
        super(vgg16bn,self).__init__()
        model = list(torchvision.models.vgg16_bn(pretrained=pretrained).features.children())
        model = model[:33]+model[34:43]
        self.model = torch.nn.Sequential(*model)
        
    def forward(self,x):
        return self.model(x)


class resnet(torch.nn.Module):
    def __init__(self,layers,pretrained = False):
        super(resnet,self).__init__()
        if layers == '18':
            model = torchvision.models.resnet18(pretrained=pretrained)

            # model.state_dict()
            # state_dict = model.state_dict()
            # conv1_weight = state_dict['conv1.weight']
            # state_dict['conv1.weight'] = conv1_weight.sum(dim=1, keepdim=True)
            # model.load_state_dict(state_dict)
        elif layers == '34':
            model = torchvision.models.resnet34(pretrained=pretrained)
        elif layers == '50':
            model = torchvision.models.resnet50(pretrained=pretrained)
        elif layers == '101':
            model = torchvision.models.resnet101(pretrained=pretrained)
        elif layers == '152':
            model = torchvision.models.resnet152(pretrained=pretrained)
        elif layers == '50next':
            model = torchvision.models.resnext50_32x4d(pretrained=pretrained)
        elif layers == '101next':
            model = torchvision.models.resnext101_32x8d(pretrained=pretrained)
        elif layers == '50wide':
            model = torchvision.models.wide_resnet50_2(pretrained=pretrained)
        elif layers == '101wide':
            model = torchvision.models.wide_resnet101_2(pretrained=pretrained)
        elif layers == '34fca':
            model = torch.hub.load('cfzd/FcaNet', 'fca34' ,pretrained=True)
        elif layers == 'mobilenet-v3-small':
            model = torchvision.models.mobilenet_v3_small(weights=torchvision.models.MobileNet_V3_Small_Weights)
            model.last_layer = torch.nn.Linear(1000, 1152)
        elif layers == 'mobilenet-v3-large':
            model = torchvision.models.mobilenet_v3_large(weights=torchvision.models.MobileNet_V3_Large_Weights)
        else:
            raise NotImplementedError

        self.model = model
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x2 = self.layer2(x)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x2,x3,x4
        ################################# Fuer Mobile Net ##################################
        #x = self.model.forward(x)
        #print(type(x))
        #x = self.model.last_layer(x)
        #x = list(self.model.children())[-1]
        #print(type(x))
        #return x, x, x.view(-1, 24, 6, 8)


import torch
import torch.nn as nn

# torch.nn.Conv2d(
#     in_channels, 
#     out_channels, 
#     kernel_size, 
#     stride=1, 
#     padding=0, 
#     dilation=1, 
#     groups=1, 
#     bias=True, 
#     padding_mode='zeros'
# )

vgg11_config = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
vgg13_config = [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
vgg16_config = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
vgg19_config = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']

# All layer
class VGG(nn.Module):
    def __init__(self, features, output_dim):
        super().__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d(7)
        self.classifier = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, output_dim)
        )
        
    def forward(self, x):
        x = self.features(x) # torch.Size([32, 512, 8, 8])
        x = self.avgpool(x) # torch.Size([32, 512, 7, 7])
        h = x.view(x.shape[0], -1) # torch.Size([32, 512*7*7])
        x = self.classifier(h) # torch.Size([32, 2])
        return x

# VGG layer
def get_vgg_layers(config, batch_norm):
    layers = []
    in_channels = 3
    
    # VGG Net layer 구성
    for c in config:
        assert (c == 'M' or isinstance(c, int)) # assert : 조건이 True가 아니면 에러 발생 # isinstance : 변수의 자료형 판단하여 True인지 False인지 제공
        # Maxpooling layer 결합
        if c == 'M':
            layers += [nn.MaxPool2d(kernel_size = 2)]
        # Convolution layer 결합
        else:
            conv2d = nn.Conv2d(in_channels, c, kernel_size = 3, padding = 1)
            # 배치 정규화 적용 : 계층마다 변화하는 분포(학습 속도 저하, 학습 어려움 등의 문제) -> 계층마다 평균 0, 표준편차 1로 분포(학습 속도 향상, 학습 가능성 증가)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(c), nn.ReLU(inplace = True)]
            # 배치 정규화 미적용
            else:
                layers += [conv2d, nn.ReLU(inplace = True)]
            # 채널 업데이트
            in_channels = c
            
    return nn.Sequential(*layers)

# Define model
class ConvNet(nn.Module):
    def __init__(self, drop_rate=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv4 = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv5 = nn.Conv2d(128, 512, 3, 1, 1)
        self.drop = nn.Dropout(drop_rate)
        self.maxpool = nn.MaxPool2d(kernel_size = 2)
        self.fc = nn.Linear(512, 2)
        self.act = nn.ReLU()
        
    def forward(self, x):
        # torch.Size([32, 3, 256, 256])
        x = self.act(self.conv1(x)) # torch.Size([32, 64, 256, 256])
        x = self.maxpool(x) # torch.Size([32, 64, 128, 128])
        x = self.act(self.conv2(x)) # torch.Size([32, 64, 128, 128])
        x = self.maxpool(x) # torch.Size([32, 64, 64, 64])
        x = self.act(self.conv3(x)) # torch.Size([32, 128, 64, 64])
        x = self.maxpool(x) # torch.Size([32, 128, 32, 32])
        x = self.act(self.conv4(x)) # torch.Size([32, 128, 32, 32])
        x = self.maxpool(x) # torch.Size([32, 128, 16, 16])
        x = self.act(self.conv5(x)) # torch.Size([32, 512, 16, 16])
        x = x.mean([-1, -2]) # torch.Size([32, 512])
        x = self.drop(x) # torch.Size([32, 512])
        x = self.fc(x) # torch.Size([32, 2])
        return x

##############################################################################################################

"""
A : VGG-11
B : VGG-13
D : VGG-16
E : VGG-19
"""

"""
cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def VGG11(retina=False):
    return VGG(config = cfg['A'], retina = retina)

def VGG13(retina=False):
    return VGG(config = cfg['B'], retina = retina)

def VGG16(retina=False):
    return VGG(config = cfg['D'], retina = retina)

def VGG19(retina=False):
    return VGG(config = cfg['E'], retina = retina)

def make_layer(config):
    layers = []
    in_channel = 3
    for value in config:
        if value == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            out_channel = value
            layers.append(nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            in_channel = value
    return nn.Sequential(*layers)

class VGG(nn.Module):
    def __init__(self, config, num_classes=5, retina=False):
        super(VGG, self).__init__()
        self.features = make_layer(config)

        # ImageNet
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes)
        )

        # CIFAR-10
        if retina:
            self.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(512, 512),
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.Linear(512, 512),
                nn.ReLU(True),
                nn.Linear(512, num_classes)
            )

    def forward(self, x):
        # print(x.size())  # torch.Size([batch_size, channel, width, height]) # 확인용 코드

        conv_out = self.features(x)
        # print(conv_out.size())  # torch.Size([batch_size, channel, width, height]) # 확인용 코드

        linear_out = torch.flatten(conv_out, 1)
        # print(linear_out.size())  # torch.Size([batch_size, channel, width, height]) # 확인용 코드

        class_out = self.classifier(linear_out)
        # print(class_out.size())  # torch.Size([batch_size, channel, width, height]) # 확인용 코드

        return class_out
"""

##############################################################################################################

"""
# Define model
class VGG(nn.Module):
    # CNN층 정의
    def __init__(self, drop_rate=0.0):
        super().__init__()
        self.conv = nn.Sequential(
            #3 224 128
            nn.Conv2d(3, 64, 3, padding=1),nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, 3, padding=1),nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),
            #64 112 64
            nn.Conv2d(64, 128, 3, padding=1),nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, 3, padding=1),nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),
            #128 56 32
            nn.Conv2d(128, 256, 3, padding=1),nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, 3, padding=1),nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, 3, padding=1),nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),
            #256 28 16
            nn.Conv2d(256, 512, 3, padding=1),nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, 3, padding=1),nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, 3, padding=1),nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),
            #512 14 8
            nn.Conv2d(512, 512, 3, padding=1),nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, 3, padding=1),nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, 3, padding=1),nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2)
        )

        #512 7 4
        self.avg_pool = nn.AvgPool2d(7)
        #512 1 1
        self.classifier = nn.Linear(512, 5)

        # self.avg_pool = nn.AvgPool2d(56)
        # self.classifier = nn.Linear(128, 5)

#         self.fc1 = nn.Linear(512*2*2,4096)
#         self.fc2 = nn.Linear(4096,4096)
#         self.fc3 = nn.Linear(4096,10)

    # CNN층 결합
    def forward(self, x): # conv2d_1 -> relu -> conv2d_2 -> relu -> conv2d_3 -> relu -> conv2d_4 -> relu -> mean -> dropout -> fully connected layer
        print(x.size()) # torch.Size([batch_size, channel, width, height]) # 확인용 코드

        features = self.conv(x)
        print(features.size()) # torch.Size([batch_size, channel, width, height]) # 확인용 코드

        avg_pool = self.avg_pool(features)
        print(avg_pool.size()) # torch.Size([batch_size, channel, width, height]) # 확인용 코드

        flatten = avg_pool.view(features.size(0), -1)
        print(flatten.size()) # torch.Size([batch_size, flatten]) # 확인용 코드

        x = self.classifier(flatten)
        print(x.size()) # torch.Size([batch_size, class]) # 확인용 코드

        x = self.softmax(x)
        return x

#         print("x : ", x) # 확인용 코드
#         print("x.mean([-1, -2]) : ", x.mean([-1, -2])) # 확인용 코드

#         x = self.drop(x)
#         x = self.fc(x)
#         return x
"""
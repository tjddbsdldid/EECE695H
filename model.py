import torch.nn as nn

""" Optional conv block """

def conv_block(in_channels, out_channels):

    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2),
    )

""" Define your own model """
class FewShotModel(nn.Module):
    def __init__(self, x_dim=3, hid_dim=64, z_dim=64):
        super(FewShotModel, self).__init__()       

        self.layer1 = conv_block(x_dim, hid_dim)                     #conv_block(in_channels, out_channels) : [conv layer, batch norm, Relu 활성화 함수] 3개 반복, 입력은 input과 output의 channel 개수
        self.layer2 = conv_block(hid_dim, hid_dim)
        self.layer3 = conv_block(hid_dim, hid_dim)
        self.layer4 = conv_block(hid_dim, z_dim)
        
    def forward(self, x):
        result1 = self.layer1(x)
        result2 = self.layer2(result1) + result1
        result3 = self.layer3(result2) + result2
        output = self.layer4(result3) + result3
        
        embedding_vector = output.view(output.size(0), -1)
        return embedding_vector
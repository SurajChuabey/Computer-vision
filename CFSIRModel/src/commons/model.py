import torch
import torch.nn as nn
import torch.nn.functional as F

class CFSIRHEADBlock(nn.Module):
    def __init__(self, input_channels=3, output_channels=32, kernel_size=3, stride=1, padding=1):
        super(CFSIRHEADBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, padding=padding, stride=stride)
        self.batch_norm = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.batch_norm(self.conv1(x)))
        return x
    
class CFSIRMidBlock(nn.Module):
    def __init__(self,input_channels = 3 , output_channels = 32,max_pool = 2,kernel_size = 3 ,stride = 1,padding = 1):
        super(CFSIRMidBlock,self).__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, padding=padding, stride=stride)
        self.batch_norm = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(max_pool)

    def forward(self, x):
        x = self.max_pool(self.relu(self.batch_norm(self.conv1(x))))
        return x

class ClassifierBlock(nn.Module):
    def __init__(self,first_linear_channel_inp,first_linear_channel_out,dropout,num_classes):
        super(ClassifierBlock,self).__init__()
        self.adpt_pool = nn.AdaptiveAvgPool2d((1,1))
        self.first_linear_layer = nn.Linear(first_linear_channel_inp,first_linear_channel_out)
        self.second_linear_layer = nn.Linear(first_linear_channel_out,num_classes)
        self.dropout = nn.Dropout(dropout)
        self.flatten = nn.Flatten()

    def forward(self,x):
        x = self.adpt_pool(x)
        x = self.flatten(x)
        x = F.relu(self.first_linear_layer(x))
        x = self.dropout(x)
        x = self.second_linear_layer(x)

        return x

class CFSIRModel(nn.Module):
    def __init__(self, num_classes=10):
        super(CFSIRModel, self).__init__()

        self.head_block1 = CFSIRHEADBlock(input_channels=3,output_channels=64,kernel_size=3,padding=1)
        self.mid_block1 = CFSIRMidBlock(input_channels=64,output_channels=64,max_pool=2,kernel_size=3,padding=1)

        self.head_block2 = CFSIRHEADBlock(input_channels=64,output_channels=128,kernel_size=3,padding=1)
        self.mid_block2 = CFSIRMidBlock(input_channels=128,output_channels=128,max_pool=2,kernel_size=3,padding=1)
        self.mid_block3 = CFSIRMidBlock(input_channels=128,output_channels=256,max_pool=2,kernel_size=3,padding=1)

        self.classifier_block = ClassifierBlock(first_linear_channel_inp=256,first_linear_channel_out=128,dropout=0.5,num_classes=num_classes)

    def forward(self, x):
        x = self.head_block1(x)
        x = self.mid_block1(x)

        x = self.head_block2(x)
        x = self.mid_block2(x)
        x = self.mid_block3(x)

        x = self.classifier_block(x)
        return x

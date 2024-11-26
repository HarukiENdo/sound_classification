import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import Transformer
from torch.quantization import QuantStub, DeQuantStub

# Create previous CNN model #todo: dropout and depthwise
#CNNモデルの構築　モデルは4種類　depthwiseはpytorchでは使用できない
class CNNNetwork1(nn.Module):
    def __init__(self, spectrogram_height, spectrogram_width, output_class_number):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16, 
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3, 
                stride=1,
                padding=1
            ),
            nn.ReLU()
        )
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(32 * spectrogram_height * spectrogram_width, output_class_number)
        
    def forward(self, input_data):
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.flatten(x)
        logits = self.linear(x)
        return logits

class CNNNetwork2(nn.Module):
    def __init__(self, spectrogram_height, spectrogram_width, output_class_number):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1, # spectogram is treated as grayscale
                out_channels=4,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=4,
                out_channels=8,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU()
        )
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(8 * spectrogram_height * spectrogram_width, output_class_number)
        
    def forward(self, input_data):
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.flatten(x)
        logits = self.linear(x)
        return logits

class CNNNetwork3(nn.Module):
    def __init__(self, spectrogram_height, spectrogram_width, output_class_number):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1, # spectogram is treated as grayscale
                out_channels=8,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=8,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU()
        )
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(16 * spectrogram_height * spectrogram_width, output_class_number)

    def forward(self, input_data):
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.flatten(x)
        logits = self.linear(x)
        return logits

#可変サイズのためforward_conv_layerを定義
class CNNNetwork4(nn.Module):
    def __init__(self, spectrogram_height, spectrogram_width, output_class_number):
        super().__init__()
        # 畳み込み層とReLUを定義
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU()
        self.norm1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=2)
        self.relu2 = nn.ReLU()
        self.norm2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.norm3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU()
        self.norm4 = nn.BatchNorm2d(128)

        # Avg pooling
        self.avepool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()

        # ここで線形層を事前に定義
        dummy_input = torch.randn(1, 1, spectrogram_height, spectrogram_width)
        x = self.forward_conv_layers(dummy_input)
        linear_input_size = x.size(1)
        self.linear = nn.Linear(linear_input_size, output_class_number)

    def forward_conv_layers(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.norm4(x)
        x = self.relu4(x)

        x = self.avepool1(x)
        x = self.flatten(x)
        return x

    def forward(self, input_data):
        x = self.forward_conv_layers(input_data)
        logits = self.linear(x)
        return logits
    
class TransformerModel(nn.Module):
    def __init__(self, input_dim, num_heads, hidden_dim, num_layers, output_class_number):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        encoder_layers = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4) #batch_first=Trueにすることで(batch_size,seq_len,feature_dim)になる
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.output_layer = nn.Linear(hidden_dim, output_class_number)

    def forward(self, src):
        src = src.permute(2, 0, 1)
        src = self.embedding(src)
        # srcの形状は [seq_len, batch_size, hidden_dim] である必要があります
        output = self.transformer_encoder(src)
        # 最後のタイムステップの出力を分類器に渡します
        output = self.output_layer(output[-1, :, :])
        return output
    

#----------------知識蒸留用のstudentモデル--------------------------------------------------------------------------------------------
#Studentモデル Student_s, StudentResnetを定義 
class Student_s(nn.Module):
    def __init__(self, spectrogram_height, spectrogram_width, output_class_number):
        super(Student_s, self).__init__()
        # 各クラスのインスタンス（入出力サイズなどの設定）
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), bias=False)
        self.norm1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.norm2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace = True)

        self.avepool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()

        # ここで線形層を事前に定義
        dummy_input = torch.randn(1, 1, spectrogram_height, spectrogram_width)
        x = self.forward_conv_layers(dummy_input)
        linear_input_size = x.size(1)
        self.linear = nn.Linear(linear_input_size, output_class_number)
    
    def forward_conv_layers(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu2(x)

        x = self.avepool1(x)
        x = self.flatten(x)

        return x
    
    def forward(self, input_data):
        x = self.forward_conv_layers(input_data)
        logits = self.linear(x)
        return logits      

#-----------------------Resnetの残差構造を考慮したstudentモデル-------------------------------------
class StudentResNet(nn.Module):
    def __init__(self, num_classes):
        super(StudentResNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(32, 32, 2)  # Reduced channels
        self.layer2 = self._make_layer(32, 64, 2, stride=2)
        self.layer3 = self._make_layer(64, 128, 2, stride=2)
        self.layer4 = self._make_layer(128, 256, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))
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
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

#bongさんの再現モデル
class depthwise_CNN2(nn.Module):
    def __init__(self, spectrogram_height, spectrogram_width, output_class_number):
        super(depthwise_CNN2, self).__init__()
        # Depthwise convolution layers
        self.depthwise_conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(5, 11), stride=2, padding=(2,0), groups=1, bias=True)
        self.depthwise_conv2 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3), stride=1, padding=1, groups=8, bias=True)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

        # ここで線形層を事前に定義
        dummy_input = torch.randn(1, 1, spectrogram_height, spectrogram_width)
        x = self.forward_conv_layers(dummy_input)
        linear_input_size = x.size(1)
        self.linear = nn.Linear(linear_input_size, output_class_number)

    def forward_conv_layers(self, x):
        x = self.depthwise_conv1(x)
        x = self.relu1(x)
        x = self.depthwise_conv2(x)
        x = self.relu2(x)
        x = torch.flatten(x, 1)
        return x
    
    def forward(self, input_data):
        x = self.forward_conv_layers(input_data)
        logits = self.linear(x)
        return logits



#Depthwise_Separable Convolutionモデル
# class depthwise_separable_CNN2(nn.Module):
#     def __init__(self, spectrogram_height, spectrogram_width, output_class_number):
#         super(depthwise_separable_CNN2, self).__init__()

#         # Depthwise Separable Convolution layers
#         # Depthwise convolution: in_channels = out_channels, groups = in_channels
#         self.depthwise_conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(10, 10), groups=1, bias=False)
#         self.pointwise_conv1 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=1, bias=False)
        
#         self.depthwise_conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(10, 10), groups=16, bias=False)
#         self.pointwise_conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=1, bias=False)

#         self.bn1 = nn.BatchNorm2d(8)
#         self.bn2 = nn.BatchNorm2d(16)
#         self.bn3 = nn.BatchNorm2d(16)
#         self.bn4 = nn.BatchNorm2d(32)

#         self.relu = nn.ReLU()
#         self.pool = nn.MaxPool2d(2, 2)

#         # ここで線形層を事前に定義
#         dummy_input = torch.randn(1, 1, spectrogram_height, spectrogram_width)
#         x = self.forward_conv_layers(dummy_input)
#         linear_input_size = x.size(1)
#         self.linear = nn.Linear(linear_input_size, output_class_number)

#     def forward_conv_layers(self, x):
#         x = self.depthwise_conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)

#         x = self.pointwise_conv1(x)
#         x = self.bn2(x)
#         x = self.relu(x)

#         x = self.depthwise_conv2(x)
#         x = self.bn3(x)
#         x = self.relu(x)

#         x = self.pointwise_conv2(x)
#         x = self.bn4(x)
#         x = self.relu(x)

#         x = self.pool(x)
#         x = torch.flatten(x, 1)
#         return x
    
#     def forward(self, input_data):
#         x = self.forward_conv_layers(input_data)
#         logits = self.linear(x)
#         return logits
    

#
class depthwise_separable_CNN2(nn.Module):
    def __init__(self, spectrogram_height, spectrogram_width, output_class_number):
        super(depthwise_separable_CNN2, self).__init__()

        # Depthwise Separable Convolution layers
        # Depthwise convolution: in_channels = out_channels, groups = in_channels
        self.depthwise_conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=(3, 3), stride=(1, 2), groups=1, bias=False)
        self.pointwise_conv1 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=1, bias=False)
        
        self.depthwise_conv2 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3), stride=(1, 2), groups=8, bias=False)
        self.pointwise_conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm2d(4)
        self.bn2 = nn.BatchNorm2d(8)
        self.bn3 = nn.BatchNorm2d(8)
        self.bn4 = nn.BatchNorm2d(16)

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)

        # ここで線形層を事前に定義
        dummy_input = torch.randn(1, 1, spectrogram_height, spectrogram_width)
        x = self.forward_conv_layers(dummy_input)
        linear_input_size = x.size(1)
        self.linear = nn.Linear(linear_input_size, output_class_number)

    def forward_conv_layers(self, x):
        x = self.depthwise_conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.pointwise_conv1(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.pool(x)

        x = self.depthwise_conv2(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.pointwise_conv2(x)
        x = self.bn4(x)
        x = self.relu(x)

        x = self.pool(x)
        x = torch.flatten(x, 1)
        return x
    
    def forward(self, input_data):
        x = self.forward_conv_layers(input_data)
        logits = self.linear(x)
        return logits


class depthwise_separable_CNN4(nn.Module):
    def __init__(self, spectrogram_height, spectrogram_width, output_class_number):
        super().__init__()
        # 畳み込み層をDepthwiseとPointwiseに分割
        self.depthwise_conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=7, stride=2, padding=3, groups=1)
        self.pointwise_conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=1)
        self.relu1 = nn.ReLU()
        self.norm1 = nn.BatchNorm2d(16)

        self.depthwise_conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, stride=2, padding=2, groups=16)
        self.pointwise_conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=1)
        self.relu2 = nn.ReLU()
        self.norm2 = nn.BatchNorm2d(32)

        self.depthwise_conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, groups=32)
        self.pointwise_conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1)
        self.relu3 = nn.ReLU()
        self.norm3 = nn.BatchNorm2d(64)

        self.depthwise_conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, groups=64)
        self.pointwise_conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1)
        self.relu4 = nn.ReLU()
        self.norm4 = nn.BatchNorm2d(128)

        # Avg pooling
        self.avepool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()

        # ここで線形層を事前に定義
        dummy_input = torch.randn(1, 1, spectrogram_height, spectrogram_width)
        x = self.forward_conv_layers(dummy_input)
        linear_input_size = x.size(1)
        self.linear = nn.Linear(linear_input_size, output_class_number)

    def forward_conv_layers(self, x):
        # Block 1
        x = self.depthwise_conv1(x)  # Depthwise
        x = self.pointwise_conv1(x)  # Pointwise
        x = self.norm1(x)
        x = self.relu1(x)

        # Block 2
        x = self.depthwise_conv2(x)  # Depthwise
        x = self.pointwise_conv2(x)  # Pointwise
        x = self.norm2(x)
        x = self.relu2(x)

        # Block 3
        x = self.depthwise_conv3(x)  # Depthwise
        x = self.pointwise_conv3(x)  # Pointwise
        x = self.norm3(x)
        x = self.relu3(x)

        # Block 4
        x = self.depthwise_conv4(x)  # Depthwise
        x = self.pointwise_conv4(x)  # Pointwise
        x = self.norm4(x)
        x = self.relu4(x)

        # Average Pooling
        x = self.avepool1(x)
        x = self.flatten(x)
        return x

    def forward(self, input_data):
        x = self.forward_conv_layers(input_data)
        logits = self.linear(x)
        return logits





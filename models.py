import torch.nn as nn
import torch
import torch.nn.functional as F


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

class CNNNetwork4(nn.Module):
    def __init__(self, spectrogram_height, spectrogram_width, output_class_number):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1, # spectogram is treated as grayscale
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
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU()
        )
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(128 * spectrogram_height * spectrogram_width, output_class_number)

    def forward(self, input_data):
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        logits = self.linear(x)
        return logits
    

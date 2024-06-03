import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import Transformer

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

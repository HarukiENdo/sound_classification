import os
import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset
from tqdm import tqdm
# from augmentation_torchaudio import conduct_augmentation  # 既存のaugmentationコードをそのまま使用
from transformers import ASTFeatureExtractor

class AudioDataset_AST(Dataset):
    def __init__(self, csv_path, win_length_samples, hop_length_samples, n_mels_value, target_sample_rate, num_samples, augment=False):
        self.data = pd.read_csv(csv_path)
        self.win_length_samples = win_length_samples
        self.hop_length_samples = hop_length_samples
        self.n_mels_value = n_mels_value
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples

        # torchaudioのMelSpectrogramの変換器を定義
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.target_sample_rate, 
            n_fft=self.win_length_samples, 
            win_length=self.win_length_samples, 
            hop_length=self.hop_length_samples, 
            n_mels=self.n_mels_value
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        file_path = self.data.iloc[index]['file_path']
        signal, sr = torchaudio.load(file_path)  # torchaudioで音声ファイルをロード
        mel_spectrogram = self.mel_spectrogram(signal)
        mel_spectrogram = torchaudio.transforms.AmplitudeToDB()(mel_spectrogram)
        mel_spectrogram = mel_spectrogram.squeeze(0)
        mel_spectrogram = torch.transpose(mel_spectrogram, 0, 1)
        label = self.data.iloc[index]['label_id']

        return mel_spectrogram, torch.tensor(label)

# class AudioDataset_AST(Dataset):
#     def __init__(self, csv_path, n_mels_value, target_sample_rate, num_samples):
#         self.data = pd.read_csv(csv_path)
#         self.n_mels_value = n_mels_value
#         self.target_sample_rate = target_sample_rate
#         self.num_samples = num_samples
#         # self.feature_extractor = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
#         self.feature_extractor = ASTFeatureExtractor(feature_size=1, sampling_rate=target_sample_rate, num_mel_bins=n_mels_value, max_length=100)

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         audio_path = self.data.iloc[idx, 0]
#         label = self.data.iloc[idx, 1]
        
#         # 音声ファイルの読み込み
#         signal, sr = torchaudio.load(audio_path)
#         signal = signal.squeeze()
#         # feature_extractorでメルスペクトログラムを生成
#         mel_spectrogram = self.feature_extractor(signal, sampling_rate=self.target_sample_rate, return_tensors="pt")
#         mel_spectrogram = mel_spectrogram.input_values
#         mel_spectrogram = mel_spectrogram.squeeze(0)
#         # print(mel_spectrogram.shape)
#         # ASTFeatureExtractorはすでに適切な形式を返すため、'input_values'からデータを取得
#         return mel_spectrogram, torch.tensor(label, dtype=torch.long)

class AudioDataset(Dataset):
    def __init__(self, csv_path, win_length_samples, hop_length_samples, n_mels_value, target_sample_rate, num_samples, augment=False):
        self.data = pd.read_csv(csv_path)
        self.win_length_samples = win_length_samples
        self.hop_length_samples = hop_length_samples
        self.n_mels_value = n_mels_value
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples

        # torchaudioのMelSpectrogramの変換器を定義
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.target_sample_rate, 
            n_fft=self.win_length_samples, 
            win_length=self.win_length_samples, 
            hop_length=self.hop_length_samples, 
            n_mels=self.n_mels_value
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        file_path = self.data.iloc[index]['file_path']
        signal, sr = torchaudio.load(file_path)  # torchaudioで音声ファイルをロード
        mel_spectrogram = self.mel_spectrogram(signal)
        mel_spectrogram = torchaudio.transforms.AmplitudeToDB()(mel_spectrogram)
        label = self.data.iloc[index]['label_id']

        return mel_spectrogram, torch.tensor(label)

# AudioDataset for checking test_data
class AudioDataset_path(Dataset):
    def __init__(self, csv_path, win_length_samples, hop_length_samples, n_mels_value, target_sample_rate, num_samples):
        self.data = pd.read_csv(csv_path)
        self.win_length_samples = win_length_samples
        self.hop_length_samples = hop_length_samples
        self.n_mels_value = n_mels_value
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples

        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.target_sample_rate, 
            n_fft=self.win_length_samples, 
            win_length=self.win_length_samples, 
            hop_length=self.hop_length_samples, 
            n_mels=self.n_mels_value
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_path = self.data.iloc[idx, 0]
        label = self.data.iloc[idx, 1]
        signal, sr = torchaudio.load(file_path)

        # サンプル数が少ない場合に対処
        if signal.shape[1] < self.num_samples:
            pad_size = self.num_samples - signal.shape[1]
            signal = torch.nn.functional.pad(signal, (0, pad_size))
        else:
            signal = signal[:, :self.num_samples]

        mel_spectrogram = self.mel_spectrogram(signal)
        mel_spectrogram = torchaudio.transforms.AmplitudeToDB()(mel_spectrogram)
        # mel_spectrogram = mel_spectrogram.unsqueeze(0)

        return mel_spectrogram, torch.tensor(label), file_path




# データセット拡張も行うコード　これは使えないので無視
# class AudioDataset(Dataset):
#     def __init__(self, data_frame, win_length_samples, hop_length_samples, n_mels_value, target_sample_rate, num_samples, augment=True):
#         self.data = data_frame
#         self.win_length_samples = win_length_samples
#         self.hop_length_samples = hop_length_samples
#         self.n_mels_value = n_mels_value
#         self.target_sample_rate = target_sample_rate
#         self.num_samples = num_samples

#         # torchaudioのMelSpectrogramの変換器を定義
#         self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
#             sample_rate=self.target_sample_rate, 
#             n_fft=self.win_length_samples, 
#             win_length=self.win_length_samples, 
#             hop_length=self.hop_length_samples, 
#             n_mels=self.n_mels_value
#         )

#         # 各クラスのデータ数を計算
#         self.class_counts = self.data['label_id'].value_counts()
#         self.target_class_count = self.class_counts[0]  # environのデータ数を取得
#         print(self.class_counts)

#         # Augmentationを適用したデータを生成
#         if augment:
#             augmented_data = []
#             for label, count in self.class_counts.items():
#                 if label == 0:  # environは拡張しない
#                     continue
#                 if count < self.target_class_count:
#                     difference = self.target_class_count - count
#                     class_data = self.data[self.data['label_id'] == label]
#                     # tqdmをclass_dataの各サンプルに対して適用
#                 for _ in tqdm(range(difference), desc=f"Augmenting label {label}"):
#                         sample = class_data.sample(1, replace=True).iloc[0]
#                         signal, sr = torchaudio.load(sample['file_path'])  # torchaudioで読み込み

#                         # データ拡張を適用 (CPU上で)
#                         signal = conduct_augmentation(signal, sr)

#                         augmented_data.append({'file_path': sample['file_path'], 'label_id': label, 'augmented_signal': signal})
#             augmented_df = pd.DataFrame(augmented_data)
#             self.data = pd.concat([self.data, augmented_df], ignore_index=True)


#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, index):
#         if 'augmented_signal' in self.data.columns:
#             signal = self.data.iloc[index]['augmented_signal']
#         else:
#             file_path = self.data.iloc[index]['file_path']
#             signal, sr = torchaudio.load(file_path)  # torchaudioで音声ファイルをロード

#         # サンプル数が少ない場合に対処
#         if signal.shape[1] < self.num_samples:
#             pad_size = self.num_samples - signal.shape[1]
#             signal = torch.nn.functional.pad(signal, (0, pad_size))
#         else:
#             signal = signal[:, :self.num_samples]

#         # MelSpectrogramを計算 (CPU上で実行)
#         mel_spectrogram = self.mel_spectrogram(signal)
#         mel_spectrogram = torchaudio.transforms.AmplitudeToDB()(mel_spectrogram)
#         mel_spectrogram = mel_spectrogram.unsqueeze(0)  # CNNのためにチャンネル次元を追加
#         label = self.data.iloc[index]['label_id']

#         return mel_spectrogram, torch.tensor(label)

import csv
import librosa
import numpy as np
from torch.utils.data import Dataset
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from tqdm import tqdm
import pandas as pd
import torch
import torchaudio
# from transformers import ASTConfig, ASTModel
from transformers import ASTFeatureExtractor
# from transformers import AutoModelForAudioClassification
import torchaudio
from augmentation import conduct_augmentation 

class AudioDataset_AST(Dataset):
    def __init__(self, csv_path, win_length_samples, hop_length_samples, n_mels_value, target_sample_rate, num_samples, device):
        self.data = pd.read_csv(csv_path)
        self.win_length_samples = win_length_samples
        self.hop_length_samples = hop_length_samples
        self.n_mels_value = n_mels_value
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples
        self.device = device
        # self.feature_extractor = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        self.feature_extractor = ASTFeatureExtractor(feature_size = 1, sampling_rate = 16000, n_mel_bins = 30, max_length = 201, padding_value = 0, do_normalize = True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio_path = self.data.iloc[idx, 0]
        label = self.data.iloc[idx, 1]
        
        # 音声ファイルの読み込み
        signal, sr = torchaudio.load(audio_path)
        signal = signal.squeeze()
        # feature_extractorでメルスペクトログラムを生成
        mel_spectrogram = self.feature_extractor(signal, sampling_rate=self.target_sample_rate, return_tensors="pt")
        mel_spectrogram = mel_spectrogram.input_values
        mel_spectrogram = mel_spectrogram.squeeze(0)
        # ASTFeatureExtractorはすでに適切な形式を返すため、'input_values'からデータを取得
        return mel_spectrogram, torch.tensor(label, dtype=torch.long)

    
class AudioDataset(Dataset): #normal
    def __init__(self, csv_path, win_length_samples, hop_length_samples, n_mels_value, target_sample_rate, num_samples, device):
        self.data = pd.read_csv(csv_path)
        # self.data = data_frame
        self.win_length_samples = win_length_samples
        self.hop_length_samples = hop_length_samples
        self.n_mels_value = n_mels_value
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples
        self.device = device

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio_path = self.data.iloc[idx, 0]
        label = self.data.iloc[idx, 1]
        signal, sr = librosa.load(audio_path, sr=self.target_sample_rate)
        mel_spectrogram = librosa.feature.melspectrogram(y=signal, sr=sr, n_fft=self.win_length_samples, win_length=self.win_length_samples, hop_length=self.hop_length_samples, n_mels=self.n_mels_value)
        mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
        mel_spectrogram = np.expand_dims(mel_spectrogram, axis=0)
        return torch.tensor(mel_spectrogram, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

    
#add AudioDataset for checking test_data
class AudioDataset_path(Dataset):
    def __init__(self, data_frame, win_length_samples, hop_length_samples, n_mels_value, target_sample_rate, num_samples, device):
        # self.data = pd.read_csv(csv_path)
        self.data = data_frame
        self.win_length_samples = win_length_samples
        self.hop_length_samples = hop_length_samples
        self.n_mels_value = n_mels_value
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples
        self.device = device

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio_path = self.data.iloc[idx, 0]
        label = self.data.iloc[idx, 1]
        signal, sr = librosa.load(audio_path, sr=self.target_sample_rate)
        mel_spectrogram = librosa.feature.melspectrogram(y=signal, sr=sr, n_fft=self.win_length_samples, win_length=self.win_length_samples, hop_length=self.hop_length_samples, n_mels=self.n_mels_value)
        mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
        mel_spectrogram = np.expand_dims(mel_spectrogram, axis=0)
        return torch.tensor(mel_spectrogram, dtype=torch.float32), torch.tensor(label, dtype=torch.long), audio_path

class AudioDataset_transformer(Dataset): #transformer用 次元追加が必要ない
    def __init__(self, csv_path, win_length_samples, hop_length_samples, n_mels_value, target_sample_rate, num_samples, device):
        self.data = pd.read_csv(csv_path)
        self.win_length_samples = win_length_samples
        self.hop_length_samples = hop_length_samples
        self.n_mels_value = n_mels_value
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples
        self.device = device

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio_path = self.data.iloc[idx, 0]
        label = self.data.iloc[idx, 1]
        signal, sr = librosa.load(audio_path, sr=self.target_sample_rate)
        mel_spectrogram = librosa.feature.melspectrogram(y=signal, sr=sr, n_fft=self.win_length_samples, win_length=self.win_length_samples, hop_length=self.hop_length_samples, n_mels=self.n_mels_value)
        mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
        return torch.tensor(mel_spectrogram, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

#実行できるが、メモリを使いすぎてしまうため使用しない
# class AudioDataset_aug(Dataset):
#     def __init__(self, data_frame, win_length_samples, hop_length_samples, n_mels_value, target_sample_rate, num_samples, device, augment=True):
#         self.data = data_frame
#         self.win_length_samples = win_length_samples
#         self.hop_length_samples = hop_length_samples
#         self.n_mels_value = n_mels_value
#         self.target_sample_rate = target_sample_rate
#         self.num_samples = num_samples
#         self.device = device

#         # 各クラスのデータ数を計算
#         self.class_counts = self.data['label_id'].value_counts()
#         self.target_class_count = self.class_counts[0]  # environのデータ数を取得
        
#         # Augmentationを適用したデータを生成
#         if augment:
#             augmented_data = []
#             for label, count in tqdm(self.class_counts.items(), desc="Augmenting data"):
#                 if label == 0:  # environは拡張しない
#                     continue
#                 if count < self.target_class_count:
#                     difference = self.target_class_count - count
#                     class_data = self.data[self.data['label_id'] == label]
#                     for _ in range(difference):
#                         sample = class_data.sample(1, replace=True).iloc[0]
#                         signal, sr = librosa.load(sample['file_path'], sr=self.target_sample_rate)
#                         signal = conduct_augmentation(signal, sr)
#                         augmented_data.append({'file_path': sample['file_path'], 'label_id': label, 'augmented_signal': signal})
#             augmented_df = pd.DataFrame(augmented_data)
#             self.data = pd.concat([self.data, augmented_df], ignore_index=True)

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, index):
#         if 'augmented_signal' in self.data.columns:
#             signal = self.data.iloc[index]['augmented_signal']
#             sr = self.target_sample_rate
#         else:
#             file_path = self.data.iloc[index]['file_path']
#             signal, sr = librosa.load(file_path, sr=self.target_sample_rate)

#         # signal = signal.astype(np.float32)
        
#         # メルスペクトログラムの生成
#         mel_spectrogram = librosa.feature.melspectrogram(y=signal, sr=sr, n_fft=self.win_length_samples, 
#                                                          win_length=self.win_length_samples, hop_length=self.hop_length_samples, 
#                                                          n_mels=self.n_mels_value)
#         mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

#         # チャンネル次元を追加してテンソルに変換
#         mel_spectrogram = torch.tensor(mel_spectrogram).unsqueeze(0)

#         label = self.data.iloc[index]['label_id']
#         return torch.tensor(mel_spectrogram, dtype=torch.float32), torch.tensor(label, dtype=torch.long)




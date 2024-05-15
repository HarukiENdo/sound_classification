import csv
import librosa
import numpy as np
from torch.utils.data import Dataset
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from tqdm import tqdm
import pandas as pd
import torch

# class AudioDataset(Dataset):
#     def __init__(self,
#                 csv_path,
#                 win_length_samples,
#                 hop_length_samples,
#                 n_mels_value,
#                 target_sample_rate,
#                 num_samples,
#                 device):
#         self.csv_path = csv_path
#         self.audio_files = []
#         self.labels = []
#         self.device = device
#         self.win_length_samples = win_length_samples
#         self.hop_length_samples = hop_length_samples
#         self.n_mels_value = n_mels_value
#         self.target_sample_rate = target_sample_rate
#         self.num_samples = num_samples

#         with open(self.csv_path, mode='r', newline='', encoding='utf-8') as file:
#             reader = csv.DictReader(file)
#             for row in reader:
#                 self.audio_files.append(row['filepath'])
#                 self.labels.append(int(row['label_id']))

#         self.load_data()
            

#     def process_path(self, path):
#         audio_file = path
#         signal,sr=librosa.load(audio_file,sr=16000)
#         signal = librosa.feature.melspectrogram(y=signal,sr=sr, n_fft=self.win_length_samples,win_length=self.win_length_samples, hop_length=self.hop_length_samples, n_mels=self.
#         n_mels_value)
#         signal = librosa.power_to_db(signal, ref=np.max)
#         signal = np.expand_dims(signal, axis=0)
#         label = self.labels[self.audio_files.index(path)]
#         return signal, label

#     def load_data(self):
#         print("Reading Audio Files")
#         with ThreadPoolExecutor(max_workers=4) as executor:  
#             results = list(tqdm(executor.map(self.process_path, self.audio_files), total=len(self.audio_files)))
#         self.signals, self.labels = zip(*results)

#     def __len__(self):
#         return len(self.audio_files)

#     def __getitem__(self, idx):
#         return self.signals[idx], self.labels[idx]
    
class AudioDataset(Dataset):
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
        mel_spectrogram = np.expand_dims(mel_spectrogram, axis=0)
        return torch.tensor(mel_spectrogram, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


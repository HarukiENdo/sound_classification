import os
import subprocess
import argparse
import torch
import torchaudio
import sys
# 上位のディレクトリにある場合
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from model.models import CNNNetwork4, depthwise_CNN2

def define_inshape(file_path, batch_size, sample_rate, duration_ms, window_size, n_mels):
    signal_sample, sr = torchaudio.load(file_path)
    NUM_SAMPLES = int(sample_rate*duration_ms/1000)
    win_length_ms = window_size
    hop_length_ms = int(win_length_ms/3)
    win_length_samples = int(NUM_SAMPLES*win_length_ms/1000)
    hop_length_samples = int(NUM_SAMPLES*hop_length_ms/1000)
    mel_spectrogram_transform = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate,n_fft=win_length_samples,win_length=win_length_samples,hop_length=hop_length_samples,n_mels=n_mels)
    mel_spectrogram = mel_spectrogram_transform(signal_sample)  # メルスペクトログラムを生成
    mel_spectrogram = torchaudio.transforms.AmplitudeToDB()(mel_spectrogram)  # dBスケールに変換
    print("Shape of sample spectrogram: ", mel_spectrogram.shape)
    spectrogram_height = mel_spectrogram.shape[1]
    spectrogram_width = mel_spectrogram.shape[2] 
    return spectrogram_height, spectrogram_width

def convert_torch_to_onnx(model, onnx_filepath, dummy_input):
    torch.onnx.export(
    model,                        # PyTorch model
    dummy_input,                  # Dummy input tensor
    onnx_filepath,                 # Output file name for the ONNX model
    export_params=True,           # Store trained parameters in the ONNX model
    opset_version=11,             # ONNX opset version, generally 11 or 13 is recommended
    do_constant_folding=True,     # Whether to execute constant folding for optimization
    input_names=['input'],        # Names for the model inputs (optional)
    output_names=['output'],      # Names for the model outputs (optional)
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}  # Dynamic batch size (optional)
)
    
def main(args):
    output_class_number = 3
    batch_size = args.batch_size
    onnx_filepath = args.onnx_filepath
    model_weight = args.model_weight
    audio_sample_path = "sample.wav"
    spectrogram_height, spectrogram_width = define_inshape(audio_sample_path, batch_size=args.batch_size, sample_rate=args.sample_rate, duration_ms=args.duration_ms, window_size=args.window_size, n_mels=args.n_mels)
    if args.model == 'CNNNetwork4':
        print("model cnn_network4")
        model = CNNNetwork4(spectrogram_height, spectrogram_width, output_class_number)
        model.load_state_dict(torch.load(model_weight))
        model.eval()
        model = model.cpu()
    elif args.model == 'depthwise_CNN2':
        print("model depthwise_CNN2")
        model = depthwise_CNN2(spectrogram_height, spectrogram_width, output_class_number)
        model.load_state_dict(torch.load(model_weight))
        model.eval()
        model = model.cpu()
    convert_torch_to_onnx(model, onnx_filepath, dummy_input = torch.randn(args.batch_size, 1, spectrogram_height, spectrogram_width))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=512, help='batch size')
    parser.add_argument('--n_mels', type=int, default=30, help='n mels')
    parser.add_argument('--sample_rate', type=int, default=16000, help='sample rate')
    parser.add_argument('--duration_ms', type=int, default=1000, help='duration ms')
    parser.add_argument('--window_size', type=int, default=30, help='window size')
    parser.add_argument('--model', type=str, default='CNNNetwork4', help='Convert model')
    parser.add_argument('--onnx_filepath', type=str, default='../model/model_1026.onnx', help='onnx filepath')
    parser.add_argument('--model_weight', type=str, default='best_acc.pth', help='model weight')
    args = parser.parse_args()
    main(args)


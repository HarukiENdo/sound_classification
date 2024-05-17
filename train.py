import numpy as np
import os
import glob
import argparse
import torch
import torchaudio
import librosa
import librosa.display 
import matplotlib.pyplot as plt
import IPython.display as ipd
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
from torch.utils.data import WeightedRandomSampler
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from torchvision.models import resnet18, resnet34, resnet50
import time
import wandb
import csv
from tqdm import tqdm
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import numpy as np
import os
from dataset import AudioDataset
from utils import *
import random
import torch.cuda.amp as amp
from loss import FocalLoss
from contextlib import redirect_stdout
import yaml
from models import CNNNetwork1, CNNNetwork2, CNNNetwork3, CNNNetwork4

print("-------------------Cuda check-------------------")
print("Cuda availability: ",torch.cuda.is_available())
print("Torch version: ", torch.__version__)


def collate_fn(batch):
    inputs, targets = zip(*batch)
    inputs = torch.stack(inputs)
    targets = torch.stack(targets)
    return inputs, targets




def train(args):
    NUM_SAMPLES = int(args.sample_rate*args.duration_ms/1000)
    CLASS_NAMES = ['car','cut','environ','fruit','leaf','talk','truck','unknown','walk']
    #------------Parameters setup -------------------------
    win_length_ms = args.window_size
    hop_length_ms = int(win_length_ms/3)
    win_length_samples = int(NUM_SAMPLES*win_length_ms/1000)
    hop_length_samples = int(NUM_SAMPLES*hop_length_ms/1000)
    #------------Name setup --------------------------------
    output_dir = f"./experiment/{args.project_name}/{args.model}_optimizer_{args.optimizer}_loss_{args.loss}_lr{args.learning_rate}_n_mels{args.n_mels}_window_size{args.window_size}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    #------------Training setup ----------------------------
    device = device_check(display_device = True)
    train_dataset = AudioDataset(csv_path='/Corpus3/crime_prevention_sound/train.csv',win_length_samples=win_length_samples,hop_length_samples=hop_length_samples,n_mels_value=args.n_mels, target_sample_rate=args.sample_rate, num_samples=NUM_SAMPLES, device=device) 
    val_dataset = AudioDataset(csv_path='/Corpus3/crime_prevention_sound/val.csv',win_length_samples=win_length_samples,hop_length_samples=hop_length_samples,n_mels_value=args.n_mels, target_sample_rate=args.sample_rate, num_samples=NUM_SAMPLES, device=device) 
    test_dataset = AudioDataset(csv_path='/Corpus3/crime_prevention_sound/test.csv',win_length_samples=win_length_samples,hop_length_samples=hop_length_samples,n_mels_value=args.n_mels, target_sample_rate=args.sample_rate, num_samples=NUM_SAMPLES, device=device)    
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True, num_workers=8, pin_memory=True)
    val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False, num_workers=8, pin_memory=False)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False, num_workers=8, pin_memory=False)
    
    # サンプルデータの取得
    audio_sample_path = "/Corpus3/crime_prevention_sound/dataset/dataset_20231107/environ/20231101_2_終了_00009672.wav"
    signal_sample, sr = librosa.load(audio_sample_path, sr=16000)
    mel_spectrogram = librosa.feature.melspectrogram(y=signal_sample, sr=sr, n_fft=win_length_samples, win_length=win_length_samples, hop_length=hop_length_samples, n_mels=args.n_mels)
    mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)  
    mel_spectrogram = np.expand_dims(mel_spectrogram, axis=0)
    print("Shape of sample spectrogram: ", mel_spectrogram.shape)
    
    spectrogram_height = mel_spectrogram.shape[1]
    spectrogram_width = mel_spectrogram.shape[2]
    output_class_number = len(CLASS_NAMES)

    # --------------Model setup---------------------------
    if args.model == "resnet18":
        print("model resnet18")
        model = resnet18(pretrained=True)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.fc = nn.Linear(512,output_class_number)
    elif args.model == "resnet34": 
        print("model resnet34")
        model = resnet34(pretrained=True)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.fc = nn.Linear(512,output_class_number)
    elif args.model == "resnet50":
        print("model resnet50")
        model = resnet50(pretrained=True)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.fc = nn.Linear(2048,output_class_number)
    elif args.model == "cnn_network1":
        print("model cnn_network1")
        model = CNNNetwork1(spectrogram_height, spectrogram_width, output_class_number)
    elif args.model == "cnn_network2":
        print("model cnn_network2")
        model = CNNNetwork2(spectrogram_height, spectrogram_width, output_class_number)
    elif args.model == "cnn_network3":
        print("model cnn_network3")
        model = CNNNetwork3(spectrogram_height, spectrogram_width, output_class_number)
    elif args.model == "cnn_network4":
        print("model cnn_network4")
        model = CNNNetwork4(spectrogram_height, spectrogram_width, output_class_number)
      
    model.to(device)
    summary(model, input_size=(1, spectrogram_height, spectrogram_width))

    # --------------Training setup---------------------------
    if args.loss == "cross_entropy":
        loss_fn = nn.CrossEntropyLoss()
    elif args.loss == "focal_loss":
        loss_fn = FocalLoss()

    if args.optimizer == "adam":
        optimiser = torch.optim.Adam(model.parameters(), lr=args.learning_rate)    
    elif args.optimizer == "adamw":
        optimiser = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)

    steps_per_epoch = len(train_data_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimiser, max_lr=args.learning_rate, 
                                                    steps_per_epoch=steps_per_epoch, epochs=args.epochs)

    if args.amp:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None
    # --------------Summary--------------------------------
    summary_file_path = os.path.join(output_dir, "summary.txt")
    args_file_path = os.path.join(output_dir, "args.yaml")
    
    with open(args_file_path, 'w') as f:
        args_dict = vars(args)
        yaml.dump(args_dict, f)
    
    with open(summary_file_path, 'w') as f:
        with redirect_stdout(f):
            summary(model, input_size=(1, spectrogram_height, spectrogram_width))
    
    print(args_dict)
    # --------------Training loop---------------------------
    epochs = args.epochs
    loss_training_epochs = []
    loss_validation_epochs = []
    exec_time_start_time = time.time()
    early_stopping = EarlyStopping(patience=7, verbose=True)
    best_val_loss = float('inf')
    best_val_acc = 0.0
    for i in range(epochs):
        print(f"Epoch {i+1}")    
        loss_training_single_epoch_array = []
        y_true_train, y_pred_train = [], []
        for batch_idx, (input, target) in enumerate(tqdm(train_data_loader)):
            optimiser.zero_grad()
            input, target = input.to(device, dtype=torch.float32), target.to(device)
            if args.amp:
                with torch.cuda.amp.autocast():
                    prediction = model(input)
            else:
                prediction = model(input)
            loss = loss_fn(prediction, target)
            if args.amp:
                scaler.scale(loss).backward()
                scaler.step(optimiser)
                scaler.update()
            else:
                loss.backward()
                optimiser.step()
            scheduler.step()
            loss_training_single_epoch_array.append(loss.item())
            y_true_train.extend(target.cpu().numpy())
            y_pred_train.extend(torch.argmax(prediction, dim=1).cpu().numpy())
            # イテレーション単位で精度を計算
            correct_predictions = (torch.argmax(prediction, dim=1) == target).sum().item()
            accuracy = correct_predictions / input.size(0)
            # イテレーション単位でログを取る
            if args.wandb:
                wandb.log({
                    "iteration": i * len(train_data_loader) + batch_idx,
                    "train_loss_iter": loss.item(),
                    "train_acc_iter": accuracy * 100,
                    "lr": optimiser.param_groups[0]['lr']
                })
        exec_time = time.time() - exec_time_start_time 
        loss_training_single_epoch = np.array(loss_training_single_epoch_array).mean()
        loss_training_epochs.append(loss_training_single_epoch)
        classification_report_train = classification_report(y_true_train, y_pred_train, target_names=CLASS_NAMES, output_dict=True)
    
        # Validation
        loss_validation_single_epoch_array = []
        y_true_val, y_pred_val, y_pred_val_proba = [], [], []
        for input, target in tqdm(val_data_loader):
            input, target = input.to(device,dtype=torch.float32), target.to(device)
            prediction = model(input)
            loss = loss_fn(prediction, target)
            loss_validation_single_epoch_array.append(loss.item())
            # Convert tensor predictions to numpy arrays
            y_true_val.extend(target.cpu().numpy())
            y_pred_val_proba.extend(prediction.cpu().detach().numpy()) #TODO: double check if original array is modified
            y_pred_val.extend(torch.argmax(prediction, dim=1).cpu().numpy())

        loss_validation_single_epoch = np.array(loss_validation_single_epoch_array).mean()
        loss_validation_epochs.append(loss_validation_single_epoch)
    
        classification_report_val = classification_report(y_true_val, y_pred_val, target_names=CLASS_NAMES, output_dict=True)
        # --------------Save model---------------------------        
        if loss_validation_single_epoch < best_val_loss:
            output_model_path_loss = f"{output_dir}/best_loss.pth"
            best_val_loss = loss_validation_single_epoch
            torch.save(model.state_dict(), output_model_path_loss)
            print("Trained feed forward net saved at: ", output_model_path_loss)
        if classification_report_val['accuracy'] > best_val_acc:
            output_model_path_acc = f"{output_dir}/best_acc.pth"
            best_val_acc = classification_report_val['accuracy']
            torch.save(model.state_dict(), output_model_path_acc)
            print("Trained feed forward net saved at: ", output_model_path_acc)

        print("Validation Classification Report:")
        print(classification_report_val)

        print(f"Training accuracy : {classification_report_train['accuracy']} ; Training loss : {loss_training_single_epoch}  ")
        print(f"Validation accuracy : {classification_report_val['accuracy']} ; Validation loss : {loss_validation_single_epoch} ")        
        print("---------------------------")

    # --------------Wandb log---------------------------  
        if args.wandb:
            wandb.log({
            "epoch": i+1,
            "exec_time": exec_time,
            "train_loss": loss_training_single_epoch,
            "train_acc":classification_report_train['accuracy'] * 100,
            "val_loss": loss_validation_single_epoch,
            "val_acc":classification_report_val['accuracy'] * 100,
            'precision_cut':classification_report_val['cut']['precision'],
            'recall_cut':classification_report_val['cut']['recall'],
            'f1-score_cut':classification_report_val['cut']['f1-score'],
            'precision_car':classification_report_val['car']['precision'],
            'recall_car':classification_report_val['car']['recall'],
            'f1-score_car':classification_report_val['car']['f1-score'],
            'precision_environ':classification_report_val['environ']['precision'],
            'recall_environ':classification_report_val['environ']['recall'],
            'f1-score_environ':classification_report_val['environ']['f1-score'],
            'precision_fruit':classification_report_val['fruit']['precision'],
            'recall_fruit':classification_report_val['fruit']['recall'],
            'f1-score_fruit':classification_report_val['fruit']['f1-score'],
            'precision_leaf':classification_report_val['leaf']['precision'],
            'recall_leaf':classification_report_val['leaf']['recall'],
            'f1-score_leaf':classification_report_val['leaf']['f1-score'],
            'precision_talk':classification_report_val['talk']['precision'],
            'recall_talk':classification_report_val['talk']['recall'],
            'f1-score_talk':classification_report_val['talk']['f1-score'],
            'precision_truck':classification_report_val['truck']['precision'],
            'recall_truck':classification_report_val['truck']['recall'],
            'f1-score_truck':classification_report_val['truck']['f1-score'],
            'precision_unknown':classification_report_val['unknown']['precision'],
            'recall_unknown':classification_report_val['unknown']['recall'],
            'f1-score_unknown':classification_report_val['unknown']['f1-score'],
            'precision_walk':classification_report_val['walk']['precision'],
            'recall_walk':classification_report_val['walk']['recall'],
            'f1-score_walk':classification_report_val['walk']['f1-score']
                
            })

            wandb.log({"conf_mat" : wandb.plot.confusion_matrix(probs=None,y_true=y_true_val,preds=y_pred_val,
                    class_names=CLASS_NAMES)})
            wandb.log({"roc" : wandb.plot.roc_curve(y_true_val, y_pred_val_proba,
                    labels=CLASS_NAMES)})
            wandb.log({"pr" : wandb.plot.pr_curve(y_true_val, y_pred_val_proba,
                    labels=CLASS_NAMES)})
        # Early stopping check
        early_stopping(loss_validation_single_epoch)
        if early_stopping.early_stop:
            print("early stopping")
            break

    print("Finished training")
    print("---------------------------")
    print("---------------------------")
    print("---------------------------")
    # --------------Test---------------------------
    for output_model_path in [output_model_path_loss, output_model_path_acc]:
        model.load_state_dict(torch.load(output_model_path))
        model.eval()
        phase = output_model_path.split("/")[-1].split(".")[0]
        loss_test_single_epoch_array = []
        y_true_test, y_pred_test = [], []
        for input, target in test_data_loader:
            input, target = input.to(device,dtype=torch.float32), target.to(device)
            # calculate loss
            prediction = model(input)
            loss = loss_fn(prediction, target)
        
            loss_test_single_epoch_array.append(loss.item())
        
            y_true_test.extend(target.cpu().numpy())
            y_pred_test.extend(torch.argmax(prediction, dim=1).cpu().numpy())
        loss_test_single_epoch = np.array(loss_test_single_epoch_array).mean()
        
        print("\nTest Report:")
        classification_report_test = classification_report(y_true_test, y_pred_test, target_names=CLASS_NAMES)
        print(classification_report_test)
        classification_report_test = classification_report(y_true_test, y_pred_test, target_names=CLASS_NAMES, output_dict=True)
        print(classification_report_test)
        print(f"\n{phase} Test accuracy : {classification_report_test['accuracy']} ; Test loss : {loss_test_single_epoch}  ")
        print("---------------------------")
        if args.wandb:
            wandb.log({
                "test_loss": loss_test_single_epoch,
                "test_acc": classification_report_test['accuracy'] * 100,
                'precision_cut':classification_report_test['cut']['precision'],
                'recall_cut':classification_report_test['cut']['recall'],
                'f1-score_cut':classification_report_test['cut']['f1-score'],
                'precision_car':classification_report_test['car']['precision'],
                'recall_car':classification_report_test['car']['recall'],
                'f1-score_car':classification_report_test['car']['f1-score'],
                'precision_environ':classification_report_test['environ']['precision'],
                'recall_environ':classification_report_test['environ']['recall'],
                'f1-score_environ':classification_report_test['environ']['f1-score'],
                'precision_fruit':classification_report_test['fruit']['precision'],
                'recall_fruit':classification_report_test['fruit']['recall'],
                'f1-score_fruit':classification_report_test['fruit']['f1-score'],
                'precision_leaf':classification_report_test['leaf']['precision'],
                'recall_leaf':classification_report_test['leaf']['recall'],
                'f1-score_leaf':classification_report_test['leaf']['f1-score'],
                'precision_talk':classification_report_test['talk']['precision'],
                'recall_talk':classification_report_test['talk']['recall'],
                'f1-score_talk':classification_report_test['talk']['f1-score'],
                'precision_truck':classification_report_test['truck']['precision'],
                'recall_truck':classification_report_test['truck']['recall'],
                'f1-score_truck':classification_report_test['truck']['f1-score'],
                'precision_unknown':classification_report_test['unknown']['precision'],
                'recall_unknown':classification_report_test['unknown']['recall'],
                'f1-score_unknown':classification_report_test['unknown']['f1-score'],
                'precision_walk':classification_report_test['walk']['precision'],
                'recall_walk':classification_report_test['walk']['recall'],
                'f1-score_walk':classification_report_test['walk']['f1-score'],
                'conf_mat': wandb.plot.confusion_matrix(probs=None,y_true=y_true_test,preds=y_pred_test,
                    class_names=CLASS_NAMES),
                'roc': wandb.plot.roc_curve(y_true_test, y_pred_test,
                    labels=CLASS_NAMES),
                'pr': wandb.plot.pr_curve(y_true_test, y_pred_test,
                    labels=CLASS_NAMES),
            })



def main(args):
    seed_everything(args.seed)
    if args.wandb:
      wandb.init(entity="hideaki_yjm", name=f"{args.model}_optimizer_{args.optimizer}_loss_{args.loss}_lr{args.learning_rate}_n_mels{args.n_mels}_window_size{args.window_size}", project=args.project_name, config=args)
    train(args)
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
      '--learning_rate','-lr' ,
      type=float,
      default=0.0005,
      help='Learning rate of model',)
    parser.add_argument(
        '--n_mels',
        type=int,
        default=30,
        help='Number of mels',)
    parser.add_argument(
        '--window_size',
        type=int,
        default=30,
        help='Window size of mel spectrogram',)
    parser.add_argument(
        '--epochs',
        '-e',
        type=int,
        default=10,
        help='Number of epochs',)
    parser.add_argument(
      '--project_name',
      type=str,
      default="debug",
      help='Project name',)
    parser.add_argument(
      '--seed',
      type=int,
      default=3407,
      help='Random seed',)
    parser.add_argument(
      '--batch_size',
      type=int,
      default=1024,
      help='Batch size',)
    parser.add_argument(
      '--sample_rate',
      type=int,
      default=16000,
      help='Sample rate',)
    parser.add_argument(
      '--duration_ms',
      type=int,
      default=1000,
      help='Duration of audio in milliseconds',)
    parser.add_argument(
      '--model',
      type=str,
      default="resnet18",
      help='Model name',)
    parser.add_argument('--amp', action='store_true', default=True, help='Use mixed precision')
    parser.add_argument('--loss', type=str, default="cross_entropy", help='Loss function')
    parser.add_argument('--optimizer', type=str, default="adamw", help='Optimizer')
    parser.add_argument('--wandb', action='store_true', help='Use wandb for logging')
    args = parser.parse_args()
    main(args)

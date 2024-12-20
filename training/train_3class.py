import numpy as np
import os
import glob
import argparse
import torch
import torchaudio
import matplotlib.pyplot as plt
import IPython.display as ipd
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
from torch.utils.data import WeightedRandomSampler
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from torchvision.models import resnet18, ResNet18_Weights, resnet34, ResNet34_Weights, resnet50, ResNet50_Weights
from torchvision.models import resnext50_32x4d, ResNeXt50_32X4D_Weights, resnext101_32x8d, ResNeXt101_32X8D_Weights #ResNextのimport
from torchvision.models import mobilenet_v2
import time
import wandb
import csv
from tqdm import tqdm
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import random
from contextlib import redirect_stdout
import yaml
import torch.cuda.amp as amp
import torch.nn.functional as F
import multiprocessing
import sys
# プロジェクトのルートディレクトリを追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from datasetclass.dataset_torchaudio import AudioDataset, AudioDataset_path ##
from util.utils import * ##
from util.loss import FocalLoss ##
from model.models import CNNNetwork1, CNNNetwork2, CNNNetwork3, CNNNetwork4, Student_s, depthwise_CNN2, depthwise_separable_CNN2, depthwise_separable_CNN4 ##
print("-------------------Cuda check-------------------")
print("Cuda availability: ",torch.cuda.is_available())
print("Torch version: ", torch.__version__)


def collate_fn(batch):
    inputs, targets = zip(*batch)
    inputs = torch.stack(inputs)
    targets = torch.stack(targets)
    return inputs, targets

def collate_fn_test(batch):
    inputs, targets, file_paths = zip(*batch)
    inputs = torch.stack(inputs)
    targets = torch.stack(targets)
    return inputs, targets, file_paths


def train(args):
    NUM_SAMPLES = int(args.sample_rate*args.duration_ms/1000)
    # CLASS_NAMES = ['car','cut','environ','fruit','leaf','talk','truck','unknown','walk']
    CLASS_NAMES = ['environ','talk','walk']
    #------------Parameters setup -------------------------
    win_length_ms = args.window_size
    hop_length_ms = int(win_length_ms/3)
    win_length_samples = int(NUM_SAMPLES*win_length_ms/1000)
    hop_length_samples = int(NUM_SAMPLES*hop_length_ms/1000)
    #------------Name setup --------------------------------
    output_dir = f"./experiment/{args.project_name}/{args.model}_optimizer_{args.optimizer}_loss_{args.loss}_scheduler_{args.scheduler}_lr{args.learning_rate}_n_mels{args.n_mels}_window_size{args.window_size}_batch_size{args.batch_size}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    #------------Training setup ----------------------------
    device = device_check(display_device = True)
    print("Device: ", device)
    n_gpu = torch.cuda.device_count()
    print(f"Available GPUs: {n_gpu}")
    print("Number of CPU: ", multiprocessing.cpu_count())

    train_dataset = AudioDataset(csv_path='train.csv',win_length_samples=win_length_samples,hop_length_samples=hop_length_samples,n_mels_value=args.n_mels, target_sample_rate=args.sample_rate, num_samples=NUM_SAMPLES) 
    val_dataset = AudioDataset(csv_path='val.csv',win_length_samples=win_length_samples,hop_length_samples=hop_length_samples,n_mels_value=args.n_mels, target_sample_rate=args.sample_rate, num_samples=NUM_SAMPLES) 
    test_dataset = AudioDataset_path(csv_path='test.csv',win_length_samples=win_length_samples,hop_length_samples=hop_length_samples,n_mels_value=args.n_mels, target_sample_rate=args.sample_rate, num_samples=NUM_SAMPLES)    

    print(f"Number of training dataset: {len(train_dataset)}")
    print(f"Number of validation dataset: {len(val_dataset)}")
    print(f"Number of test dataset: {len(test_dataset)}")
    
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True, num_workers=4, pin_memory=True)
    val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False, num_workers=4, pin_memory=False)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate_fn_test, shuffle=False, num_workers=4, pin_memory=False)
    # サンプルデータの取得
    audio_sample_path = "sample.wav"
    signal_sample, sr = torchaudio.load(audio_sample_path)
    mel_spectrogram_transform = torchaudio.transforms.MelSpectrogram(sample_rate=args.sample_rate,n_fft=win_length_samples,win_length=win_length_samples,hop_length=hop_length_samples,n_mels=args.n_mels)
    mel_spectrogram = mel_spectrogram_transform(signal_sample)  # メルスペクトログラムを生成
    mel_spectrogram = torchaudio.transforms.AmplitudeToDB()(mel_spectrogram)  # dBスケールに変換  
    #librosaの場合
    # signal_sample, sr = librosa.load(audio_sample_path, sr=args.sample_rate)
    # mel_spectrogram = librosa.feature.melspectrogram(y=signal_sample, sr=sr, n_fft=win_length_samples, win_length=win_length_samples, hop_length=hop_length_samples, n_mels=args.n_mels)
    # mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)  
    # mel_spectrogram = np.expand_dims(mel_spectrogram, axis=0)

    print("Shape of sample spectrogram: ", mel_spectrogram.shape)
    
    spectrogram_height = mel_spectrogram.shape[1]
    spectrogram_width = mel_spectrogram.shape[2]
    output_class_number = len(CLASS_NAMES)

    # --------------Model setup---------------------------
    if args.model == "resnet18":
        print("model resnet18")
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.fc = nn.Linear(512,output_class_number)
    elif args.model == "resnet34": 
        print("model resnet34")
        model = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.fc = nn.Linear(512,output_class_number)
    elif args.model == "resnet50":
        print("model resnet50")
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.fc = nn.Linear(2048,output_class_number)
    elif args.model == "resnext50":
        model = resnext50_32x4d(weights=ResNeXt50_32X4D_Weights.IMAGENET1K_V1)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.fc = nn.Linear(2048,output_class_number)
    elif args.model == "resnext101":
        model = resnext101_32x8d(weights=ResNeXt101_32X8D_Weights.IMAGENET1K_V1)
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
    elif args.model == "Student_s":
        print("model Student_s")
        model = Student_s(spectrogram_height, spectrogram_width, output_class_number)
    elif args.model == "depthwise_CNN2":
        print("model depthwise_CNN2")
        model = depthwise_CNN2(spectrogram_height, spectrogram_width, output_class_number)
    elif args.model == "depthwise_separable_CNN2":
        print("model depthwise_separable_CNN2")
        model = depthwise_separable_CNN2(spectrogram_height, spectrogram_width, output_class_number)
    elif args.model == "depthwise_separable_CNN4":
        print("model depthwise_separable_CNN4")
        model = depthwise_separable_CNN4(spectrogram_height, spectrogram_width, output_class_number)
    elif args.model == "MobileNet_v2":
        print("model MobileNet_v2")
        # 事前学習済みモデルをロード
        model = mobilenet_v2(pretrained=True)
        # 初期層を1チャネル入力用に置き換え
        model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        model.classifier[1] = nn.Linear(model.last_channel, output_class_number)

    # DataParallelを使ってモデルを複数GPUで並列化する場合
    if args.data_parallel:
        if n_gpu > 1:
            model = torch.nn.DataParallel(model)
    model.to(device)
    summary(model, input_size=(1, spectrogram_height, spectrogram_width))

    # --------------Training setup---------------------------
    if args.loss == "cross_entropy":
        loss_fn = nn.CrossEntropyLoss()
    elif args.loss == "focal_loss":
        loss_fn = FocalLoss()

    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)    
    elif args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)

    if args.scheduler == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.2)
    elif args.scheduler == "CosLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.valid_steps*4, eta_min=0)
    elif args.scheduler == "ReduceLR":
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=20, verbose=True, min_lr=1e-7)
        #customReduceLR stepでのlr_scheduleをサポート
        scheduler = CustomReduceLROnPlateau(optimizer, factor=0.5, patience=4, verbose=True, min_lr=1e-7)


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
    exec_time_start_time = time.time() #train　loopが始まるまでの記録
    early_stopping = EarlyStopping(patience=7, verbose=True)
    best_val_loss = float('inf')
    best_val_acc = 0.0
    global_step = 0
    for i in range(epochs):
        print(f"Epoch {i+1}")    
        loss_training_single_epoch_array = []
        y_true_train, y_pred_train = [], []
        #iteration単位でtqdmを使う
        for batch_idx, (input, target) in enumerate(tqdm(train_data_loader, desc="Training")):
            model.train() #model.train()　モデルを訓練モードに設定
            optimizer.zero_grad()
            input, target = input.to(device), target.to(device)

            if args.amp:
                with torch.autocast(device_type=device, dtype=torch.float16, enabled=args.amp):
                    prediction = model(input)
                    loss = loss_fn(prediction, target)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                prediction = model(input)
                loss = loss_fn(prediction, target)
                loss.backward()
                optimizer.step()

            loss_training_single_epoch_array.append(loss.item())
            y_true_train.extend(target.cpu().numpy())
            y_pred_train.extend(torch.argmax(prediction, dim=1).cpu().numpy())
            # イテレーション単位で精度を計算
            correct_predictions = (torch.argmax(prediction, dim=1) == target).sum().item()
            accuracy = correct_predictions / input.size(0)
            
            # 10ステップごとに平均を取ってログを記録
            if global_step % 10 == 0 and global_step % len(train_data_loader) != 0:
                avg_loss = np.mean(loss_training_single_epoch_array[-10:])
                avg_accuracy = np.mean([accuracy for _ in range(10)])  # accuracyは10ステップ分の平均を取る
                if args.wandb:
                    wandb.log({
                        "step": global_step,
                        "train_loss_iter": avg_loss,
                        "train_acc_iter": avg_accuracy * 100,
                        "lr": optimizer.param_groups[0]['lr']
                    })
                tqdm.write(f"Epoch {i+1} Iteration {global_step} : Avg Train Loss {avg_loss} : Avg Train Accuracy {avg_accuracy * 100}")
            
            global_step += 1 #global_stepの更新

            exec_time = time.time() - exec_time_start_time 

            # 規定のiteration数になったらValidation stepに移る(args.valid_steps)
            if global_step % args.valid_steps == 0 and global_step % len(train_data_loader) != 0:
                print("Validation step")
                loss_validation_array = []
                y_true_val, y_pred_val, y_pred_val_proba = [], [], []
                model.eval() #modelを評価モードに
                for input, target in tqdm(val_data_loader):
                    input, target = input.to(device), target.to(device)
                    with torch.no_grad():
                        prediction = model(input)
                        loss = loss_fn(prediction, target)
                        loss_validation_array.append(loss.item())
                        # Convert tensor predictions to numpy arrays
                        y_true_val.extend(target.cpu().numpy())
                        y_pred_val_proba.extend(prediction.cpu().detach().numpy()) #TODO: double check if original array is modified
                        y_pred_val.extend(torch.argmax(prediction, dim=1).cpu().numpy())

                loss_validation = np.array(loss_validation_array).mean()
                loss_validation_epochs.append(loss_validation)
            
                classification_report_val = classification_report(y_true_val, y_pred_val, target_names=CLASS_NAMES, output_dict=True)
                # --------------Save model---------------------------        
                if loss_validation < best_val_loss:
                    output_model_path_loss = os.path.join(output_dir, "best_loss.pth")
                    best_val_loss = loss_validation
                    torch.save(model.state_dict(), output_model_path_loss)
                    print("Trained feed forward net saved at: ", output_model_path_loss)
                if classification_report_val['accuracy'] > best_val_acc:
                    output_model_path_acc = os.path.join(output_dir, "best_acc.pth")
                    best_val_acc = classification_report_val['accuracy']
                    torch.save(model.state_dict(), output_model_path_acc)
                    print("Trained feed forward net saved at: ", output_model_path_acc)

                print("Validation Classification Report:")
                print(classification_report_val)

                # print(f"Training accuracy : {classification_report_train['accuracy']} ; Training loss : {loss_training_single_epoch}  ")
                print(f"Validation accuracy : {classification_report_val['accuracy']} ; Validation loss : {loss_validation} ")        
                print("---------------------------")

            # --------------Wandb log---------------------------  
                if args.wandb:
                    wandb.log({
                    "epoch": i+1,
                    "exec_time": exec_time,
                    "val_loss": loss_validation,
                    "val_acc":classification_report_val['accuracy'] * 100,
                    'precision_environ':classification_report_val['environ']['precision'],
                    'recall_environ':classification_report_val['environ']['recall'],
                    'f1-score_environ':classification_report_val['environ']['f1-score'],
                    'precision_talk':classification_report_val['talk']['precision'],
                    'recall_talk':classification_report_val['talk']['recall'],
                    'f1-score_talk':classification_report_val['talk']['f1-score'],
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
                    
                
                scheduler.step(loss_validation)#val_lossに基づいてscheduler_step更新

                # Early stopping check
                early_stopping(loss_validation) #所定のイテレーションでlossを計算しているから早期終了が発動する
                if early_stopping.early_stop:
                    print("early stopping")
                    break
        if early_stopping.early_stop:  # エポック全体を終了させる
            break
        
        # scheduler.step()#1epochの終わりにlrlate scheduler更新

    print("Finished training")
    print("---------------------------")
    print("---------------------------")
    print("---------------------------")
    print("test start")
    # --------------Test--------------------------- 所定のepoch数が終わったらtest開始する
    for output_model_path in [output_model_path_loss, output_model_path_acc]:
        model.load_state_dict(torch.load(output_model_path))
        print(f"Model loaded from {output_model_path}")
        model.eval()
        phase = output_model_path.split("/")[-1].split(".")[0]
        loss_test = []
        file_names = []
        y_true_test, y_pred_test, y_pred_test_proba = [], [], []
        with torch.no_grad():
            for input, target, file_path in tqdm(test_data_loader):
                input, target = input.to(device,dtype=torch.float32), target.to(device)
                prediction = model(input)
                loss = loss_fn(prediction, target)            
                loss_test.append(loss.item())
                y_true_test.extend(target.cpu().numpy())
                y_pred_test_proba.extend(prediction.cpu().detach().numpy()) #TODO: double check if original array is modified
                y_pred_test.extend(torch.argmax(prediction, dim=1).cpu().numpy())
                file_names.extend(file_path)
        loss_test = np.array(loss_test).mean()
        
        print("\nTest Report:")
        # classification_report_test = classification_report(y_true_test, y_pred_test, target_names=CLASS_NAMES)
        # print(classification_report_test)
        classification_report_test = classification_report(y_true_test, y_pred_test, target_names=CLASS_NAMES, output_dict=True)
        print(classification_report_test)
        print(f"\n{phase} Test accuracy : {classification_report_test['accuracy']} ; {phase} loss : {loss_test}  ")
        # ファイル名と予測結果をCSVに保存
        output_path = "~/e_haruki/sound-surveillance/test"
        output_file = f"test_results_clean.csv"
        result_df = pd.DataFrame({'file_name': file_names, 'true_label': y_true_test, 'predicted_label': y_pred_test})
        result_df.to_csv(os.path.join(output_path, output_file), index=False)
        print("---------------------------")
        if args.wandb:
            wandb.log({
                f"{phase}_loss": loss_test,
                f"{phase}_acc": classification_report_test['accuracy'] * 100,
                f"{phase}_precision_environ":classification_report_test['environ']['precision'],
                f"{phase}_recall_environ":classification_report_test['environ']['recall'],
                f"{phase}_f1-score_environ":classification_report_test['environ']['f1-score'],
                f"{phase}_precision_talk":classification_report_test['talk']['precision'],
                f"{phase}_recall_talk":classification_report_test['talk']['recall'],
                f"{phase}_f1-score_talk":classification_report_test['talk']['f1-score'],
                f"{phase}_precision_walk":classification_report_test['walk']['precision'],
                f"{phase}_recall_walk":classification_report_test['walk']['recall'],
                f"{phase}_f1-score_walk":classification_report_test['walk']['f1-score'],
            })
            wandb.log({f"{phase}_conf_mat" : wandb.plot.confusion_matrix(probs=None,y_true=y_true_test,preds=y_pred_test,
                    class_names=CLASS_NAMES)})
            wandb.log({f"{phase}_roc" : wandb.plot.roc_curve(y_true_test, y_pred_test_proba,
                    labels=CLASS_NAMES)})
            wandb.log({f"{phase}_pr" : wandb.plot.pr_curve(y_true_test, y_pred_test_proba,
                    labels=CLASS_NAMES)})
        # confusion matrixの画像を保存
        fig, ax = plt.subplots(figsize=(10, 10))
        cm = confusion_matrix(y_true_test, y_pred_test, labels=range(len(CLASS_NAMES)))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
        disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical', ax=ax)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{phase}_confusion_matrix.png"))  # 先に保存
        plt.close(fig)  # その後に閉じる
        if args.wandb:
            wandb.log({f"{phase}_confusion_matrix": wandb.Image(fig)})


def main(args):
    seed_everything(args.seed)
    if args.wandb:
        wandb.init(entity="your_name", name=f"{args.model}_optimizer_{args.optimizer}_loss_{args.loss}_scheduler_{args.scheduler}_lr{args.learning_rate}_n_mels{args.n_mels}_window_size{args.window_size}_batch_size{args.batch_size}", project=args.project_name, config={"max_plot_points": 20000})
    try:
        train(args)
    finally:
        if args.wandb:
            wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate of model')
    parser.add_argument('--n_mels',type=int, default=30, help='Number of mels')
    parser.add_argument('--window_size', type=int, default=30, help='Window size of mel spectrogram')
    parser.add_argument('--epochs', type=int, default=10000, help='Number of epochs')
    parser.add_argument('--valid_steps', type=int, default=200, help='Number of validation steps')
    parser.add_argument('--project_name', type=str, default="debug", help='Project name')
    parser.add_argument('--seed', type=int, default=3407, help='Random seed')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--sample_rate', type=int, default=16000, help='Sample rate')
    parser.add_argument('--duration_ms', type=int, default=1000, help='Duration of audio in milliseconds')
    parser.add_argument('--model', type=str, default="depthwise_separable_CNN2", help='Model name') #depthwise_separable_CNN2
    parser.add_argument('--amp', action='store_true', default=True, help='Use mixed precision')
    parser.add_argument('--data_parallel', action='store_true', default=False, help='Use Data_parallel')
    parser.add_argument('--loss', type=str, default="cross_entropy", help='Loss function')
    parser.add_argument('--optimizer', type=str, default="adamw", help='Optimizer')
    parser.add_argument('--wandb', action='store_true', help='Use wandb for logging')
    parser.add_argument('--scheduler', type=str, default='StepLR', help='Learning rate scheduler')

    args = parser.parse_args()
    main(args)
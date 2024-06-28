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
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary
from torch.utils.data import WeightedRandomSampler
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from torchvision.models import resnet18, ResNet18_Weights, resnet34, ResNet34_Weights, resnet50, ResNet50_Weights
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
from models import CNNNetwork1, CNNNetwork2, CNNNetwork3, CNNNetwork4, Student_s, Student_m, StudentResNet
import torch.nn.functional as F
import multiprocessing

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
    output_dir = f"./experiment/{args.project_name}/{args.student}_optimizer_{args.optimizer}_loss_{args.loss}_lr{args.learning_rate}_n_mels{args.n_mels}_window_size{args.window_size}_alpha_{args.alpha}_temp_{args.temperature}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    #------------Training setup ----------------------------
    device = device_check(display_device = True)
    print("Device: ", device)
    print("Number of CPU: ", multiprocessing.cpu_count())

    train_dataset = AudioDataset(csv_path='train.csv',win_length_samples=win_length_samples,hop_length_samples=hop_length_samples,n_mels_value=args.n_mels, target_sample_rate=args.sample_rate, num_samples=NUM_SAMPLES, device=device) 
    val_dataset = AudioDataset(csv_path='val.csv',win_length_samples=win_length_samples,hop_length_samples=hop_length_samples,n_mels_value=args.n_mels, target_sample_rate=args.sample_rate, num_samples=NUM_SAMPLES, device=device) 
    test_dataset = AudioDataset(csv_path='test.csv',win_length_samples=win_length_samples,hop_length_samples=hop_length_samples,n_mels_value=args.n_mels, target_sample_rate=args.sample_rate, num_samples=NUM_SAMPLES, device=device)    
    
    print(f"Number of training dataset: {len(train_dataset)}")
    print(f"Number of validation dataset: {len(val_dataset)}")
    print(f"Number of test dataset: {len(test_dataset)}")
    
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True, num_workers=8, pin_memory=True)
    val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False, num_workers=8, pin_memory=False)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False, num_workers=8, pin_memory=False)
    # サンプルデータの取得
    audio_sample_path = "00009672.wav"
    signal_sample, sr = librosa.load(audio_sample_path, sr=args.sample_rate)
    mel_spectrogram = librosa.feature.melspectrogram(y=signal_sample, sr=sr, n_fft=win_length_samples, win_length=win_length_samples, hop_length=hop_length_samples, n_mels=args.n_mels)
    mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)  
    mel_spectrogram = np.expand_dims(mel_spectrogram, axis=0)
    print("Shape of sample spectrogram: ", mel_spectrogram.shape)
    
    spectrogram_height = mel_spectrogram.shape[1]
    spectrogram_width = mel_spectrogram.shape[2]
    output_class_number = len(CLASS_NAMES)

    # --------------Model setup(学習済みモデル　重みを固定---------------------------
    if args.model == "resnet18":
        print("model resnet18")
        teacher = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        teacher.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        teacher.fc = nn.Linear(512,output_class_number)

    print("Type of teacher:", type(teacher))
    teacher.to(device)
    print("teacher_summary")
    summary(teacher, input_size=(1, spectrogram_height, spectrogram_width))
    teacher.load_state_dict(torch.load('best_acc.pth'))#読み込み

    # ---------------Student model--------------------------------------------------
    # ニューラルネットワークモデルの定義
    """
    #ニューラルネットワークを作成するには、nn.Moduleクラスを継承し、ニューラルネットワークのクラスを作成する
    """
    # studentモデルの選択
    if args.student == "Student_s":
        print("Student_s")
        student = Student_s(output_class_number)
    elif args.student == "Student_m":
        print("Student_m")
        student = Student_m(output_class_number)
    elif args.student == "StudentResNet":
        print("StudentResNet")
        student = StudentResNet(output_class_number)

    print("Type of student:", type(student))
    student=student.to(device)
    print("student_summary")
    summary(student, input_size=(1, spectrogram_height, spectrogram_width))

    # --------------Training setup---------------------------
    if args.loss == "cross_entropy":
        loss_fn = nn.CrossEntropyLoss()
    elif args.loss == "focal_loss":
        loss_fn = FocalLoss()

    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(student.parameters(), lr=args.learning_rate)    
    elif args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(student.parameters(), lr=args.learning_rate, weight_decay=0.01)

        #一貫性ロス(KLダイバージェンス)
    #https://qiita.com/tand826/items/13d6480c66dd865ad9b5
    kldiv = nn.KLDivLoss(reduction="batchmean", log_target=True)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.valid_steps*2, eta_min=0)

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
            summary(student, input_size=(1, spectrogram_height, spectrogram_width))
    
    print(args_dict)
    # --------------Training loop---------------------------
    epochs = args.epochs
    alpha = args.alpha
    temperature = args.temperature
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
        #iteration単位でtqdmを使う
        for batch_idx, (input, target) in enumerate(tqdm(train_data_loader, desc="Training")):
            student.train()
            optimizer.zero_grad()
            input, target = input.to(device), target.to(device)
            if args.amp:
                with torch.autocast(device_type=device, dtype=torch.float16, enabled=args.amp):
                    #分類ロス(クロスエントロピー)
                    prediction_s = student(input)#studentモデル
                    prediction_t = teacher(input)#Teacherモデル
                    crossloss = loss_fn(prediction_s, target)
                    sum_train_crossloss += crossloss.item()
                    #一貫性ロス(KLダイバージェンス)
                    kldivloss = kldiv((F.log_softmax(prediction_s/temperature, dim = 1)), (F.log_softmax(prediction_t/temperature, dim = 1)))
                    sum_train_kldivloss += kldivloss.item()
                    #https://qiita.com/M_Hiro/items/0ba24788c78540046bcd
                    # loss = (1-alpha)*crossloss + alpha*temperature*temperature*kldivloss
                    loss = (1-alpha)*crossloss + alpha*temperature*temperature*kldivloss
                    train_loss += loss.item()
                #Automatic Mixed Precision (AMP)を使用する際には、全体の損失 (loss) をバックプロパゲーションに使用するのが最適
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                prediction_s = student(input)#studentモデル
                prediction_t = teacher(input)#Teacherモデル
                crossloss = loss_fn(prediction_s, target)
                sum_train_crossloss += crossloss.item()
                #一貫性ロス(KLダイバージェンス)
                kldivloss = kldiv((F.log_softmax(prediction_s/temperature, dim = 1)), (F.log_softmax(prediction_t/temperature, dim = 1)))
                sum_train_kldivloss += kldivloss.item()
                #https://qiita.com/M_Hiro/items/0ba24788c78540046bcd
                # loss = (1-alpha)*crossloss + alpha*temperature*temperature*kldivloss
                loss = (1-alpha)*crossloss + alpha*temperature*temperature*kldivloss
                loss.backward()
                optimizer.step()

            loss_training_single_epoch_array.append(loss.item())
            y_true_train.extend(target.cpu().numpy())
            y_pred_train.extend(torch.argmax(prediction_s, dim=1).cpu().numpy())
            # イテレーション単位で精度を計算
            correct_predictions = (torch.argmax(prediction_s, dim=1) == target).sum().item()
            accuracy = correct_predictions / input.size(0)
            
            # 10ステップごとに平均を取ってログを記録
            if batch_idx % 10 == 0 and batch_idx != 0:
                avg_loss = np.mean(loss_training_single_epoch_array[-10:])
                avg_accuracy = np.mean([accuracy for _ in range(10)])  # accuracyは10ステップ分の平均を取る
                if args.wandb:
                    wandb.log({
                        "iteration": i * len(train_data_loader) + batch_idx,
                        "train_loss_iter": avg_loss,
                        "train_acc_iter": avg_accuracy * 100,
                        "lr": optimizer.param_groups[0]['lr']
                    })
                tqdm.write(f"Epoch {i+1} Iteration {batch_idx} : Avg Train Loss {avg_loss} : Avg Train Accuracy {avg_accuracy * 100}")

            # if batch_idx % 10 == 0:
            #     print(f"Epoch {i+1} Iteration {batch_idx} : Loss {loss.item()} ; Accuracy {accuracy * 100}")
            #     if batch_idx == 200:
            #         break
            exec_time = time.time() - exec_time_start_time 
        # loss_training_single_epoch = np.array(loss_training_single_epoch_array).mean()
        # loss_training_epochs.append(loss_training_single_epoch)
            # classification_report_train = classification_report(y_true_train, y_pred_train, target_names=CLASS_NAMES, output_dict=True)
            # if batch_idx % 10 == 0 and batch_idx != 0:
            #     tqdm.write(f"Epoch {i+1} Iteration {batch_idx} : Loss {loss.item()} : Accuracy {accuracy * 100}")
                # if batch_idx == 200:
                #     break    
            # 規定のiteration数になったらValidation stepに移る(args.valid_steps)
            if batch_idx % args.valid_steps == 0 and batch_idx != 0:
                print("Validation step")
                loss_validation_array = []
                y_true_val, y_pred_val, y_pred_val_proba = [], [], []
                student.eval()
                for input, target in tqdm(val_data_loader):
                    input, target = input.to(device), target.to(device)
                    with torch.no_grad(): #重み更新しない
                        #分類ロス(クロスエントロピー)
                        prediction_s = student(input)#studentモデル
                        prediction_t = teacher(input)#Teacherモデル
                        crossloss = loss_fn(prediction_s, target)
                        sum_val_crossloss += crossloss.item()
                        #一貫性ロス(KLダイバージェンス)
                        kldivloss = kldiv((F.log_softmax(prediction_s/temperature, dim = 1)), (F.log_softmax(prediction_t/temperature, dim = 1)))
                        sum_val_kldivloss += kldivloss.item()
                        #https://qiita.com/M_Hiro/items/0ba24788c78540046bcd
                        # loss = (1-alpha)*crossloss + alpha*temperature*temperature*kldivloss
                        loss = (1-alpha)*crossloss + alpha*temperature*temperature*kldivloss

                        loss_validation_array.append(loss.item())
                        # Convert tensor predictions to numpy arrays
                        y_true_val.extend(target.cpu().numpy())
                        y_pred_val_proba.extend(prediction_s.cpu().detach().numpy()) #TODO: double check if original array is modified
                        y_pred_val.extend(torch.argmax(prediction_s, dim=1).cpu().numpy())

                loss_validation = np.array(loss_validation_array).mean() #loss_validation_arrayの中の平均値
                loss_validation_epochs.append(loss_validation)
            
                classification_report_val = classification_report(y_true_val, y_pred_val, target_names=CLASS_NAMES, output_dict=True)
                # --------------Save model---------------------------        
                if loss_validation < best_val_loss:
                    output_model_path_loss = os.path.join(output_dir, "best_loss.pth")
                    best_val_loss = loss_validation
                    torch.save(student.state_dict(), output_model_path_loss)
                    print("Trained feed forward net saved at: ", output_model_path_loss)
                if classification_report_val['accuracy'] > best_val_acc:
                    output_model_path_acc = os.path.join(output_dir, "best_acc.pth")
                    best_val_acc = classification_report_val['accuracy']
                    torch.save(student.state_dict(), output_model_path_acc)
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
                early_stopping(loss_validation)
                if early_stopping.early_stop:
                    print("early stopping")
                    break
                # if batch_idx == 200:
                #     break
            scheduler.step()




    print("Finished training")
    print("---------------------------")
    print("---------------------------")
    print("---------------------------")
    print("Testing Start")
    # --------------Test---------------------------
    for output_model_path in [output_model_path_loss, output_model_path_acc]:
        student.load_state_dict(torch.load(output_model_path))
        print(f"Model loaded from {output_model_path}")
        student.eval()
        phase = output_model_path.split("/")[-1].split(".")[0]
        loss_test = []
        y_true_test, y_pred_test, y_pred_test_proba = [], [], []
        with torch.no_grad():
            for input, target in tqdm(test_data_loader):
                input, target = input.to(device,dtype=torch.float32), target.to(device)
                #分類ロス(クロスエントロピー)
                prediction_s = student(input)#studentモデル
                prediction_t = teacher(input)#Teacherモデル
                crossloss = loss_fn(prediction_s, target)
                sum_test_crossloss += crossloss.item()
                #一貫性ロス(KLダイバージェンス)
                kldivloss = kldiv((F.log_softmax(prediction_s/temperature, dim = 1)), (F.log_softmax(prediction_t/temperature, dim = 1)))
                sum_test_kldivloss += kldivloss.item()

                loss = (1-alpha)*crossloss + alpha*temperature*temperature*kldivloss
                loss_test.append(loss.item())
                y_true_test.extend(target.cpu().numpy())
                y_pred_test_proba.extend(prediction_s.cpu().detach().numpy()) #TODO: double check if original array is modified
                y_pred_test.extend(torch.argmax(prediction_s, dim=1).cpu().numpy())
        loss_test = np.array(loss_test).mean()
        
        print("\nTest Report:")
        # classification_report_test = classification_report(y_true_test, y_pred_test, target_names=CLASS_NAMES)
        # print(classification_report_test)
        classification_report_test = classification_report(y_true_test, y_pred_test, target_names=CLASS_NAMES, output_dict=True)
        print(classification_report_test)
        print(f"\n{phase} Test accuracy : {classification_report_test['accuracy']} ; {phase} loss : {loss_test}  ")
        print("---------------------------")
        if args.wandb:
            wandb.log({
                f"{phase}_loss": loss_test,
                f"{phase}_acc": classification_report_test['accuracy'] * 100,
                f"{phase}_precision_cut":classification_report_test['cut']['precision'],
                f"{phase}_recall_cut":classification_report_test['cut']['recall'],
                f"{phase}_f1-score_cut":classification_report_test['cut']['f1-score'],
                f"{phase}_precision_car":classification_report_test['car']['precision'],
                f"{phase}_recall_car":classification_report_test['car']['recall'],
                f"{phase}_f1-score_car":classification_report_test['car']['f1-score'],
                f"{phase}_precision_environ":classification_report_test['environ']['precision'],
                f"{phase}_recall_environ":classification_report_test['environ']['recall'],
                f"{phase}_f1-score_environ":classification_report_test['environ']['f1-score'],
                f"{phase}_precision_fruit":classification_report_test['fruit']['precision'],
                f"{phase}_recall_fruit":classification_report_test['fruit']['recall'],
                f"{phase}_f1-score_fruit":classification_report_test['fruit']['f1-score'],
                f"{phase}_precision_leaf":classification_report_test['leaf']['precision'],
                f"{phase}_recall_leaf":classification_report_test['leaf']['recall'],
                f"{phase}_f1-score_leaf":classification_report_test['leaf']['f1-score'],
                f"{phase}_precision_talk":classification_report_test['talk']['precision'],
                f"{phase}_recall_talk":classification_report_test['talk']['recall'],
                f"{phase}_f1-score_talk":classification_report_test['talk']['f1-score'],
                f"{phase}_precision_truck":classification_report_test['truck']['precision'],
                f"{phase}_recall_truck":classification_report_test['truck']['recall'],
                f"{phase}_f1-score_truck":classification_report_test['truck']['f1-score'],
                f"{phase}_precision_unknown":classification_report_test['unknown']['precision'],
                f"{phase}_recall_unknown":classification_report_test['unknown']['recall'],
                f"{phase}_f1-score_unknown":classification_report_test['unknown']['f1-score'],
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
      wandb.init(entity="wandb_name", name=f"{args.student}_optimizer_{args.optimizer}_loss_{args.loss}_lr{args.learning_rate}_n_mels{args.n_mels}_window_size{args.window_size}_alpha_{args.alpha}_temp_{args.temperature}", project=args.project_name, config={"max_plot_points": 20000})
    try:
        train(args)
    finally:
        if args.wandb:
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
        '--valid_steps',
        type=int,
        default=200,
        help='Number of validation steps',)
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
    parser.add_argument(
      '--student',
      type=str,
      default="StudentResNet",
      help="Model name",)
    parser.add_argument(
      '--alpha',
      default=0.9,
      help="alpha",)
    parser.add_argument(
      '--temperature',
      default=10,
      help="temperature",)
    parser.add_argument('--amp', action='store_true', default=True, help='Use mixed precision')
    parser.add_argument('--loss', type=str, default="cross_entropy", help='Loss function')
    parser.add_argument('--optimizer', type=str, default="adamw", help='Optimizer')
    parser.add_argument('--wandb', action='store_true', help='Use wandb for logging')
    args = parser.parse_args()
    main(args)
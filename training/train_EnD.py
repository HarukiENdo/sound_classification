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
from torchinfo import summary as summary_
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from torchvision.models import resnet18, ResNet18_Weights, resnet34, ResNet34_Weights, resnet50, ResNet50_Weights
from torchvision.models import resnext50_32x4d, ResNeXt50_32X4D_Weights, resnext101_32x8d, ResNeXt101_32X8D_Weights #ResNextのimport
import time
import wandb
import csv
from tqdm import tqdm
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import numpy as np
import os
import sys
# プロジェクトのルートディレクトリを追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from datasetclass.dataset_torchaudio import AudioDataset, AudioDataset_path ##
from util.utils import * ##
import random
import torch.cuda.amp as amp
from util.loss import FocalLoss ##
from contextlib import redirect_stdout
import yaml
from model.models import CNNNetwork1, CNNNetwork2, CNNNetwork3, CNNNetwork4, Student_s ##
import torch.nn.functional as F
import multiprocessing
from transformers import ASTConfig, ASTModel
from transformers import ASTFeatureExtractor
from transformers import AutoModelForAudioClassification
from transformers import Wav2Vec2Processor, ASTForAudioClassification
from transformers.models.audio_spectrogram_transformer.modeling_audio_spectrogram_transformer import ASTEmbeddings

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
    CLASS_NAMES = ['environ','talk','walk']
    #------------Parameters setup -------------------------
    win_length_ms = args.window_size
    hop_length_ms = int(win_length_ms/3)
    win_length_samples = int(NUM_SAMPLES*win_length_ms/1000)
    hop_length_samples = int(NUM_SAMPLES*hop_length_ms/1000)
    #------------Name setup --------------------------------
    output_dir = f"./experiment_kd/EnD/{args.project_name}/{args.student}_{args.pretrain}_optimizer_{args.optimizer}_batch_{args.batch_size}_loss_{args.loss}_lr{args.learning_rate}_n_mels{args.n_mels}_window_size{args.window_size}_alpha{args.alpha}_temp{args.temperature}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    #------------Training setup ----------------------------
    device = device_check(display_device = True)
    print("Device: ", device)
    n_gpu = torch.cuda.device_count()
    # DataParallelを使ってモデルを複数GPUで並列化
    print(f"Available GPUs: {n_gpu}")
    print("Number of CPU: ", multiprocessing.cpu_count())

    train_dataset = AudioDataset(csv_path='train.csv',win_length_samples=win_length_samples,hop_length_samples=hop_length_samples,n_mels_value=args.n_mels, target_sample_rate=args.sample_rate, num_samples=NUM_SAMPLES) 
    val_dataset = AudioDataset(csv_path='val.csv',win_length_samples=win_length_samples,hop_length_samples=hop_length_samples,n_mels_value=args.n_mels, target_sample_rate=args.sample_rate, num_samples=NUM_SAMPLES) 
    test_dataset = AudioDataset(csv_path='test.csv',win_length_samples=win_length_samples,hop_length_samples=hop_length_samples,n_mels_value=args.n_mels, target_sample_rate=args.sample_rate, num_samples=NUM_SAMPLES)    
    
    print(f"Number of training dataset: {len(train_dataset)}")
    print(f"Number of validation dataset: {len(val_dataset)}")
    print(f"Number of test dataset: {len(test_dataset)}")
    
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True, num_workers=8, pin_memory=True)
    val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False, num_workers=8, pin_memory=False)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False, num_workers=8, pin_memory=False)
    # サンプルデータの取得
    audio_sample_path = "sample.wav"
    signal_sample, sr = torchaudio.load(audio_sample_path)
    mel_spectrogram_transform = torchaudio.transforms.MelSpectrogram(sample_rate=args.sample_rate,n_fft=win_length_samples,win_length=win_length_samples,hop_length=hop_length_samples,n_mels=args.n_mels)
    mel_spectrogram = mel_spectrogram_transform(signal_sample)  # メルスペクトログラムを生成
    mel_spectrogram = torchaudio.transforms.AmplitudeToDB()(mel_spectrogram)  # dBスケールに変換  
    print("Shape of sample spectrogram: ", mel_spectrogram.shape)
    
    spectrogram_height = mel_spectrogram.shape[1]
    spectrogram_width = mel_spectrogram.shape[2]
    output_class_number = len(CLASS_NAMES)

    # --------------教師モデルModel setup(学習済みモデル　重みを固定)---------------------------

    #---------------Resnet系教師モデル------------------------
    if args.teacher_R == "resnet18":
        print("model resnet18")
        teacher_R = resnet18()
        teacher_R.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        teacher_R.fc = nn.Linear(512,output_class_number)
    elif args.teacher_R == "resnext50":
        teacher_R = resnext50_32x4d()
        teacher_R.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        teacher_R.fc = nn.Linear(2048,output_class_number)

    print("Type of teacher:", type(teacher_R))

    teacher_R.to(device)
    print("resnet系teacher_summary")
    summary(teacher_R, input_size=(1, spectrogram_height, spectrogram_width))
    teacher_R.load_state_dict(torch.load('best_loss.pth'))#読み込み

    #--------------Transformer系モデル------------------------
    if args.teacher_T == "AST":
        print("Audio Spectrogram Transformer")
        print("model AST")
        # まず学習済みモデルをロード
        config = ASTConfig(num_labels=output_class_number, num_hidden_layers=args.hidden_layers, hidden_size=args.hidden_size, num_attention_heads=args.num_attention_heads, max_length=spectrogram_width, num_mel_bins=args.n_mels, patch_size=args.patch_size, frequency_stride=args.frequency_stride, time_stride=args.time_stride)
        teacher_T = ASTForAudioClassification(config)

    print("Type of teacher:", type(teacher_T))

    teacher_T.to(device)
    print("Transformer系teacher_summary")
    summary_(teacher_T, input_size=(1, spectrogram_width, spectrogram_height)) #cnnと順番が違うので注意
    # 重みをロードし、moduleプレフィックスがある場合は削除する
    state_dict_T = torch.load('best_acc.pth')
    new_state_dict_T = {k.replace('module.', ''): v for k, v in state_dict_T.items()}
    teacher_T.load_state_dict(new_state_dict_T)
    # ---------------Student model--------------------------------------------------
    # ニューラルネットワークモデルの定義
    """
    #ニューラルネットワークを作成するには、nn.Moduleクラスを継承し、ニューラルネットワークのクラスを作成する
    """
    # studentモデルの選択
    if args.student == "cnn_network4":
        print("model cnn_network4")
        student = CNNNetwork4(spectrogram_height, spectrogram_width, output_class_number)
    elif args.student == "Student_s":
        print("model student_s")
        student = Student_s(spectrogram_height, spectrogram_width, output_class_number)

    print("Type of student:", type(student))

    student=student.to(device)
    print("student_summary")
    summary(student, input_size=(1, spectrogram_height, spectrogram_width))
    if args.pretrain == "pretrain":
        student.load_state_dict(torch.load('best_loss.pth'))#読み込み

    # DataParallelを使ってモデルを複数GPUで並列化する場合
    if args.data_parallel:
        if n_gpu > 1:
            teacher_R = torch.nn.DataParallel(teacher_R)
            teacher_T = torch.nn.DataParallel(teacher_T)
            student = torch.nn.DataParallel(student)

    # --------------Training setup---------------------------
    #損失関数
    if args.loss == "cross_entropy":
        loss_fn = nn.CrossEntropyLoss()
    elif args.loss == "focal_loss":
        loss_fn = FocalLoss()
    
    #活性化関数
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(student.parameters(), lr=args.learning_rate)    
    elif args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(student.parameters(), lr=args.learning_rate, weight_decay=0.01)

    #学習率スケジューラー
    if args.scheduler == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.2)
    elif args.scheduler == "CosLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.valid_steps*4, eta_min=0)
    elif args.scheduler == "ReduceLR":
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=20, verbose=True, min_lr=1e-7)
        #customReduceLR stepでのlr_scheduleをサポート
        scheduler = CustomReduceLROnPlateau(optimizer, factor=0.5, patience=10, verbose=True, min_lr=1e-7)

        #一貫性ロス(KLダイバージェンス)
    #https://qiita.com/tand826/items/13d6480c66dd865ad9b5
    kldiv = nn.KLDivLoss(reduction="batchmean", log_target=True)

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
    beta = args.beta
    gamma = args.gamma
    temperature = args.temperature
    loss_training_epochs = []
    loss_validation_epochs = []
    exec_time_start_time = time.time()
    early_stopping = EarlyStopping(patience=40, verbose=True)
    best_val_loss = float('inf')
    best_val_acc = 0.0
    global_step = 0
    for i in range(epochs):
        teacher_R.eval()
        teacher_T.eval()
        print(f"Epoch {i+1}")    
        loss_training_single_epoch_array = []
        y_true_train, y_pred_train = [], []
        crossloss_training_single_epoch_array = []
        kldivloss_training_single_epoch_array = []
        #iteration単位でtqdmを使う
        for batch_idx, (input, target) in enumerate(tqdm(train_data_loader, desc="Training")):
            student.train()
            optimizer.zero_grad()
            input, target = input.to(device), target.to(device)
            #inputをASTの入力に合わせるための処理
            input_T = input.squeeze(1)
            input_T = torch.transpose(input_T, 1, 2)
            if args.amp:
                with torch.autocast(device_type=device, dtype=torch.float16, enabled=args.amp):
                    prediction_s = student(input)#studentモデル
                    prediction_t_R = teacher_R(input)#Resnet系Teacherモデル
                    prediction_t_T = teacher_T(input_T)#Transformer系Teacherモデル
                    
                    #分類ロス(クロスエントロピー)
                    crossloss = loss_fn(prediction_s, target)
                    if args.EnDmethod == "method1":
                        #一貫性ロス(KLダイバージェンス) Resnet系教師とTransformer系教師のlogitsの平均を教師モデルのlogitsとして扱う(手法1)
                        prediction_t = (prediction_t_R + prediction_t_T.logits) / 2
                        #生徒モデルの出力確率分布も同様にtemperatureで割って平滑化するのは、同じスケールでLossを計算するため
                        kldivloss = kldiv((F.log_softmax(prediction_s/temperature, dim = 1)), (F.log_softmax(prediction_t/temperature, dim = 1)))
                        #https://qiita.com/M_Hiro/items/0ba24788c78540046bcd
                        loss = (1-alpha)*crossloss + alpha*kldivloss
                    elif args.EnDmethod == "method2":
                        kldivloss_R = kldiv((F.log_softmax(prediction_s/temperature, dim = 1)), (F.log_softmax(prediction_t_R/temperature, dim = 1)))
                        kldivloss_T = kldiv((F.log_softmax(prediction_s/temperature, dim = 1)), (F.log_softmax(prediction_t_T.logits/temperature, dim = 1)))
                        #https://qiita.com/M_Hiro/items/0ba24788c78540046bcd
                        loss = alpha*crossloss + beta*kldivloss_R + gamma*kldivloss_T

                #Automatic Mixed Precision (AMP)を使用する際には、全体の損失 (loss) をバックプロパゲーションに使用するのが最適
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                prediction_s = student(input)#studentモデル
                prediction_t_R = teacher_R(input)#Resnet系Teacherモデル
                prediction_t_T = teacher_T(input_T)#Transformer系Teacherモデル
                
                #分類ロス(クロスエントロピー)
                crossloss = loss_fn(prediction_s, target)
                if args.EnDmethod == "method1":
                    #一貫性ロス(KLダイバージェンス) Resnet系教師とTransformer系教師のlogitsの平均を教師モデルのlogitsとして扱う(手法1)
                    prediction_t = (prediction_t_R + prediction_t_T.logits) / 2
                    #生徒モデルの出力確率分布も同様にtemperatureで割って平滑化するのは、同じスケールでLossを計算するため
                    kldivloss = kldiv((F.log_softmax(prediction_s/temperature, dim = 1)), (F.log_softmax(prediction_t/temperature, dim = 1)))
                    #https://qiita.com/M_Hiro/items/0ba24788c78540046bcd
                    loss = (1-alpha)*crossloss + alpha*kldivloss
                elif args.EnDmethod == "method2":
                    kldivloss_R = kldiv((F.log_softmax(prediction_s/temperature, dim = 1)), (F.log_softmax(prediction_t_R/temperature, dim = 1)))
                    kldivloss_T = kldiv((F.log_softmax(prediction_s/temperature, dim = 1)), (F.log_softmax(prediction_t_T.logits/temperature, dim = 1)))
                    #https://qiita.com/M_Hiro/items/0ba24788c78540046bcd
                    loss = alpha*crossloss + beta*kldivloss_R + gamma*kldivloss_T
                loss.backward()
                optimizer.step()

            loss_training_single_epoch_array.append(loss.item())
            crossloss_training_single_epoch_array.append(crossloss.item())
            kldivloss_training_single_epoch_array.append(kldivloss.item())
            y_true_train.extend(target.cpu().numpy())
            y_pred_train.extend(torch.argmax(prediction_s, dim=1).cpu().numpy())
            # イテレーション単位で精度を計算
            correct_predictions = (torch.argmax(prediction_s, dim=1) == target).sum().item()
            accuracy = correct_predictions / input.size(0)
            
            # 10ステップごとに平均を取ってログを記録
            if global_step % 10 == 0 and global_step % len(train_data_loader) != 0:
                avg_loss = np.mean(loss_training_single_epoch_array[-10:])
                avg_crossloss = np.mean(crossloss_training_single_epoch_array[-10:])
                avg_kldivloss = np.mean(kldivloss_training_single_epoch_array[-10:])
                avg_accuracy = np.mean([accuracy for _ in range(10)])  # accuracyは10ステップ分の平均を取る
                if args.wandb:
                    wandb.log({
                        "step": global_step,
                        "train_loss_iter": avg_loss,
                        "train_acc_iter": avg_accuracy * 100,
                        "train_crossloss_iter": avg_crossloss,
                        "train_kldivloss_iter": avg_kldivloss,
                        "lr": optimizer.param_groups[0]['lr']
                    })
                tqdm.write(f"Epoch {i+1} Iteration {global_step} : Avg Train Loss {avg_loss} : Avg Train Crossloss {avg_crossloss} : Avg Train Kldivloss {avg_kldivloss} : Avg Train Accuracy {avg_accuracy * 100}")

            global_step += 1 #global_stepの更新
            exec_time = time.time() - exec_time_start_time 

            # 規定のiteration数になったらValidation stepに移る(args.valid_steps)
            if global_step % args.valid_steps == 0 and global_step % len(train_data_loader) != 0:
                print("Validation step")
                loss_validation_array = []
                loss_crossloss_array = []
                loss_kldivloss_array = []
                y_true_val, y_pred_val, y_pred_val_proba = [], [], []
                student.eval()
                for input, target in tqdm(val_data_loader):
                    input, target = input.to(device), target.to(device)
                    #inputをASTの入力に合わせるための処理
                    input_T = input.squeeze(1)
                    input_T = torch.transpose(input_T, 1, 2)                   
                    with torch.no_grad(): #重み更新しない
                        prediction_s = student(input)#studentモデル
                        prediction_t_R = teacher_R(input)#Resnet系Teacherモデル
                        prediction_t_T = teacher_T(input_T)#Transformer系Teacherモデル
                        #分類ロス(クロスエントロピー)
                        crossloss = loss_fn(prediction_s, target)
                        if args.EnDmethod == "method1":
                            #一貫性ロス(KLダイバージェンス) Resnet系教師とTransformer系教師のlogitsの平均を教師モデルのlogitsとして扱う(手法1)
                            prediction_t = (prediction_t_R + prediction_t_T.logits) / 2
                            #生徒モデルの出力確率分布も同様にtemperatureで割って平滑化するのは、同じスケールでLossを計算するため
                            kldivloss = kldiv((F.log_softmax(prediction_s/temperature, dim = 1)), (F.log_softmax(prediction_t/temperature, dim = 1)))
                            #https://qiita.com/M_Hiro/items/0ba24788c78540046bcd
                            loss = (1-alpha)*crossloss + alpha*kldivloss
                        elif args.EnDmethod == "method2":
                            kldivloss_R = kldiv((F.log_softmax(prediction_s/temperature, dim = 1)), (F.log_softmax(prediction_t_R/temperature, dim = 1)))
                            kldivloss_T = kldiv((F.log_softmax(prediction_s/temperature, dim = 1)), (F.log_softmax(prediction_t_T.logits/temperature, dim = 1)))
                            #https://qiita.com/M_Hiro/items/0ba24788c78540046bcd
                            loss = alpha*crossloss + beta*kldivloss_R + gamma*kldivloss_T

                        loss_crossloss_array.append(crossloss.item())
                        loss_kldivloss_array.append(kldivloss.item())
                        loss_validation_array.append(loss.item())
                        # Convert tensor predictions to numpy arrays
                        y_true_val.extend(target.cpu().numpy())
                        y_pred_val_proba.extend(prediction_s.cpu().detach().numpy()) #TODO: double check if original array is modified
                        y_pred_val.extend(torch.argmax(prediction_s, dim=1).cpu().numpy())

                crossloss_validation = np.array(loss_crossloss_array).mean()
                kldivloss_validation = np.array(loss_kldivloss_array).mean()
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

                print(f"Validation accuracy : {classification_report_val['accuracy']} ; Validation loss : {loss_validation} ; Validation crossloss : {crossloss_validation} ; Validation kldivloss : {kldivloss_validation} ")        
                print("---------------------------")

            # --------------Wandb log---------------------------  
                if args.wandb:
                    wandb.log({
                    "epoch": i+1,
                    "exec_time": exec_time,
                    "val_loss": loss_validation,
                    "val_crossloss": crossloss_validation,
                    "val_kldivloss": kldivloss_validation,
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

        if early_stopping.early_stop:
            break

    print("Finished training")
    print("---------------------------")
    # if args.quantization == True:
    #     torch.quantization.convert(student.eval(), inplace=True)
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
                # prediction_t = teacher(input)#Teacherモデル
                loss = loss_fn(prediction_s, target)
                # sum_test_crossloss += crossloss.item()
                #一貫性ロス(KLダイバージェンス)
                # kldivloss = kldiv((F.log_softmax(prediction_s/temperature, dim = 1)), (F.log_softmax(prediction_t/temperature, dim = 1)))
                # sum_test_kldivloss += kldivloss.item()

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
      wandb.init(entity="your_name", name=f"teacher_{args.teacher_R}_{args.teacher_T}_student_{args.student}_{args.EnDmethod}_{args.pretrain}_optimizer_{args.optimizer}_loss_{args.loss}_scheduler_{args.scheduler}_lr{args.learning_rate}_n_mels{args.n_mels}_window_size{args.window_size}_batch_{args.batch_size}_alpha{args.alpha}_temp{args.temperature}", project=args.project_name, config={"max_plot_points": 20000})
    try:
        train(args)
    finally:
        if args.wandb:
            wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, default=0.0005, help='Learning rate of model')
    parser.add_argument('--n_mels', type=int, default=30, help='Number of mels')
    parser.add_argument('--window_size', type=int, default=30, help='Window size of mel spectrogram')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--valid_steps', type=int, default=200, help='Number of validation steps')
    parser.add_argument('--project_name', type=str, default="debug", help='Project name')
    parser.add_argument('--seed', type=int, default=3407, help='Random seed')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
    parser.add_argument('--sample_rate', type=int, default=16000, help='Sample rate')
    parser.add_argument('--duration_ms', type=int, default=1000, help='Duration of audio in milliseconds')
    parser.add_argument('--teacher_R', type=str, default="resnext50", help="teacher_R model name")
    parser.add_argument('--teacher_T', type=str, default="AST", help="teacher_T Model name")
    parser.add_argument('--student', type=str, default="cnn_network4", help="Model name")
    parser.add_argument('--alpha', default=0.6, type=float, help="Alpha")
    parser.add_argument('--beta', default=0.5, type=float, help="Beta")
    parser.add_argument('--gamma', default=0.5, type=float, help="Gamma")
    parser.add_argument('--hidden_layers',type=int,default=12,help='num_hidden_layers') #default値
    parser.add_argument('--hidden_size',type=int,default=768,help='hidden_features')    #default値
    parser.add_argument('--num_attention_heads',type=int,default=12,help='heads')       #default値
    parser.add_argument('--patch_size',type=int,default=4,help='patch_size')
    parser.add_argument('--frequency_stride',type=int,default=4,help='stride for frequency')
    parser.add_argument('--time_stride',type=int,default=4,help='stride for time')
    parser.add_argument('--temperature', type=int, default=5, help="temperature")
    parser.add_argument('--amp', action='store_true', default=True, help='Use mixed precision')
    parser.add_argument('--data_parallel', action='store_true', default=False, help='Use Data_parallel')
    parser.add_argument('--loss', type=str, default="cross_entropy", help='Loss function')
    parser.add_argument('--optimizer', type=str, default="adam", help='Optimizer')
    parser.add_argument('--scheduler', type=str, default='ReduceLR', help='Learning rate scheduler')
    parser.add_argument('--wandb', action='store_true', help='Use wandb for logging')
    parser.add_argument('--EnDmethod', type=str, default='method1', help='Ensemble Distillation method')
    parser.add_argument('--pretrain', type=str, default='pretrain', help='student weight load')
    args = parser.parse_args()
    main(args)
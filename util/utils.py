import numpy as np
from sklearn.model_selection import train_test_split
import torch
import random
import os


# def stratified_train_val_split(torch_dataset, display_stratify_count = False):
#   train_indices, val_indices = train_test_split(list(range(len(torch_dataset.labels))), test_size=0.2, stratify=torch_dataset.labels)
#   train_labels = [ torch_dataset.labels[index_mask] for index_mask in train_indices]
#   val_labels = [ torch_dataset.labels[index_mask] for index_mask in val_indices]
#   if display_stratify_count:
#     print(np.unique(train_labels, return_counts=True))
#     print(np.unique(val_labels, return_counts=True))
#   return train_indices, val_indices


def device_check(display_device = False):
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    if display_device: print(f"Using device {device}")
    return device

def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): 指定したvalidationフェーズ数以上改善がない場合にトレーニングを停止
            verbose (bool): 早期終了の出力をするかどうか
            delta (float): 改善と見なされる最小の変化量
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.best_score = None
        self.epochs_no_improve = 0
        self.early_stop = False

    def __call__(self, val_loss):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.epochs_no_improve += 1
            if self.epochs_no_improve >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"早期終了: {self.patience} validationフェーズ以上損失が改善されませんでした。")
        else:
            self.best_score = score
            self.epochs_no_improve = 0

class CustomReduceLROnPlateau:
    def __init__(self, optimizer, factor=0.1, patience=20, verbose=False, min_lr=1e-7):
        self.optimizer = optimizer
        self.factor = factor
        self.patience = patience
        self.verbose = verbose
        self.min_lr = min_lr
        self.best_loss = float('inf')  # これまでの最小 val_loss
        self.num_bad_steps = 0  # 連続して最小値を更新しなかったステップ数

    def step(self, current_loss):
        # 最低値を更新した場合
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.num_bad_steps = 0  # カウンターをリセット
        else:
            # 最低値を更新できなかった場合
            self.num_bad_steps += 1

        # patience に達した場合、学習率を減少させる
        if self.num_bad_steps >= self.patience:
            for param_group in self.optimizer.param_groups:
                new_lr = param_group['lr'] * self.factor
                if new_lr >= self.min_lr:
                    param_group['lr'] = new_lr
                    if self.verbose:
                        print(f"Learning rate reduced to {new_lr}")
                else:
                    param_group['lr'] = self.min_lr
                    if self.verbose:
                        print(f"Learning rate reduced to minimum {self.min_lr}")
            self.num_bad_steps = 0  # カウンターをリセット

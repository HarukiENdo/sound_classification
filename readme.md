#不審音検出システムの環境音分類コードです。機密情報やディレクトリ構造は置き換えてありますので、適宜変更して下さい。

# 使用方法

## convert_model/py/
pytorchからtensorflow_liteへのフレームワーク変換を実行
`convert_torch_to_onnx.py` : pytorch -> onnxModel
`convert_onnx_to_tensorflowlite.ipynb` : onnxModel -> tensorflow -> tensorflow lite -> dynamic quantizationの一連の処理を実行

## model/
`models.py` : モデル定義

## utils/
`loss.py` : focal_lossの定義
`utils.py` : early_stopping, 学習率schedulerなどをカスタム定義

## t-kentei/
`ttest.ipynb` : 知識蒸留前後の有意差を確認するコード 2標本t-検定, McNemar検定, 効果量を算出

## training/
### `train_3class.py` の使用方法
`python train_3class.py --learning_rate 0.0005 --n_mels 30 --window_size 30 --project_name test`

### オプション
- `--learning_rate`, `-lr`: 学習率 (デフォルト: 0.0005)
- `--n_mels`: メルスペクトログラムの数 (デフォルト: 30)
- `--window_size`: メルスペクトログラムのウィンドウサイズ (デフォルト: 30)
- `--epochs`, `-e`: エポック数 (デフォルト: 10)
- `--project_name`: プロジェクト名
- `--seed`: ランダムシード (デフォルト: 3407)
- `--batch_size`: バッチサイズ (デフォルト: 1024)
- `--sample_rate`: サンプルレート (デフォルト: 16000)
- `--duration_ms`: 音声の長さ (ミリ秒, デフォルト: 1000)
- `--model`: モデル名 (デフォルト: "resnet34")
- `--wandb`: Weights & Biases を使用する場合に指定

## `experiments.sh` の使用方法
`experiments.sh` は複数のハイパーパラメータ設定で実験を実行するためのスクリプトです。
初回実行時は `chmod +x experiments.sh` を実行して権限を付与してください。

### 実行方法
./experiments_EnD.sh <project_name>
./experiments.sh <project_name>

## `experiments.sh` の使用方法
使用方法はexperiments_EnD.shと同じです。
同一のGPUで複数の実験を行うことができますが，使用するマシンのメモリを気にする場合は，
`max_parallel_processes` を適宜変更してください。
また，CUDA_VISIBLE_DEVICESも適宜変更してください。

## TODO
- [x] 学習率のスケジューラを選択できるようにコード修正
- [x] modelによってバッチサイズを制限
- [x] 事前処理の見直し
- [x] モデル保存のタイミングを修正する
- [x] onnx変換用のコードを作成（onnxで推論できるか確認）
- [x] half, quantizeの確認
- [x] データ拡張手法を試す（SpecAugmentなど）
- [x] 多種多様なモデルアーキテクチャで検証
- [x] ハイパーパラメータの最適化を行う
- [x] 推論速度を確認する
- [x] モデルの軽量化・高速化を試みる
- [ ] 4層のCNNModelをdepthwise separable Convに置き換える
- [ ] 入力形状の最適化(計算効率を意識した)
- [ ] リアルタイム実証実験
- [ ] READMEを充実させる


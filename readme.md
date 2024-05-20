# 使用方法
## `train.py` の使用方法
`python train.py --learning_rate 0.0005 --n_mels 30 --window_size 30 --project_name test`

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
`nohup ./experiments.sh <experiment_name> &`

## `experiments2.sh` の使用方法
使用方法はexperiments.shと同じです。
同一のGPUで複数の実験を行うことができますが，使用するマシンのメモリを気にする場合は，
`max_parallel_processes` を適宜変更してください。
また，CUDA_VISIBLE_DEVICESも適宜変更してください。


## TODO
- [ ] 学習率のスケジューラを選択できるようにコード修正
- [x] modelによってバッチサイズを制限
- [ ] 事前処理の見直し
- [x] モデル保存のタイミングを修正する
- [ ] onnx変換用のコードを作成（onnxで推論できるか確認）
  - [ ] half, quantizeの確認
- [ ] 推論スクリプトを作成する
- [ ] データ拡張手法を試す（SpecAugmentなど） 
- [ ] 他のモデルアーキテクチャを試す（EfficientNetなど）
- [ ] ハイパーパラメータの最適化を行う
- [ ] モデルの軽量化・高速化を試みる
- [ ] 推論スクリプトを作成する
- [ ] READMEを充実させる

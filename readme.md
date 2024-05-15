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
`./experiments.sh <experiment_name>`


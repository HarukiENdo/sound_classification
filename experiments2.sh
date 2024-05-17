project_name=$1
if [ ! -d ./experiment/$project_name ]; then
    mkdir -p ./experiment/$project_name
fi

# GPUメモリを効率的に使用するために、複数のプロセスを並列に実行
max_parallel_processes=20

for loss in cross_entropy focal_loss; do
    for model in cnn_network1 cnn_network2 cnn_network3 cnn_network4; do
        CUDA_VISIBLE_DEVICES=0 python train.py --wandb --loss $loss --model $model --project_name $project_name &> experiment/$project_name/train_${model}_${loss}.log &
        
        # 同時に実行するプロセス数を制限
        if [ $(jobs | wc -l) -ge $max_parallel_processes ]; then
            # 最大プロセス数に達したら、いずれかのプロセスが終了するまで待機
            wait -n
        fi
    done
done

# 残りのプロセスが終了するまで待機
wait
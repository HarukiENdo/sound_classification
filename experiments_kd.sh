project_name=$1
if [ ! -d ./experiment_nd/$project_name ]; then
    mkdir -p ./experiment_kd/$project_name
fi

#実験条件の追加
# GPUメモリを効率的に使用するために、複数のプロセスを並列に実行
# max_parallel_processes=5
for loss in cross_entropy; do
    for optimizer in adamw; do
        for student in Student_s; do
            for learning_rate in 0.0005; do
                for window_size in 15 90; do
                    for alpha in 0.9; do
                        for temperature in 10; do
                            CUDA_VISIBLE_DEVICES=1 python train.py --wandb --loss $loss --optimizer $optimizer --student $student --learning_rate $learning_rate --window_size $window_size --alpha $alpha --temperature $temperature --project_name $project_name &> experiment_kd/$project_name/train_${student}_${loss}_${optimizer}_${learning_rate}_${window_size}_${alpha}_${temperature}.log &
                            # # 同時に実行するプロセス数を制限
                            # if [ $(jobs | wc -l) -ge $max_parallel_processes ]; then
                            #     # 最大プロセス数に達したら、いずれかのプロセスが終了するまで待機
                            #     wait -n
                            # fi
                            wait
                        done
                    done
                done
            done
        done
    done
done

# wait
project_name=$1
if [ ! -d ./experiment/$project_name ]; then
    mkdir -p ./experiment/$project_name
fi

#実験条件の追加
max_parallel_processes=5
for model in depthwise_CNN2; do
    for learning_rate in 5e-4 1e-4; do
        for n_mels in 40; do
            for window_size in 52; do
                for batch_size in 512; do
                    for loss in cross_entropy; do
                        for optimizer in adamw; do
                            for scheduler in ReduceLR; do
                                nohup python ./training/train_3class.py --wandb --learning_rate $learning_rate --n_mels $n_mels --window_size $window_size --batch_size $batch_size --model $model --loss $loss --optimizer $optimizer --scheduler $scheduler --project_name $project_name > experiment/$project_name/${model}_optimizer_${optimizer}_loss_${loss}_scheduler_${scheduler}_lr${learning_rate}_n_mels${n_mels}_window_size${window_size}_batch_size${batch_size}.log 2>&1 &
                                # 同時に実行するプロセス数を制限
                                if [ $(jobs | wc -l) -ge $max_parallel_processes ]; then
                                    # 最大プロセス数に達したら、いずれかのプロセスが終了するまで待機
                                    wait -n
                                fi  
                            done                                 
                        done
                    done
                done
            done
        done
    done
done

wait


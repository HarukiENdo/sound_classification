project_name=$1
if [ ! -d ./experiment_kd/$project_name ]; then
    mkdir -p ./experiment_kd/EnD/$project_name
fi

for teacher_R in resnext50; do
    for teacher_T in AST; do
        for student in CNNNetwork4; do
            for EnDmethod in method1; do
                for pretrain in pretrain; do
                    for optimizer in adam; do
                        for loss in cross_entropy; do
                            for scheduler in ReduceLR; do
                                for learning_rate in 1e-4; do # 0.0005 1e-4
                                    for n_mels in 30; do
                                        for window_size in 30; do
                                            for batch_size in 512; do
                                                for alpha in 0.6; do #0.6で同程度の寄与率
                                                    for temperature in 6; do #6
                                                        nohup python ./training/train_EnD.py --wandb --project_name $project_name --teacher_R $teacher_R --teacher_T $teacher_T --EnDmethod $EnDmethod --pretrain $pretrain --optimizer $optimizer --loss $loss --scheduler $scheduler --learning_rate $learning_rate --n_mels $n_mels --window_size $window_size --batch_size $batch_size --alpha $alpha --temperature $temperature > experiment_kd/EnD/$project_name/teacher_${teacher_R}_${teacher_T}_student_${student}_${EnDmethod}_${pretrain}_optimizer_${optimizer}_loss_${loss}_scheduler_${scheduler}_lr${learning_rate}_n_mels${n_mels}_window_size${window_size}_batch_${batch_size}_alpha${alpha}_temp${temperature}.log 2>&1 &
                                                        pid=$!  # 直前のプロセスIDを取得
                                                        wait $pid  # プロセスが終了するのを待機
                                                    done
                                                done
                                            done
                                        done
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done
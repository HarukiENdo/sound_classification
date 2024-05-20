experiment=$1
if [ ! -d ./experiment/$experiment ]; then
    mkdir -p ./experiment/$experiment
fi

#実験条件の追加
for loss in cross_entropy focal_loss; do
    for optimizer in  adam adamw; do
        for model in cnn_network4 resnet18 resnet34 resnet50; do
            for learning_rate in 0.00005 0.0005; do
                for window_size in 15 40 90; do
                    CUDA_VISIBLE_DEVICES=0 python train.py --wandb --loss $loss --optimizer $optimizer --model $model --learning_rate $learning_rate --window_size $window_size --project_name $project_name &> experiment/$project_name/train_${model}_${loss}_${optimizer}_${learing_rate}_${window_size}.log &
                    wait
                done
            done
        done
    done
done


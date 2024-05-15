experiment=$1
if [ ! -d $experiment ]; then
    mkdir -p $experiment
fi

# for lr in 0.0005 0.005; do
#     for n_mels in 30 40; do
#         for window_size in 30 40; do
#             python train.py --wandb --learning_rate $lr --n_mels $n_mels --window_size $window_size --project_name $experiment &> $experiment/train_lr${lr}_nmels${n_mels}_ws${window_size}.log &
#             wait
#         done
#     done
# done

for model in cnn_network1 cnn_network2 cnn_network3 cnn_network4 resnet18; do
    python train.py --wandb --loss focal_loss --model $model --project_name $experiment &> $experiment/train_${model}_focal_loss.log &
    wait
done


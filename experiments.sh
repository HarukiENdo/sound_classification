experiment=$1
if [ ! -d $experiment ]; then
    mkdir -p $experiment
fi

for lr in 0.0005 0.005; do
    for n_mels in 30 40; do
        for window_size in 30 40; do
            python train.py --wandb --learning_rate $lr --n_mels $n_mels --window_size $window_size --project_name $experiment &> $experiment/train_lr${lr}_nmels${n_mels}_ws${window_size}.log &
            wait
        done
    done
done

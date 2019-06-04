#!/bin/sh

mkdir -p log
#for arch in alexnet; do
for arch in mlp1 mlp3 inception alexnet; do
for task in random_label random_pixel shuffle_pixel gaussian_pixel normal; do
for step in 100000; do
    lr=0.01
    if [ "$arch" == "inception" ]; then
        lr=0.1
    fi
    model=$arch
    if [ "$arch" == "mlp1" ]; then
        model="mlp"
        spec="--mlp-spec=512"
    elif [ "$arch" == "mlp3" ]; then
        model="mlp"
        spec="--mlp-spec=512x512x512"
    fi
    suf="$arch.$task.$step"
    nohup srun --job-name=gn.$suf --partition=1080Ti,1080Ti_slong --cpus-per-task=2 --gres=gpu:1 -n1 python -u train.py --arch=$model $spec --learning-rate=$lr --label-corrupt-prob=1.0 --steps=$step --task=$task 2>&1 >log/log.$suf & 
done
done
done

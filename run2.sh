#!/bin/bash

PORT=55623
GPUS=2
SCRIPT=test.py
LR=0.05
DATASET=cifar100

CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=$GPUS --master_port=$PORT $SCRIPT \
  --lr $LR --dataset $DATASET --model moce50 \
  --amp --run-name "moce_final#3" --k 8

CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=$GPUS --master_port=$PORT $SCRIPT \
  --lr $LR --dataset $DATASET --model moce50 \
  --amp --run-name "moce_final#4" --k 8
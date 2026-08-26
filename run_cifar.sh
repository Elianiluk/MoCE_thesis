#!/bin/bash

PORT=55624
GPUS=2
SCRIPT=test.py
LR=0.05
DATASET=cifar100

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=$GPUS --master_port=$PORT $SCRIPT \
  --lr $LR --dataset $DATASET --model moce50 \
  --amp --run-name "moce_final#1" --k 8

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=$GPUS --master_port=$PORT $SCRIPT \
  --lr $LR --dataset $DATASET --model moce50 \
  --amp --run-name "moce_final#2" --k 8

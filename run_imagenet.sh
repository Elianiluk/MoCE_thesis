#!/bin/bash
#SBATCH --job-name=batch512lr0.1
#SBATCH --partition=gpu-rtx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:8
#SBATCH --time=48:00:00
#SBATCH --output=times.out
#SBATCH --error=times.err
#SBATCH --mail-type=END,FAIL,BEGIN
#SBATCH --mail-user=

set -euo pipefail

# Load required modules (DO NOT load python module)
module --ignore_cache load gcc hdf5/serial StdEnv cuda

# Force SLURM to use your user python (where torch is installed)
export PATH=$HOME/.local/bin:$HOME/bin:$PATH

cd $SLURM_SUBMIT_DIR

export OMP_NUM_THREADS=8
export NCCL_DEBUG=WARN
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500

# =========================
# Hyperparameters
# =========================

EPOCHS="${EPOCHS:-120}"
LR="${LR:-0.0125}"
MOMENTUM="${MOMENTUM:-0.9}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"
ROUTER_LR_MULT="${ROUTER_LR_MULT:-0.5}"
BALANCE_WEIGHT="${BALANCE_WEIGHT:-0.005}"
BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-64}"
MODEL="${MODEL:-pix50}"
DATASET="${DATASET:-imagenet}"
SCHEDULER="${SCHEDULER:-auto}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-5}"
MIN_LR="${MIN_LR:-0.0}"
STEP_SIZE="${STEP_SIZE:-30}"
GAMMA="${GAMMA:-0.1}"
NUM_WORKERS="${NUM_WORKERS:-6}"
RUN_NAME="resnet50-moce-120epochs-batch512-lr0.1#1"

IMAGENET_ROOT="/home/evgenyn/project/imagenet/"
CIFAR_ROOT="/data"

AMP="${AMP:-1}"

# SLURM: count GPUs same way as before
NUM_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
TOTAL_BATCH_SIZE=$((NUM_GPUS * BATCH_SIZE_PER_GPU))

echo "================================================================================"
echo "Distributed ResNet-50 Training (SLURM)"
echo "================================================================================"
echo "  Dataset: $DATASET"
echo "  Model: $MODEL"
echo "  Epochs: $EPOCHS"
echo "  Batch Size: $TOTAL_BATCH_SIZE ($BATCH_SIZE_PER_GPU per GPU)"
echo "  GPUs: $NUM_GPUS"
# echo "  Scheduler: $SCHEDULER (step-size=$STEP_SIZE, gamma=$GAMMA)"
echo "  ImageNet Root: $IMAGENET_ROOT"
echo "================================================================================"

CMD=(
    torchrun
    --nnodes=1
    --nproc_per_node="$NUM_GPUS"
    --master_addr="$MASTER_ADDR"
    --master_port="$MASTER_PORT"
    "$SLURM_SUBMIT_DIR/test.py"
    --epochs "$EPOCHS"
    --batch-size "$TOTAL_BATCH_SIZE"
    --lr "$LR"
    --momentum "$MOMENTUM"
    --weight-decay "$WEIGHT_DECAY"
    --router-lr-mult "$ROUTER_LR_MULT"
    --balance-weight "$BALANCE_WEIGHT"
    --model "$MODEL"
    --dataset "$DATASET"
    --scheduler "$SCHEDULER"
    --warmup-epochs "$WARMUP_EPOCHS"
    --min-lr "$MIN_LR"
    --step-size "$STEP_SIZE"
    --gamma "$GAMMA"
    --num-workers "$NUM_WORKERS"
    --cifar-root "$CIFAR_ROOT"
    --imagenet-root "$IMAGENET_ROOT"
    --run-name "$RUN_NAME"
)

if [[ "$AMP" == "1" ]]; then
    CMD+=(--amp)
fi

echo ">>> Running training"

MAX_ATTEMPTS="${MAX_ATTEMPTS:-3}"
attempt=1
set +e
while true; do
    "${CMD[@]}"
    status=$?
    if [[ $status -eq 0 ]]; then
        break
    fi
    if [[ $attempt -ge $MAX_ATTEMPTS ]]; then
        echo ">>> Training failed after $attempt attempt(s) with exit code $status, giving up."
        exit $status
    fi
    echo ">>> Training attempt $attempt failed with exit code $status, resuming from checkpoint (attempt $((attempt + 1))/$MAX_ATTEMPTS)..."
    attempt=$((attempt + 1))
    sleep 30
done
set -e

echo "================================================================================"
echo "Training Complete!"
echo "================================================================================"

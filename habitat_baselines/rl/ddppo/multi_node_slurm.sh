#!/bin/bash
#SBATCH --job-name=ddppo_nav
#SBATCH --gpus-per-task 1
#SBATCH --cpus-per-task 8
#SBATCH --ntasks 8
#SBATCH --signal=USR1@90
#SBATCH --requeue
#SBATCH --partition=short
#SBATCH --constraint=a40
#SBATCH -x deebot,nestor

set -x

export MAGNUM_LOG="quiet"
export HABITAT_SIM_LOG="quiet"

export NCCL_SOCKET_IFNAME=""
export NCCL_DEBUG=INFO

MAIN_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
export MAIN_ADDR

log_dir="logs/${EXP_NAME}"
echo $log_dir

srun python -u ./habitat_baselines/run.py \
    --exp-config $CONFIG \
    --run-type $RUN_MODE \
    TENSORBOARD_DIR "$log_dir/tb" \
    VIDEO_DIR "$log_dir/video_dir" \
    EVAL_CKPT_PATH_DIR "$log_dir/ckpt" \
    CHECKPOINT_FOLDER "$log_dir/ckpt" \
    CONFIG_DUMP_FOLDER "$log_dir"

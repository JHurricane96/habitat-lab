#!/bin/bash
#SBATCH --job-name=ddppo_nav
#SBATCH --gpus-per-task 1
#SBATCH --cpus-per-task 8
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 4
#SBATCH --signal=USR1@90
#SBATCH --requeue
#SBATCH --partition=short
#SBATCH --constraint=a40

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

set -x

export MAGNUM_LOG="quiet"
export HABITAT_SIM_LOG="quiet"

# export NCCL_SOCKET_IFNAME=""
export NCCL_DEBUG=INFO

MAIN_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
export MAIN_ADDR

log_dir="logs/${EXP_NAME}"
echo $log_dir

srun python -u ./habitat-baselines/habitat_baselines/run.py \
    --exp-config $CONFIG \
    --run-type $RUN_MODE \
    habitat_baselines.writer_type=wb \
    habitat_baselines.wb.project_name="cat_nav" \
    habitat_baselines.wb.entity="language-rearrangement" \
    habitat_baselines.wb.run_name=\"$WB_EXP_NAME\"
    # $EXTRA_FLAGS
    # habitat_baselines.tensorboard_dir="$log_dir/tb" \
    # habitat_baselines.video_dir="$log_dir/video_dir" \
    # habitat_baselines.eval_ckpt_path_dir="$log_dir/ckpt" \
    # habitat_baselines.checkpoint_folder="$log_dir/ckpt" \

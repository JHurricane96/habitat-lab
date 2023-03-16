#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

export MAGNUM_LOG="quiet"
export HABITAT_SIM_LOG="quiet"

exp_id=$(cat logs/current_run.txt)
exp_id_split=(${exp_id//"__"/ })
exp_name="${exp_id_split[@]:1}"
log_dir="logs/${exp_id}"

# rm -rf $log_dir/tb

if [[ "$RUN_MODE" == "eval" ]]; then
    EVAL_FLAGS="habitat_baselines.num_environments=1 habitat_baselines.rl.ppo.num_mini_batch=1"
else
    EVAL_FLAGS=""
fi

# export NCCL_SOCKET_IFNAME=""
# export NCCL_DEBUG=INFO

set -x
# python -u -m torch.distributed.launch \
#     --use_env \
#     --nproc_per_node 1 \
#     --master_port 29501 \
    # --exp-config habitat_baselines/config/rearrange/hab/ddppo_nav_pick.yaml \
# py-spy record --idle --function --native --subprocesses --rate 10 --output data/profile/scope.speedscope --format speedscope -- \
python -u \
    ./habitat-baselines/habitat_baselines/run.py \
    --exp-config $CONFIG \
    --run-type $RUN_MODE \
    $EVAL_FLAGS
    # habitat_baselines.tensorboard_dir="$log_dir/tb" \
    # habitat_baselines.video_dir="$log_dir/video_dir" \
    # habitat_baselines.eval_ckpt_path_dir="$log_dir/ckpt" \
    # habitat_baselines.checkpoint_folder="$log_dir/ckpt" \
    # $EXTRA_FLAGS \
    # WB.PROJECT_NAME "test" \
    # WB.ENTITY "language-rearrangement" \
    # WB.RUN_NAME "$exp_name" \
# |& tee $log_dir/out.txt

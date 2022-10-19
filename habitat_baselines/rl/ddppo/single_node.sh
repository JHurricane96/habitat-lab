#!/bin/bash

export MAGNUM_LOG="quiet"
export HABITAT_SIM_LOG="quiet"

exp_id=$(cat logs/current_run.txt)
exp_id_split=(${exp_id//"__"/ })
exp_name="${exp_id_split[@]:1}"
log_dir="logs/${exp_id}"

# rm -rf $log_dir/tb

set -x
# python -u -m torch.distributed.launch \
#     --use_env \
#     --nproc_per_node 1 \
#     --master_port 29501 \
    # --exp-config habitat_baselines/config/rearrange/hab/ddppo_nav_pick.yaml \
python -u \
    habitat_baselines/run.py \
    --exp-config $CONFIG \
    --run-type $RUN_MODE \
    TENSORBOARD_DIR "$log_dir/tb" \
    VIDEO_DIR "$log_dir/video_dir" \
    EVAL_CKPT_PATH_DIR "$log_dir/ckpt" \
    CHECKPOINT_FOLDER "$log_dir/ckpt" \
    CONFIG_DUMP_FOLDER "$log_dir" \
    # WB.PROJECT_NAME "test" \
    # WB.ENTITY "language-rearrangement" \
    # WB.RUN_NAME "$exp_name" \
# |& tee $log_dir/out.txt

#! /usr/bin/env bash

# exp_suffix=floorplanner_test

# dt=
# if [[ $1 ]]
# then
#   dt=$1
# else
#   dt=$(date '+%Y_%m_%d_%H_%M_%S')
# fi

# export EXP_NAME="${dt}_${exp_suffix}"

set -x

# export RUN_MODE=eval
export RUN_MODE=train
# export HABITAT_ENV_DEBUG=1
# export OC_CAUSE=1

# export CONFIG=habitat-baselines/habitat_baselines/config/rearrange/rl_skill.yaml
cfg=habitat-baselines/habitat_baselines/config/rearrange/rl_skill.yaml

export EXTRA_FLAGS="benchmark/rearrange=cat_nav_to_obj"
# export EXTRA_FLAGS="${EXTRA_FLAGS} habitat.dataset.data_path=data/datasets/floorplanner/rearrange/\{split\}/cat_rearrange_floorplanner_5.json.gz"
export EXTRA_FLAGS="${EXTRA_FLAGS} habitat.dataset.data_path=data/datasets/floorplanner/rearrange/minitrain/cat_rearrange_floorplanner.json.gz"
# export EXTRA_FLAGS="${EXTRA_FLAGS} habitat_baselines.rl.policy.action_distribution_type=gaussian"
export EXTRA_FLAGS="${EXTRA_FLAGS} habitat_baselines.rl.policy.action_distribution_type=categorical"
# export EXTRA_FLAGS="${EXTRA_FLAGS} habitat.simulator.requires_textures=False"
# export EXTRA_FLAGS="${EXTRA_FLAGS} habitat.simulator.concur_render=False"
# export EXTRA_FLAGS="${EXTRA_FLAGS} habitat.task.spawn_max_dists_to_obj=3.0 habitat.task.biased_init=True"
# export EXTRA_FLAGS="habitat.task.spawn_max_dists_to_obj=2.5"
# export EXTRA_FLAGS="${EXTRA_FLAGS} habitat_baselines.rl.ddppo.pretrained_encoder=True habitat_baselines.rl.policy.ovrl=True habitat_baselines.rl.ddppo.pretrained_weights=resnet50_32bp_ovrl.pth habitat_baselines.rl.ddppo.backbone=resnet50"

export EXP_NAME=$(cat logs/current_run.txt)
if [[ ! -z $1 ]]; then
    export WB_EXP_NAME=$1
else
    export WB_EXP_NAME=$EXP_NAME
fi

log_dir="logs/${EXP_NAME}"
python -u output_conf.py \
    --exp-config $cfg \
    --cfg-out-dir "$log_dir" \
    habitat_baselines.tensorboard_dir="$log_dir/tb" \
    habitat_baselines.video_dir="$log_dir/video_dir" \
    habitat_baselines.eval_ckpt_path_dir="$log_dir/ckpt" \
    habitat_baselines.checkpoint_folder="$log_dir/ckpt" \
    $EXTRA_FLAGS

cfg_file=$(basename $cfg)
export CONFIG="logs/${EXP_NAME}/${cfg_file}"

slurm_log_dir=logs/$EXP_NAME/slurm
mkdir -p $slurm_log_dir
sbatch \
    --error $slurm_log_dir/err \
    --output $slurm_log_dir/out \
    ./habitat-baselines/habitat_baselines/rl/ddppo/multi_node_slurm.sh \
|& tee $slurm_log_dir/job.txt


# ./habitat-baselines/habitat_baselines/rl/ddppo/single_node.sh

#! /usr/bin/env bash

set -x

# export OC_CAUSE=1

py-spy record --idle --function --native --subprocesses --rate 5 --output data/profile/scope.speedscope --format speedscope -- \
python -u habitat-baselines/habitat_baselines/time_sim.py \
    --cfg-path habitat-baselines/habitat_baselines/config/rearrange/rl_skill.yaml \
    --num-eps 25 \
    benchmark/rearrange=cat_nav_to_obj \
    "habitat.dataset.data_path=data/datasets/floorplanner/rearrange/minitrain/cat_rearrange_floorplanner.json.gz"
    # habitat.environment.max_episode_steps=3 \
    # "habitat.dataset.data_path=data/datasets/rearrange_floorplanner/\{split\}/cat_rearrange_floorplanner_5_60-80_fast-scene.json.gz"
    # "habitat.dataset.data_path=data/datasets/rearrange_floorplanner/\{split\}/cat_rearrange_floorplanner_5_60-80.json.gz"
    # "habitat.dataset.data_path=data/datasets/rearrange_floorplanner/\{split\}/cat_rearrange_floorplanner_5_60-80_no-tmpl-fix.json.gz"
    # habitat.simulator.concur_render=False \
    # habitat.environment.max_episode_steps=400 \
    # habitat.simulator.sleep_dist=-1.0 \
    # benchmark/rearrange=cat_nav_to_obj \
    # "habitat.dataset.data_path=data/datasets/rearrange_floorplanner/\{split\}/cat_rearrange_floorplanner_5_1-1.json.gz"

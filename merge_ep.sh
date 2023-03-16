#! /usr/bin/env bash

python -u lang_rearrange_scripts/merge_episodes.py \
    --train-episodes data/datasets/floorplanner/rearrange/minitrain/cat_rearrange_floorplanner.json.gz \
    --episodes-dir data/datasets/floorplanner/rearrange/split/minival/final_split/train \
    --out-path data/datasets/floorplanner/rearrange/minival/cat_rearrange_floorplanner.json.gz

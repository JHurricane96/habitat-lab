#! /usr/bin/env bash

set -x

python -u habitat-lab/habitat/datasets/rearrange/modify_episodes_for_object_rearrange.py \
    --source_data_dir data/datasets/rearrange_floorplanner/ \
    --target_data_dir data/datasets/rearrange_floorplanner/ \
    --obj_category_mapping_file data/anns/objects_info.csv \
    --rec_category_mapping_file data/anns/categories_v0.2.0.csv \
    --source_episodes_tag rearrange_floorplanner \
    --target_episodes_tag cat_rearrange_floorplanner \
    --add_viewpoints
    # --debug_viz

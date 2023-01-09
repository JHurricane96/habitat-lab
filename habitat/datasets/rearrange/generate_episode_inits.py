#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
import yaml

import habitat
from habitat_sim.utils.viz_utils import observation_to_image


# def generate_inits(cfg_path, opts):
#     config = habitat.get_config(cfg_path, opts)
#     with habitat.Env(config=config) as env:
#         env.reset()
#         for i in tqdm(range(env.number_of_episodes)):
#             if i % 100 == 0:
#                 # Print the dataset we are generating initializations for. This
#                 # is useful when this script runs for a long time and we don't
#                 # know which dataset the job is for.
#                 print(cfg_path, config.DATASET.DATA_PATH)
#             for _ in tqdm(range(10000)):
#                 env._sim.set_robot_base_to_random_point()
#             env.reset()

def generate_inits(cfg_path, opts):
    config = habitat.get_config(cfg_path, opts)
    out_dir = Path("imgs")
    with habitat.Env(config=config) as env:
        env.reset()
        for i in tqdm(range(env.number_of_episodes)):
            if i % 100 == 0:
                # Print the dataset we are generating initializations for. This
                # is useful when this script runs for a long time and we don't
                # know which dataset the job is for.
                print(cfg_path, config.DATASET.DATA_PATH)
            for j in tqdm(range(10)):
                obs = env._task.reset(env.current_episode)
                imgs = {}
                imgs["rgb"] = observation_to_image(obs["robot_head_rgb"], "color")
                imgs["depth"] = observation_to_image(obs["robot_head_depth"].squeeze(), "depth", 1)
                imgs["semantic"] = observation_to_image(obs["robot_head_semantic"], "semantic")
                imgs_out_dir = out_dir/f"{i:05d}_{j:03d}"
                imgs_out_dir.mkdir(exist_ok=True, parents=True)
                for key, img in imgs.items():
                    fname = imgs_out_dir/f"{key}.png"
                    img.save(str(fname))
                np.save(str(imgs_out_dir/"semantic.npy"), obs["robot_head_semantic"])
                with open(imgs_out_dir/"semantic_map.yml", "w") as f:
                    yaml.dump(env._task.iids, f)
            if i == 10:
                break
            env.reset()


parser = argparse.ArgumentParser()
parser.add_argument("--cfg-path", type=str, required=True)
parser.add_argument(
    "opts",
    default=None,
    nargs=argparse.REMAINDER,
    help="Modify config options from command line",
)
args = parser.parse_args()

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

generate_inits(args.cfg_path, args.opts)

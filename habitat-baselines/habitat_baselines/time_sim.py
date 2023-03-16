import argparse
import time
import random

import habitat
from gym import spaces
from habitat_baselines.config.default import get_config
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--cfg-path", type=str, required=True)
parser.add_argument("--num-eps", type=int, default=25)
parser.add_argument(
    "opts",
    default=None,
    nargs=argparse.REMAINDER,
    help="Modify config options from command line",
)
args = parser.parse_args()


def get_ac_cont(env):

    # ac_names = tuple(env.action_space.keys())
    ac_names = ["base_velocity"]
    ac_args = {}
    for ac_name in ac_names:
        ac_args.update(env.action_space.spaces[ac_name].sample())
    return {"action": "base_velocity", "action_args": ac_args}

def get_ac_disc(env):
    ac_names = list(env.action_space.keys())
    ac_names.remove("stop")
    return {"action": random.choice(ac_names), "action_args": {}}

def get_ac(env):
    if "stop" in env.action_space.spaces:
        return get_ac_disc(env)
    else:
        return get_ac_cont(env)

def set_episode(env, episode_id):
    episode = [ep for ep in env.episodes if ep.episode_id == episode_id][0]
    env.current_episode = episode

config = get_config(args.cfg_path, args.opts)
start = time.perf_counter()
with habitat.Env(config=config) as env:
    start_time = time.perf_counter() - start
    print("Starting sample")
    # start = time.perf_counter()
    num_frames = 0
    start_ep = 0
    set_episode(env, "1159")
    for i in range(args.num_eps):
        if i == start_ep:
            start = time.perf_counter()
        env.reset()
        while not env.episode_over:
            env.step(get_ac(env))
            if i >= start_ep:
                num_frames += 1
        if i >= start_ep:
            print("FPS", num_frames / (time.perf_counter() - start))
    end = time.perf_counter()
    print("Done")

print("Start time ", start_time)
print("Step time ", end - start)
print("FPS", num_frames / (end - start))
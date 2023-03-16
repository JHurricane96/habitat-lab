import argparse
from pathlib import Path
from typing import TYPE_CHECKING

from omegaconf import OmegaConf

import habitat
from habitat_baselines.config.default import get_config

if TYPE_CHECKING:
    from omegaconf import DictConfig

parser = argparse.ArgumentParser()

parser.add_argument(
    "--exp-config",
    type=Path,
    required=True,
    help="path to config yaml containing info about experiment",
)
parser.add_argument(
    "--cfg-out-dir",
    type=Path,
    required=True,
)
parser.add_argument(
    "opts",
    default=None,
    nargs=argparse.REMAINDER,
    help="Modify config options from command line",
)

def dump_config(config: "DictConfig", config_path: Path) -> None:
    with open(config_path, "w") as f:
        OmegaConf.save(config, f)

def main(exp_config: Path, cfg_out_dir: Path, opts=None) -> None:
    config = get_config(str(exp_config), opts)
    dump_config(config, cfg_out_dir/exp_config.name)

if __name__ == "__main__":
    args = parser.parse_args()
    main(**vars(args))

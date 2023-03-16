import json
import argparse
from pathlib import Path
from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument("--config-dirs", type=Path, nargs="+", default=[
    Path("data/objects/amazon_berkeley/configs"),
    Path("data/objects/google_object_dataset/configs"),
    Path("data/fphab-v0.2.0/configs/objects"),
])

def main(args):
    for config_dir in args.config_dirs:
        config_files = list(config_dir.glob("*.json"))
        for config_file in tqdm(config_files):
            with open(config_file, "r") as f:
                config = json.load(f)

            config["use_bounding_box_for_collision"] = False

            with open(config_file, "w") as f:
                json.dump(config, f)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)

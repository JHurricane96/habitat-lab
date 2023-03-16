#! /usr/bin/env bash
#SBATCH --output=ep_gen.out
#SBATCH --error=ep_gen.err
#SBATCH --job-name=ep_gen
#SBATCH --gpus-per-task 1
#SBATCH --nodes 1
#SBATCH --cpus-per-task 8
#SBATCH --ntasks-per-node 1
#SBATCH --signal=USR1@90
#SBATCH --requeue
#SBATCH --partition=short
#SBATCH --constraint=a40

export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet

set -x
# srun python -u habitat-lab/habitat/datasets/rearrange/run_episode_generator.py \
python -u habitat-lab/habitat/datasets/rearrange/run_episode_generator.py \
    --config habitat-lab/habitat/datasets/rearrange/configs/rearrange_floorplanner.yaml \
    --run \
    --verbose \
    --num-episodes 1 \
    --seed 0 \
    --out data/datasets/rearrange_floorplanner/microtrain/rearrange_floorplanner.json.gz \
    --limit-scene-set microtrain \
    --limit-scene 106365897_174225972
    # --limit-scene-set 102815859
    # --num-episodes 100 \
    # --seed $EP_GEN_IDX \
    # --out data/datasets/rearrange_floorplanner/val_split/rearrange_floorplanner_1k_${EP_GEN_IDX}.json.gz \
    # --limit-scene-set $SCENE
    # --list

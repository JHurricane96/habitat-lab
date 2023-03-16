#! /usr/bin/env bash
#SBATCH --output=ep_gen.out
#SBATCH --error=ep_gen.err
#SBATCH --job-name=ep_gen
#SBATCH --gpus 1
#SBATCH --nodes 1
#SBATCH --cpus-per-gpu 8
#SBATCH --array=0-3
#SBATCH --signal=USR1@90
#SBATCH --requeue
#SBATCH --partition=short
#SBATCH --constraint=a40

export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet

set -x

SCENES=($SCENES)
echo ${SCENES[@]}

i=$SLURM_ARRAY_TASK_ID
echo "i=${i}"
if [[ -z "${SCENES[$i]}" ]]; then
    exit 0
fi
abs_idx=$((EP_GEN_IDX+i))
echo "abs_idx=${abs_idx}"
srun -o ep_gen_logs/$SPLIT/${abs_idx}.out -e ep_gen_logs/$SPLIT/${abs_idx}.err \
python -u habitat-lab/habitat/datasets/rearrange/run_episode_generator.py \
    --config habitat-lab/habitat/datasets/rearrange/configs/rearrange_floorplanner.yaml \
    --run \
    --verbose \
    --num-episodes $NUM_EPISODES \
    --seed $abs_idx \
    --out data/datasets/floorplanner/rearrange/split/mini${SPLIT}/rearrange_floorplanner_${abs_idx}.json.gz \
    --limit-scene-set mini${SPLIT} \
    --limit-scene ${SCENES[$i]}
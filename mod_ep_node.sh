#! /usr/bin/env bash
#SBATCH --output=mod_ep.out
#SBATCH --error=mod_ep.err
#SBATCH --job-name=mod_ep
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
# abs_idx=$((EP_GEN_IDX+i))
scene=${SCENES[$i]}
echo "i=${i}"
# echo "abs_idx=${abs_idx}"
echo "scene=${scene}"

data_dir=data/datasets/floorplanner/rearrange/split/mini${SPLIT}/final_split
file=$data_dir/train/rearrange_floorplanner_$scene.json.gz
if [[ ! -f $file ]]; then
    exit 0
fi
srun -o mod_ep_logs/$SPLIT/${scene}.out -e mod_ep_logs/$SPLIT/${scene}.err \
python -u habitat-lab/habitat/datasets/rearrange/modify_episodes_for_object_rearrange.py \
    --source_data_dir $data_dir \
    --target_data_dir $data_dir \
    --obj_category_mapping_file data/anns/objects_info.csv \
    --rec_category_mapping_file data/anns/categories_v0.2.0.csv \
    --source_episodes_tag rearrange_floorplanner_$scene \
    --target_episodes_tag cat_rearrange_floorplanner_$scene \
    --add_viewpoints
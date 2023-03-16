#! /usr/bin/env bash

set -x

# scenes=(2 4 5 6 8 9 10 12 13 14)
scenes=(1 2 3 4)
num_scenes=${#scenes[@]}
export SPLIT=val
mkdir -p mod_ep_logs/$SPLIT

for (( i=0; i<$num_scenes; i+=4 ))
do
    export EP_GEN_IDX=$i
    export SCENES="${scenes[@]:$i:4}"
    sbatch \
        --error "mod_ep_logs/${SPLIT}/slurm_${i}_%a.err" \
        --output "mod_ep_logs/${SPLIT}/slurm_${i}_%a.out" \
        --job-name mod_ep_${i} \
        mod_ep_node.sh
done

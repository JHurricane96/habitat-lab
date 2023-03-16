#! /usr/bin/env bash

if [[ -z $1 ]]; then
    exp_id=$(cat logs/current_run.txt)
else
    exp_id=$1
fi

set -x

tensorboard --logdir logs/${exp_id}*

#! /usr/bin/env bash

# watch -n 15 "grep '/1000' ep_gen_logs/train/3.err | tail -n 5"
grep '/400' ep_gen_logs/val/$1.err | tail -n 5
# watch -n 15 "grep '/1000' ep_gen.err | tail -n 5"

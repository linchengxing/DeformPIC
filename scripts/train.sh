#!/bin/bash

CUDA_VISIBLE_DEVICES=$1 python main.py --config cfgs/DeformPIC.yaml \
    --exp_name DeformPIC \
    --val_freq 1 --swanlab

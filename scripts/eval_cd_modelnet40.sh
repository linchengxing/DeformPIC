#!/bin/bash

CUDA_VISIBLE_DEVICES=$1 python eval_cd.py --config cfgs/DeformPIC.yaml \
    --exp_name <exp_name> \
    --ckpts <ckpt_path> \
    --data_path <data_path> \
    --dataset_name ModelNet40
#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1 \
python ../src/train.py -c $@

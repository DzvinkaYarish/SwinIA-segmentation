#!/bin/bash

export root=/gpfs/space/projects/transformers_uss/

conda activate pt
python evaluate_seg.py --exp_dir ${root}seg_experiments/fmd_mice_livecell_test_random_500/ --config_path /gpfs/space/home/dzvenymy/SwinIA-segmentation/config_seg/config_full_livecell.yaml --checkpoint_path ${root}tp_mice.pth
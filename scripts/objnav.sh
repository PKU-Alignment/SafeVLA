#!/bin/bash
export PYTHONPATH=$PYTHONPATH:/root/poliformer
export OBJAVERSE_HOUSES_BASE_DIR=/root/data/houses/objaverse_houses,
export OBJAVERSE_HOUSES_DIR=/root/data/houses/objaverse_houses/houses_2023_07_28
export OBJAVERSE_DATA_BASE_DIR=/root/data/assets/objaverse_houses
export OBJAVERSE_DATA_DIR=/root/data/assets/objaverse_assets/2023_07_28
export OBJAVERSE_ANNOTATIONS_PATH=/root/data/assets/objaverse_assets/2023_07_28/annotations.json.gz
export WANDB_DIR=/root/data/wandb


export ALLENACT_DEBUG=True
export ALLENACT_DEBUG_VST_TIMEOUT=2000

export HF_ENDPOINT=https://hf-mirror.com


python training/online/online_eval.py --shuffle \
    --eval_subset minival \
    --output_basedir ./eval/objnav \
    --test_augmentation \
    --task_type ObjectNavType \
    --input_sensors raw_navigation_camera raw_manipulation_camera last_actions an_object_is_in_hand \
    --house_set objaverse \
    --num_workers 8 \
    --ckpt_path /root/poliformer/rl_ckpt/objectnav/model.ckpt
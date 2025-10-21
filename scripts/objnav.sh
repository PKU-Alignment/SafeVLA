export PYTHONPATH=/path/to/safevla
export OBJAVERSE_HOUSES_DIR=/path/to/objaverse_houses
export OBJAVERSE_DATA_DIR=/path/to/objaverse_assets

export HF_ENDPOINT=https://hf-mirror.com

export ALLENACT_DEBUG=True
export ALLENACT_DEBUG_VST_TIMEOUT=2000

python training/online/online_eval.py --shuffle \
    --eval_subset minival \
    --output_basedir ./eval/objnav \
    --test_augmentation \
    --task_type ObjectNavType \
    --input_sensors raw_navigation_camera raw_manipulation_camera last_actions an_object_is_in_hand \
    --house_set objaverse \
    --num_workers 1 \
    --ckpt_path /path/to/ckpt/model.ckpt

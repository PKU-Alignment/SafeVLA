export PYTHONPATH=$PYTHONPATH:/root/poliformer
export OBJAVERSE_HOUSES_BASE_DIR=/root/data/houses/objaverse_houses,
export OBJAVERSE_HOUSES_DIR=/root/data/houses/objaverse_houses/houses_2023_07_28
export OBJAVERSE_DATA_BASE_DIR=/root/data/assets/objaverse_houses
export OBJAVERSE_DATA_DIR=/root/data/assets/objaverse_assets/2023_07_28
export OBJAVERSE_ANNOTATIONS_PATH=/root/data/assets/objaverse_assets/2023_07_28/annotations.json.gz
export CKPT_DIR="/root/poliformer/il_ckpt/spoc_IL"



# "CKPT_DIR": "/root/poliformer/rl_ckpt/pickup",
# "WANDB_DIR": "./wandb",
# "PYTHONPATH": "",

echo "Download objaverse assets and annotation"
if [ ! -f $OBJAVERSE_DATA_BASE_DIR/2023_07_28/annotations.json.gz ] ; then
  python -m objathor.dataset.download_annotations --version 2023_07_28 --path $OBJAVERSE_DATA_BASE_DIR
else
  echo "Annotations already downloaded"
fi

if [ ! -d $OBJAVERSE_DATA_BASE_DIR/2023_07_28/assets ] ; then
  python -m objathor.dataset.download_assets --version 2023_07_28 --path $OBJAVERSE_DATA_BASE_DIR
else
  echo "Assets already downloaded"
fi

echo "Download objaverse houses"
if [ ! -f $OBJAVERSE_HOUSES_BASE_DIR/houses_2023_07_28/val.jsonl.gz ] ; then
  python scripts/download_objaverse_houses.py --save_dir $OBJAVERSE_HOUSES_BASE_DIR --subset val
else
  echo "Houses already downloaded"
fi

echo "Download ckpt from SigLIP-ViTb-3-double-det-CHORES-S"
if [ ! -d $CKPT_DIR/SigLIP-ViTb-3-double-det-CHORES-S ] ; then
  python -m scripts.download_trained_ckpt --save_dir $CKPT_DIR --ckpt_ids SigLIP-ViTb-3-double-det-CHORES-S
else
  echo "Checkpoint already downloaded"
fi

echo "Start evaluatioon"
python -m training.offline.online_eval --shuffle --eval_subset minival --output_basedir tmp_log \
 --test_augmentation --task_type ObjectNavType \
 --input_sensors raw_navigation_camera raw_manipulation_camera last_actions an_object_is_in_hand \
 nav_task_relevant_object_bbox manip_task_relevant_object_bbox nav_accurate_object_bbox manip_accurate_object_bbox \
 --house_set objaverse --wandb_logging False --num_workers 10 \
 --gpu_devices 0 1 --training_run_id spoc_IL --local_checkpoint_dir $CKPT_DIR
export PYTHONPATH=/path/to/SafeVLA
export OBJAVERSE_HOUSES_DIR=/path/to/objaverse_houses
export OBJAVERSE_DATA_DIR=/path/to/objaverse_assets
export HF_ENDPOINT=https://hf-mirror.com
export ALLENACT_DEBUG=True
export ALLENACT_DEBUG_VST_TIMEOUT=2000

# task_type: ObjectNavType | PickupType | FetchType
python training/online/dinov2_vits_tsfm_base.py train \
    --il_ckpt_path /path/to/il_ckpt/spoc_IL/model.ckpt \
    --num_train_processes 8 \
    --output_dir /path/to/ckpt_dir \
    --dataset_dir /path/to/dataset/astar/ObjectNavType \
    --cost_limit 2.31 \
    --tag ObjectNav\

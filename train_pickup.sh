export PYTHONPATH=/root/pr/SafeVLA
export OBJAVERSE_HOUSES_DIR=/root/data/houses/objaverse_houses/houses_2023_07_28
export OBJAVERSE_DATA_DIR=/root/data/assets/objaverse_assets/2023_07_28

export HF_ENDPOINT=https://hf-mirror.com
# export CUDA_VISIBLE_DEVICES=0

export ALLENACT_DEBUG=True
export ALLENACT_DEBUG_VST_TIMEOUT=2000

python training/online/dinov2_vits_tsfm_rgb_augment_objectnav.py train \
    --il_ckpt_path /root/data/il_ckpt/spoc_IL/model.ckpt \
    --num_train_processes 24 \
    --output_dir results \
    --dataset_dir /root/data/data/astar/PickupType \
    --cost_limit 0.2 \
    --tag Extended_Lagrange\

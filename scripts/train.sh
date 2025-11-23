export PYTHONPATH=/path/to/SafeVLA
export OBJAVERSE_HOUSES_DIR=/path/to/objaverse_houses
export OBJAVERSE_DATA_DIR=/path/to/objaverse_assets
export HF_ENDPOINT=https://hf-mirror.com
export ALLENACT_DEBUG=True
export ALLENACT_DEBUG_VST_TIMEOUT=2000

if [ $# -lt 2 ]; then
    echo "Usage: $0 <task_type> <ckpt_path>"
    echo "  task_type: 0 (ObjectNavType) | 1 (PickupType) | 2 (FetchType)"
    echo "  ckpt_path: Path to IL checkpoint file"
    exit 1
fi

task_type=$1
il_ckpt_path=$2
if [ "$task_type" == "0" ]; then
    task_type="ObjectNavType"
elif [ "$task_type" == "1" ]; then
    task_type="PickupType"
elif [ "$task_type" == "2" ]; then
    task_type="FetchType"
else
    echo "Invalid task type"
    exit 1
fi

python training/online/dinov2_vits_tsfm_base.py train \
    --il_ckpt_path $il_ckpt_path \
    --num_train_processes 8 \
    --output_dir /path/to/ckpt_dir \
    --dataset_dir /path/to/dataset/astar/$task_type \
    --cost_limit 2.31 \
    --tag $task_type\

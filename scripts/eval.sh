export PYTHONPATH=/path/to/SafeVLA
export OBJAVERSE_HOUSES_DIR=/path/to/objaverse_houses
export OBJAVERSE_DATA_DIR=/path/to/objaverse_assets
export HF_ENDPOINT=https://hf-mirror.com
export ALLENACT_DEBUG=True
export ALLENACT_DEBUG_VST_TIMEOUT=2000

if [ $# -lt 2 ]; then
    echo "Usage: $0 <task_type> <ckpt_path>"
    echo "  task_type: objectnav | pickup | fetch"
    echo "  ckpt_path: Path to checkpoint file"
    exit 1
fi

task_type=$1
ckpt_path=$2
if [ "$task_type" == "objectnav" ]; then
    task_type="ObjectNavType"
elif [ "$task_type" == "pickup" ]; then
    task_type="PickupType"
elif [ "$task_type" == "fetch" ]; then
    task_type="FetchType"
else
    echo "Invalid task type"
    exit 1
fi

python training/online/online_eval.py --shuffle \
    --eval_subset minival \
    --output_basedir ./eval/$task_type \
    --test_augmentation \
    --task_type $task_type \
    --input_sensors raw_navigation_camera raw_manipulation_camera last_actions an_object_is_in_hand \
    --house_set objaverse \
    --num_workers 8 \
    --seed 123 \
    --ckpt_path $ckpt_path
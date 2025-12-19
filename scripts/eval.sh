#!/bin/bash

export PYTHONPATH=/path/to/SafeVLA  # change to your own path
export OBJAVERSE_HOUSES_DIR=/path/to/objaverse_houses  # change to your own path
export OBJAVERSE_DATA_DIR=/path/to/objaverse_assets  # change to your own path
export HF_ENDPOINT=https://hf-mirror.com
export ALLENACT_DEBUG=True
export ALLENACT_DEBUG_VST_TIMEOUT=2000

# Default values
task_type=""
ckpt_path=""
eval_subset="minival"
output_basedir="./eval"
num_workers=8
seed=123
shuffle=true
test_augmentation=true
house_set="objaverse"
input_sensors="raw_navigation_camera raw_manipulation_camera last_actions an_object_is_in_hand"

# Function to print usage
print_usage() {
    echo "Usage: $0 --task_type <type> --ckpt_path <path> [OPTIONS]"
    echo ""
    echo "Required arguments:"
    echo "  --task_type           Task type: objectnav | pickup | fetch"
    echo "  --ckpt_path           Path to checkpoint file"
    echo ""
    echo "Optional arguments:"
    echo "  --eval_subset         Evaluation subset (default: minival)"
    echo "  --output_basedir      Output base directory (default: ./eval)"
    echo "  --num_workers         Number of workers (default: 8)"
    echo "  --seed                Random seed (default: 123)"
    echo "  --no_shuffle          Disable shuffling (default: enabled)"
    echo "  --no_test_augmentation Disable test augmentation (default: enabled)"
    echo "  --house_set           House set to use (default: objaverse)"
    echo "  --input_sensors       Input sensors (default: raw_navigation_camera raw_manipulation_camera last_actions an_object_is_in_hand)"
    echo "  --help                Show this help message"
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --task_type)
            task_type="$2"
            shift 2
            ;;
        --ckpt_path)
            ckpt_path="$2"
            shift 2
            ;;
        --eval_subset)
            eval_subset="$2"
            shift 2
            ;;
        --output_basedir)
            output_basedir="$2"
            shift 2
            ;;
        --num_workers)
            num_workers="$2"
            shift 2
            ;;
        --seed)
            seed="$2"
            shift 2
            ;;
        --no_shuffle)
            shuffle=false
            shift
            ;;
        --no_test_augmentation)
            test_augmentation=false
            shift
            ;;
        --house_set)
            house_set="$2"
            shift 2
            ;;
        --input_sensors)
            input_sensors="$2"
            shift 2
            ;;
        --help)
            print_usage
            ;;
        *)
            echo "Unknown option: $1"
            print_usage
            ;;
    esac
done

# Check required arguments
if [ -z "$task_type" ] || [ -z "$ckpt_path" ]; then
    echo "Error: Missing required arguments"
    echo ""
    print_usage
fi

# Convert task type to internal format
if [ "$task_type" == "objectnav" ]; then
    task_type_internal="ObjectNavType"
elif [ "$task_type" == "pickup" ]; then
    task_type_internal="PickupType"
elif [ "$task_type" == "fetch" ]; then
    task_type_internal="FetchType"
else
    echo "Error: Invalid task type '$task_type'"
    echo "Valid options: objectnav, pickup, fetch"
    exit 1
fi

# Build the command
cmd="python training/online/online_eval.py"

# Add shuffle flag if enabled
if [ "$shuffle" = true ]; then
    cmd="$cmd --shuffle"
fi

# Add test_augmentation flag if enabled
if [ "$test_augmentation" = true ]; then
    cmd="$cmd --test_augmentation"
fi

# Add other parameters
cmd="$cmd \
    --eval_subset $eval_subset \
    --output_basedir $output_basedir/$task_type_internal \
    --task_type $task_type_internal \
    --input_sensors $input_sensors \
    --house_set $house_set \
    --num_workers $num_workers \
    --seed $seed \
    --ckpt_path $ckpt_path"

# Execute the command
echo "Executing command:"
echo "$cmd"
echo ""
eval $cmd

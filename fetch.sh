export PYTHONPATH=/root/pr/SafeVLA
export OBJAVERSE_HOUSES_DIR=/root/data/houses/objaverse_houses/houses_2023_07_28
export OBJAVERSE_DATA_DIR=/root/data/assets/objaverse_assets/2023_07_28

export HF_ENDPOINT=https://hf-mirror.com
# export CUDA_VISIBLE_DEVICES=1

export ALLENACT_DEBUG=True
export ALLENACT_DEBUG_VST_TIMEOUT=2000

python training/online/online_eval.py --shuffle \
    --eval_subset minival \
    --output_basedir /root/pr/safevla/eval/fetch \
    --test_augmentation \
    --task_type FetchType \
    --input_sensors raw_navigation_camera raw_manipulation_camera last_actions an_object_is_in_hand \
    --house_set objaverse \
    --num_workers 32 \
    --seed 4207 \
    --ckpt_path /root/data/results/checkpoints/Fetch/2025-11-18_23-50-35/exp_Fetch__stage_02__steps_000024764284.pt
    # --ckpt_path /root/data/results/checkpoints/Fetch/2025-11-18_16-29-04/exp_Fetch__stage_02__steps_000024347636.pt

    # --ckpt_path /root/poliformer/results/checkpoints/advance_loss_add_objcost/2025-05-04_09-44-01/exp_advance_loss_add_objcost__stage_02__steps_000002512040.pt
    # --ckpt_path /root/poliformer/results/checkpoints/advance_loss_add_objcost/2025-05-04_07-32-07/exp_advance_loss_add_objcost__stage_00__steps_000000202032.pt

    # --ckpt_path /root/poliformer/results/checkpoints/advance_loss/2025-05-03_04-05-57/exp_advance_loss__stage_02__steps_000008515884.pt
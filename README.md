# SafeVLA: Towards Safety Alignment of Vision-Language-Action Model via Constrained Learning


This repository contains the code for the paper "SafeVLA: Towards Safety Alignment of Vision-Language-Action Model via Constrained Learning"



## Setting up the Python environment


Please use the pre-built image from Docker Hub:

```bash
docker pull safevla/safevla:v0
```

then:

```bash
export CODE_PATH=/path/to/this/repo
export DATA_PATH=/path/to/training_data
export DOCKER_IMAGE=safevla/safevla:v0
docker run \
    --gpus all \
    --device /dev/dri \
    --mount type=bind,source=${CODE_PATH},target=/root/spoc \
    --mount type=bind,source=${DATA_PATH},target=/root/data \
    --shm-size 50G \
    --runtime=nvidia \
    -it ${DOCKER_IMAGE}:latest
```

and use the following conda environment:

```bash
conda activate spoc
```


## Training Data

### Downloading the training data

SafeVLA is trained using `astar` from [SPOC](https://spoc-robot.github.io/) CHORES benchamrk. The `astar` type has the agent navigating and fetching one of fifteen possible object types. To download the training data for the `astar` type, run the following command:  

```bash
python -m scripts.download_training_data --save_dir /your/local/save/dir --types astar
```

for example

```bash
python -m scripts.download_training_data --save_dir data --types astar
```

#### 📁 Dataset format 📁

Once you run the above command, you will have a directory structure that looks like this

```
/your/local/save/dir/<astar OR all>_type
    <TASK_TYPE>
        house_id_to_sub_house_id_train.json # This file contains a mapping that's needed for train data loading
        house_id_to_sub_house_id_val.json   # This file contains a mapping that's needed for val data loading
        train
            <HOUSEID>
                hdf5_sensors.hdf5 -- containing all the sensors that are not videos
                    <EPISODE_NUMBER>
                        <SENSOR_NAME>
                raw_navigation_camera__<EPISODE_NUMBER>.mp4
                raw_manipulation_camera__<EPISODE_NUMBER>.mp4
        val
            # As with train
```


The `hdf5_sensors.hdf5` contains the necessary information for training, including the house id, starting pose, and target object type/id.

For more information about the downloaded data, including trajectory videos and recorded sensors, please refer to [SPOC](https://spoc-robot.github.io/) documentation.



## Training and Evaluation

In order to run training and evaluation you'll need:

1. The processed/optimized Objaverse assets along with their annotations.
2. The set of ProcTHOR-Objaverse houses you'd like to train/evaluate on.
3. For evaluation only, a trained model checkpoint.

Below we describe how to download the assets, annotations, and the ProcTHOR-Objaverse houses. We also describe how you can use one of our pre-trained models to run evaluation.

### 💾 Downloading assets, annotations, and houses 💾

#### 📦 Downloading optimized Objaverse assets and annotations 📦

Pick a directory `/path/to/objaverse_assets` where you'd like to save the assets and annotations. Then run the following commands:

```bash
python -m objathor.dataset.download_annotations --version 2023_07_28 --path /path/to/objaverse_assets
python -m objathor.dataset.download_assets --version 2023_07_28 --path /path/to/objaverse_assets
```

These will create the directory structure:

```
/path/to/objaverse_assets
    2023_07_28
        annotations.json.gz                              # The annotations for each object
        assets
            000074a334c541878360457c672b6c2e             # asset id
                000074a334c541878360457c672b6c2e.pkl.gz
                albedo.jpg
                emission.jpg
                normal.jpg
                thor_metadata.json
            ... #  39663 more asset directories
```

#### 🏠 Downloading ProcTHOR-Objaverse houses 🏠

Pick a directory `/path/to/objaverse_houses` where you'd like to save ProcTHOR-Objaverse houses. Then run: 

```bash
python -m scripts.download_objaverse_houses --save_dir /path/to/objaverse_houses --subset val
```

to download the validation set of houses as `/path/to/objaverse_houses/val.jsonl.gz`.
You can also change `val` to `train` to download the training set of houses.

#### 🛣 Setting environment variables 🛣

Next you need to set the following environment variables:

```bash
export PYTHONPATH=/path/to/code_in_docker
export OBJAVERSE_HOUSES_DIR=/path/to/objaverse_houses
export OBJAVERSE_DATA_DIR=/path/to/objaverse_assets
```

For training, we recommend to set two more environment variables to avoid timeout issues from [AllenAct](https://allenact.org/):

```bash
export ALLENACT_DEBUG=True
export ALLENACT_DEBUG_VST_TIMEOUT=2000
```

### Running Safe RL finetuning

Download pretrained IL ckpt:

```bash
python scripts/download_trained_ckpt.py --ckpt_ids spoc_IL --save_dir PATH_TO_SAVE_DIR
python training/online/dinov2_vits_tsfm_rgb_augment_objectnav.py train --il_ckpt_path IL_CKPT_PATH --num_train_processes NUM_OF_TRAIN_PROCESSES --output_dir PATH_TO_RESULT --dataset_dir PATH_TO_DATASET

python training/online/dinov2_vits_tsfm_rgb_augment_objectnav.py train --il_ckpt_path IL_CKPT_PATH --num_train_processes NUM_OF_TRAIN_PROCESSES --output_dir PATH_TO_RESULT --dataset_dir PATH_TO_DATASET --cost_limit COST_LIMIT --tag EXP_NAME
```

for example

```bash
python training/online/dinov2_vits_tsfm_rgb_augment_objectnav.py train --il_ckpt_path /root/data/il_ckpt/spoc_IL/model.ckpt --num_train_processes 32 --output_dir results --dataset_dir /root/data/data/astar/ObjectNavType --cost_limit 2.31964 --tag SafeVLA2.31964-ObjectNavType-RL-DinoV2-ViTS-TSFM
```


### Running evaluation with a trained model


#### Downloading the aligned model ckpt and evaluation results

```bash
python scripts/download_aligned_ckpt.py --save_dir ckpt
cd ckpt
cat safevla_* | tar -xz
```
Vidoes files and are evaluation results.After downloading the model
```bash
bash objnav_eval.bash
```

export PYTHONPATH=/path/to/code_in_docker
export OBJAVERSE_HOUSES_DIR=/path/to/objaverse_houses
export OBJAVERSE_DATA_DIR=/path/to/objaverse_assets

```bash
export PYTHONPATH=/path/to/code_in_docker
export OBJAVERSE_HOUSES_DIR=/path/to/objaverse_houses
export OBJAVERSE_DATA_DIR=/path/to/objaverse_assets
export WANDB_DIR=/path/to/wandb

python training/online/online_eval.py --shuffle \
    --eval_subset minival \
    --output_basedir ./eval/objectnav \
    --test_augmentation \
    --task_type ObjectNavType \
    --input_sensors raw_navigation_camera raw_manipulation_camera last_actions an_object_is_in_hand \
    --house_set objaverse \
    --num_workers 8 \
    --gpu_devices 0 1 2 3 4 5 6 7 \
    --ckpt_path ./ckpt/safevla/safe_model.pt \
```


for example
```bash
export PYTHONPATH=$PYTHONPATH:/root/poliformer
export OBJAVERSE_HOUSES_DIR=/root/data/houses/objaverse_houses/houses_2023_07_28
export OBJAVERSE_DATA_DIR=/root/data/assets/objaverse_assets/2023_07_28
export WANDB_DIR=/root/data/wandb

python training/online/online_eval.py --shuffle \
    --eval_subset minival \
    --output_basedir ./eval/objectnav \
    --test_augmentation \
    --task_type ObjectNavType \
    --input_sensors raw_navigation_camera raw_manipulation_camera last_actions an_object_is_in_hand \
    --house_set objaverse \
    --num_workers 8 \
    --gpu_devices 0 1 2 3 4 5 6 7 \
    --ckpt_path ./ckpt/safevla/safe_model.pt \
```
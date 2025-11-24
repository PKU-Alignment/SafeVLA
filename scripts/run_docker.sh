export CODE_PATH=/path/to/this/repo
export DATA_PATH=/path/to/data_dir
export DOCKER_IMAGE=safevla/safevla:v1
docker run \
    --gpus all \
    --device /dev/dri \
    --mount type=bind,source=${CODE_PATH},target=/root/SafeVLA \
    --mount type=bind,source=${DATA_PATH},target=/root/data \
    --shm-size 50G \
    --runtime=nvidia \
    --network=host \
    --name safevla \
    -it ${DOCKER_IMAGE}
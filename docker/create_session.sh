# check if DOCKER_IMAGE exists and if not ask the user to export it
if [ -z ${DOCKER_IMAGE} ]; then
    echo "Please export DOCKER_IMAGE envionrment variable first as follows:"
    echo "    export DOCKER_IMAGE=<name-of-your-docker-image>"
    echo "To get the names of available docker images run:"
    echo "    docker images"
    exit 1
fi

# if SPOC_PATH isn't provided, use the current directory
if [ -z ${SPOC_PATH} ]; then
    SPOC_PATH=$(pwd)
    echo "Mounting the current directory ${SPOC_PATH} at /root/spoc"
    echo "If you wish to mount a different directory, export the environment variable SPOC_PATH as follows:"
    echo "    export SPOC_PATH=/path/to/spoc/directory"
else
    echo "Mounting ${SPOC_PATH} at /root/spoc"
fi

# if DATA_PATH isn't provided, use the ./data
if [ -z ${DATA_PATH} ]; then
    DATA_PATH=$(pwd)/data
    echo "Mounting the current directory ${DATA_PATH} at /root/data"
    echo "If you wish to mount a different directory, export the environment variable DATA_PATH as follows:"
    echo "    export DATA_PATH=/path/to/data/directory"
else
    echo "Mounting ${DATA_PATH} at /root/data"
fi

# launch an interactive session with the docker image
docker run \
    --gpus all \
    --device /dev/dri \
    --mount type=bind,source=${SPOC_PATH},target=/root/spoc \
    --mount type=bind,source=${DATA_PATH},target=/root/data \
    --shm-size 50G \
    -it ${DOCKER_IMAGE}:latest
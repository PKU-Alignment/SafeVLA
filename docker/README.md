# Using FLaRe with Docker

## Building the Docker image

These instructions assume you are building and running this Docker image on a Linux machine with an nvidia graphics
card. You'll need to have the `docker` installed along with the `nvidia-container-toolkit`, you can install the 
`nvidia-container-toolkit` by running:
```bash
# Installing nvidia-container-toolkit
curl -s -L https://nvidia.github.io/nvidia-container-runtime/gpgkey | \
  sudo apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-container-runtime/$distribution/nvidia-container-runtime.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-runtime.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit-base nvidia-container-toolkit

# Need to restart docker to get the `nvidia-container-toolkit` to be usable
sudo systemctl restart docker
```

Change your current working directory to this directory:
```bash
cd /path/to/spoc/docker
```

You can now build the docker image by running
```bash
create_image.sh
```
This will print out the name of the docker image, this will look something like
```
spoc-base-20241104
```

## Running the Docker image

Let's assume you've built the docker image using the above instructions and this has created the docker image
```bash
DOCKER_IMAGE=spoc-base-20241104
```
You can now start this Docker image mounting your local SPOC repository to `/root/spoc` in the Docker image by running:
```bash
SPOC_PATH=/path/to/spoc
DATA_PATH=/path/to/data
docker run \
    --gpus all \
    --device /dev/dri \
    --mount type=bind,source=${SPOC_PATH},target=/root/spoc \
    --mount type=bind,source=${DATA_PATH},target=/root/data \
    --shm-size 250G \
    -it ${DOCKER_IMAGE}:latest
```
You should set `--shm-size 250G` to be a reasonable size for your system, you will need a large shared memory when
running training as the dataloader will load a lot of data into shared memory.

Alternatively, you could use our helper script to start the docker interactive session by running the following command from the topmost directory of the SPOC repository:
```bash
bash create_image.sh
```
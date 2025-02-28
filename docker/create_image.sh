BASE_DOCKER_IMAGE="spoc-base-$(date +"%Y%m%d")" \
  && DOCKER_BUILDKIT=1 docker build -t \
   $BASE_DOCKER_IMAGE:latest \
   --file Dockerfile \
   .

echo "Docker image name: ${BASE_DOCKER_IMAGE}"
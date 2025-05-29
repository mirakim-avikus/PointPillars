echo $DISPLAY
xhost +local:root

docker run --name pointpillars \
  --gpus all \
  -it -d \
  --network=host \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v $(pwd):/workspace \
  custom-open3d-python-cu111 \
  bash

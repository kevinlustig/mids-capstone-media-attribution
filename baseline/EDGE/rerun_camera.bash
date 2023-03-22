docker build -t cam -f Dockerfile .
xhost +
docker run --privileged --rm -v /data:/data -e DISPLAY -v /tmp:/tmp -ti cam python3 camera.py --device /dev/video0 
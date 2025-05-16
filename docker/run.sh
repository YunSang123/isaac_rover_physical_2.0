docker run --name nav2 -it \
        --net host \
        --runtime nvidia \
        --privileged \
        -v /dev:/dev \
        -v /tmp/.X11-unix:/tmp/.X11-unix \
        -v $HOME/.Xauthority:/root/.Xauthority:rw \
        -v /etc/nv_tegra_release:/etc/nv_tegra_release \
        -v /home/anger/ROS2/isaac_rover_physical_2.0:/isaac_rover_physical_2.0 \
        -e DISPLAY=$DISPLAY \
        -e QT_X11_NO_MITSHM=1 \
        nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 /bin/bash
        -v 
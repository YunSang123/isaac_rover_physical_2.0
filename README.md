```bash
# Clone repo
git clone https://github.com/YunSang123/isaac_rover_physical_2.0.git
cd isaac_rover_physical_2.0

# Build and source
colcon build
source install/setup.bash

# Launch the package
ros2 launch controller controller.launch.py
```
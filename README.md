1. git clone this repository
2. Change system path in joystick.py and loadpolicy.py
3. Change model path in line 14 in loadpolicy.py

// Step 1
git clone https://github.com/YunSang123/isaac_rover_physical_2.0.git
// Execute step 2 and 3
cd isaac_rover_physical_2.0
colcon build
source install/setup.bash
ros2 launch controller controller.launch.py
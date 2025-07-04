from setuptools import setup
import os
from glob import glob

package_name = 'controller'
sub_module = 'controller_utils'
setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name, sub_module],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('launch/*.launch.py'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='xavier',
    maintainer_email='jknuds19@student.aau.dk',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            #'motor_node = controller.motor_node:main',
            #'Kinematics_node = controller.Kinematics_node:main',
            'joy_listener = controller.joystick:main',
            'motor_subscriber = controller.motor:main'
        ],
    },
)
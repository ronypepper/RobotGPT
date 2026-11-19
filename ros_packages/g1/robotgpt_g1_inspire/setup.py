from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'robotgpt_g1_inspire'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (
            os.path.join("share", package_name, "launch"),
            glob("launch/*.launch.xml"),
        ),
    ],
    scripts=[
        "scripts/robotgpt_isaac_teleop_controller.launcher",
        "scripts/robotgpt_inverse_kinematics.launcher",
        "scripts/robotgpt_openpi_agent.launcher",
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='asl_team',
    maintainer_email='asl_team@todo.todo',
    description='Provides functionality for teleoperating a Unitree G1 robot with Inspire 6DOF dexterous hands via CloudXR, collecting datasets and receiving joint commands from an openpi client server (a pi VLA policy)',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'inverse_kinematics = robotgpt_g1_inspire.robotgpt_inverse_kinematics:main',
            'isaac_teleop_controller = robotgpt_g1_inspire.robotgpt_isaac_teleop_controller:main',
            'openpi_agent = robotgpt_g1_inspire.robotgpt_openpi_agent:main',
        ],
    },
)

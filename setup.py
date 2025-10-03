from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'rdj2025'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='game',
    maintainer_email='olumoruth3@gmail.com',
    description='RDJ2025 potato disease detection package',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'detection_node = rdj2025.detection_node:main',
            'inference_agent = rdj2025.inference_agent:main',
            'service_node = rdj2025.service_node:main',
        ],
    },
)

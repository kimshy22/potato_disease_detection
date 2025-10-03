from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess

def generate_launch_description():
    return LaunchDescription([
        # Start the FastAPI inference server
        ExecuteProcess(
            cmd=['python3', '/home/game/camera_ws/src/rdj2025/rdj2025/inference_agent.py'],
            output='screen'
        ),


        # Start the potato detection node
        Node(
            package='rdj2025',
            executable='detection_node',
            name='potato_detection_node',
            output='screen'
        ),

        # Start the potato service node
        Node(
            package='rdj2025',
            executable='service_node',
            name='service_node',
            output='screen'
        ),
    ])

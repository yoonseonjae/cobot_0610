import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess, IncludeLaunchDescription, SetEnvironmentVariable, GroupAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node, PushRosNamespace
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    map_file = os.path.join(os.getcwd(), 'nav2_map.yaml')
    params_file1 = os.path.join(os.getcwd(), 'robot1_nav2_params.yaml')
    params_file2 = os.path.join(os.getcwd(), 'robot2_nav2_params.yaml')
    rviz_config = os.path.join(os.getcwd(), 'multi_robot_nav2.rviz')
    namespaced_tf_remappings = [('/tf', 'tf'), ('/tf_static', 'tf_static')]
    
    nav2_bringup_dir = get_package_share_directory('nav2_bringup')
    
    return LaunchDescription([
        SetEnvironmentVariable('RMW_IMPLEMENTATION', 'rmw_cyclonedds_cpp'),
        
        # 1. Map Server (단일)
        Node(
            package='nav2_map_server',
            executable='map_server',
            name='map_server',
            output='screen',
            parameters=[{'yaml_filename': map_file, 'use_sim_time': True}]
        ),
        Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_map',
            output='screen',
            parameters=[{'use_sim_time': True},
                        {'autostart': True},
                        {'node_names': ['map_server']}]
        ),

        # RViz/tf2_echo listen to the global /tf and /tf_static topics.
        # Nav2's namespaced launch remaps TF to /robotX/tf, so publish these
        # global copies as well to make both robots visible in one RViz tree.
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='global_static_tf_map_to_robot1_odom',
            arguments=['0', '0', '0', '0', '0', '0', 'map', 'robot1/odom']
        ),
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='global_static_tf_robot1_body_to_lidar',
            arguments=['0', '0', '0.25', '0', '0', '0', 'robot1/body', 'robot1/Functional_Lidar']
        ),
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='global_static_tf_map_to_robot2_odom',
            arguments=['0', '0', '0', '0', '0', '0', 'map', 'robot2/odom']
        ),
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='global_static_tf_robot2_body_to_lidar',
            arguments=['0', '0', '0.25', '0', '0', '0', 'robot2/body', 'robot2/Functional_Lidar']
        ),
        ExecuteProcess(
            cmd=['python3', os.path.join(os.getcwd(), 'pose_file_to_ros_bridge.py')],
            name='pose_file_to_ros_bridge',
            output='screen'
        ),
        ExecuteProcess(
            cmd=['python3', os.path.join(os.getcwd(), 'robot2_room_patrol.py')],
            name='robot2_room_patrol',
            output='screen'
        ),
        # ==================== ROBOT 1 ====================
        GroupAction([
            PushRosNamespace('robot1'),
            
            # Static TF for Robot 1
            Node(
                package='tf2_ros',
                executable='static_transform_publisher',
                name='static_tf_map_to_odom_r1',
                arguments=['0', '0', '0', '0', '0', '0', 'map', 'robot1/odom'],
                remappings=namespaced_tf_remappings
            ),
            Node(
                package='tf2_ros',
                executable='static_transform_publisher',
                name='static_tf_body_to_lidar_r1',
                arguments=['0', '0', '0.25', '0', '0', '0', 'robot1/body', 'robot1/Functional_Lidar'],
                remappings=namespaced_tf_remappings
            ),
            
            # Nav2 for Robot 1
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(os.path.join(nav2_bringup_dir, 'launch', 'navigation_launch.py')),
                launch_arguments={
                    'use_sim_time': 'True',
                    'params_file': params_file1,
                    'autostart': 'True',
                    'use_composition': 'False',
                    'use_namespace': 'True',
                    'namespace': 'robot1',
                }.items(),
            ),
        ]),

        # ==================== ROBOT 2 ====================
        GroupAction([
            PushRosNamespace('robot2'),
            
            # Static TF for Robot 2
            Node(
                package='tf2_ros',
                executable='static_transform_publisher',
                name='static_tf_map_to_odom_r2',
                arguments=['0', '0', '0', '0', '0', '0', 'map', 'robot2/odom'],
                remappings=namespaced_tf_remappings
            ),
            Node(
                package='tf2_ros',
                executable='static_transform_publisher',
                name='static_tf_body_to_lidar_r2',
                arguments=['0', '0', '0.25', '0', '0', '0', 'robot2/body', 'robot2/Functional_Lidar'],
                remappings=namespaced_tf_remappings
            ),
            
            # Nav2 for Robot 2
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(os.path.join(nav2_bringup_dir, 'launch', 'navigation_launch.py')),
                launch_arguments={
                    'use_sim_time': 'True',
                    'params_file': params_file2,
                    'autostart': 'True',
                    'use_composition': 'False',
                    'use_namespace': 'True',
                    'namespace': 'robot2',
                }.items(),
            ),
        ]),

        # 3. RViz2 실행
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', rviz_config],
            parameters=[{'use_sim_time': True}]
        )
    ])

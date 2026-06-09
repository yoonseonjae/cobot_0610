#!/usr/bin/env python3
import math

import rclpy
from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose, Spin
from rclpy.action import ActionClient
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Empty


class Robot2RoomPatrol(Node):
    def __init__(self):
        super().__init__("robot2_room_patrol")
        self._namespace = "robot2"
        self._running = False
        self._waypoint_index = 0
        self._spin_after_room = None
        self._waypoints = [
            ("1번방", -1.707, -11.104, 0.8),
            ("5번방", 5.039, -6.99, 0.8),
            ("3번방", -1.707, 2.059, 0.8),
            ("4번방", -1.707, 12.753, 0.8),
            ("6번방", 5.039, 11.986, 0.8),
        ]

        self._navigate_client = ActionClient(
            self, NavigateToPose, f"/{self._namespace}/navigate_to_pose"
        )
        self._spin_client = ActionClient(self, Spin, f"/{self._namespace}/spin")
        self.create_subscription(
            Empty, f"/{self._namespace}/start_room_patrol", self._on_start, 10
        )
        self.get_logger().info(
            "Waiting for /robot2/start_room_patrol. "
            "Route: 1 -> 5 -> 3 -> 4 -> 6. "
            "Robot spins once only after each navigation goal is reached."
        )

    def _on_start(self, _msg):
        if self._running:
            self.get_logger().warn("Room patrol is already running; ignoring trigger.")
            return

        self._running = True
        self._waypoint_index = 0
        self.get_logger().info("Room patrol trigger received.")
        self._send_next_navigation_goal()

    def _send_next_navigation_goal(self):
        if self._waypoint_index >= len(self._waypoints):
            self._running = False
            self.get_logger().info("Room patrol completed.")
            return

        room_name, x, y, z = self._waypoints[self._waypoint_index]
        route_number = self._waypoint_index + 1
        if not self._navigate_client.wait_for_server(timeout_sec=5.0):
            self._running = False
            self.get_logger().error("/robot2/navigate_to_pose action server is not available.")
            return

        goal = NavigateToPose.Goal()
        goal.pose = self._make_pose(x, y, z)
        self.get_logger().info(
            f"{route_number}번째 지점({room_name}) 이동 시작: "
            f"x={x:.3f}, y={y:.3f}, z={z:.3f}"
        )
        future = self._navigate_client.send_goal_async(goal)
        future.add_done_callback(self._on_navigation_goal_response)

    def _on_navigation_goal_response(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self._running = False
            self.get_logger().error("Navigation goal was rejected.")
            return

        room_name = self._waypoints[self._waypoint_index][0]
        route_number = self._waypoint_index + 1
        self.get_logger().info(f"{route_number}번째 지점({room_name}) goal accepted.")
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._on_navigation_result)

    def _on_navigation_result(self, future):
        result = future.result()
        room_name = self._waypoints[self._waypoint_index][0]
        route_number = self._waypoint_index + 1
        if result.status != GoalStatus.STATUS_SUCCEEDED:
            self._running = False
            self.get_logger().error(
                f"{route_number}번째 지점({room_name}) 이동 실패. status={result.status}"
            )
            return

        self.get_logger().info(f"{route_number}번째 지점({room_name}) 도착 완료.")
        self.get_logger().info(f"{route_number}번째 지점({room_name}) 제자리 한 바퀴 회전 시작.")
        self._spin_after_room = room_name
        self._send_spin_goal()

    def _send_spin_goal(self):
        if not self._spin_client.wait_for_server(timeout_sec=5.0):
            self._running = False
            self.get_logger().error("/robot2/spin action server is not available.")
            return

        goal = Spin.Goal()
        goal.target_yaw = float(2.0 * math.pi)
        goal.time_allowance = Duration(sec=40, nanosec=0)
        future = self._spin_client.send_goal_async(goal)
        future.add_done_callback(self._on_spin_goal_response)

    def _on_spin_goal_response(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self._running = False
            self.get_logger().error("Spin goal was rejected.")
            return

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._on_spin_result)

    def _on_spin_result(self, future):
        result = future.result()
        room_name = self._spin_after_room
        route_number = self._waypoint_index + 1
        if result.status != GoalStatus.STATUS_SUCCEEDED:
            self._running = False
            self.get_logger().error(
                f"{route_number}번째 지점({room_name}) 회전 실패. status={result.status}"
            )
            return

        self.get_logger().info(f"{route_number}번째 지점({room_name}) 제자리 한 바퀴 회전 완료.")
        self._waypoint_index += 1
        self._send_next_navigation_goal()

    def _make_pose(self, x, y, z, yaw=0.0):
        pose = PoseStamped()
        pose.header.frame_id = "map"
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        pose.pose.position.z = float(z)
        pose.pose.orientation.z = math.sin(yaw / 2.0)
        pose.pose.orientation.w = math.cos(yaw / 2.0)
        return pose


def main(args=None):
    rclpy.init(args=args)
    node = Robot2RoomPatrol()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

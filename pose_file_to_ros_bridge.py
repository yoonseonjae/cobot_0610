#!/usr/bin/env python3
import json
import os

import rclpy
from builtin_interfaces.msg import Time
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rosgraph_msgs.msg import Clock
from tf2_msgs.msg import TFMessage
from visualization_msgs.msg import Marker, MarkerArray


class PoseFileToROSBridge(Node):
    def __init__(self):
        super().__init__("pose_file_to_ros_bridge")
        self.namespaces = ("robot1", "robot2")
        self.latest_clock = None
        self.pose_files = {
            namespace: f"/tmp/isaac_pose_{namespace}.json"
            for namespace in self.namespaces
        }
        self.clock_sub = self.create_subscription(Clock, "/clock", self._on_clock, 10)
        self.odom_publishers = {
            namespace: self.create_publisher(Odometry, f"/{namespace}/odom", 10)
            for namespace in self.namespaces
        }
        self.namespaced_tf_publishers = {
            namespace: self.create_publisher(TFMessage, f"/{namespace}/tf", 10)
            for namespace in self.namespaces
        }
        self.global_tf_publisher = self.create_publisher(TFMessage, "/tf", 10)
        self.marker_publisher = self.create_publisher(MarkerArray, "/robot_pose_markers", 10)
        self.timer = self.create_timer(1.0 / 30.0, self._publish_all)
        self.get_logger().info(
            "Publishing /robot1/odom, /robot2/odom, /tf, /robot1/tf, /robot2/tf, and /robot_pose_markers from Isaac pose files"
        )

    def _on_clock(self, msg):
        self.latest_clock = msg.clock

    def _read_pose(self, namespace):
        path = self.pose_files[namespace]
        if not os.path.exists(path):
            return None

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            return None

        if "position" not in data or "orientation" not in data:
            return None
        if len(data["position"]) != 3 or len(data["orientation"]) != 4:
            return None
        return data

    def _publish_all(self):
        stamp = self.latest_clock
        if stamp is None:
            stamp = self.get_clock().now().to_msg()

        markers = MarkerArray()
        for marker_id, namespace in enumerate(self.namespaces):
            pose = self._read_pose(namespace)
            if pose is None:
                continue

            pos = pose["position"]
            quat_wxyz = pose["orientation"]

            odom = Odometry()
            odom.header.stamp = stamp
            odom.header.frame_id = f"{namespace}/odom"
            odom.child_frame_id = f"{namespace}/body"
            odom.pose.pose.position.x = pos[0]
            odom.pose.pose.position.y = pos[1]
            odom.pose.pose.position.z = pos[2]
            odom.pose.pose.orientation.w = quat_wxyz[0]
            odom.pose.pose.orientation.x = quat_wxyz[1]
            odom.pose.pose.orientation.y = quat_wxyz[2]
            odom.pose.pose.orientation.z = quat_wxyz[3]

            transform = TransformStamped()
            transform.header = odom.header
            transform.child_frame_id = odom.child_frame_id
            transform.transform.translation.x = odom.pose.pose.position.x
            transform.transform.translation.y = odom.pose.pose.position.y
            transform.transform.translation.z = odom.pose.pose.position.z
            transform.transform.rotation = odom.pose.pose.orientation

            tf_msg = TFMessage(transforms=[transform])
            self.odom_publishers[namespace].publish(odom)
            self.global_tf_publisher.publish(tf_msg)
            self.namespaced_tf_publishers[namespace].publish(tf_msg)
            markers.markers.append(self._make_marker(marker_id, namespace, odom))

        if markers.markers:
            self.marker_publisher.publish(markers)

    def _make_marker(self, marker_id, namespace, odom):
        marker = Marker()
        marker.header = odom.header
        marker.header.frame_id = "map"
        marker.ns = "robot_pose"
        marker.id = marker_id
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.pose = odom.pose.pose
        marker.scale.x = 0.9
        marker.scale.y = 0.18
        marker.scale.z = 0.18
        marker.color.a = 1.0
        if namespace == "robot1":
            marker.color.r = 0.0
            marker.color.g = 0.65
            marker.color.b = 1.0
        else:
            marker.color.r = 1.0
            marker.color.g = 0.45
            marker.color.b = 0.0
        return marker


def main(args=None):
    rclpy.init(args=args)
    node = PoseFileToROSBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

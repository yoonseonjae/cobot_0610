#!/usr/bin/env python3
import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from tf2_msgs.msg import TFMessage


class OdomToTFBridge(Node):
    def __init__(self):
        super().__init__("odom_to_tf_bridge")
        self.namespaces = ("robot1", "robot2")
        odom_qos = QoSProfile(depth=50, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.global_tf_pub = self.create_publisher(TFMessage, "/tf", 100)
        self.namespaced_tf_pubs = {
            namespace: self.create_publisher(TFMessage, f"/{namespace}/tf", 100)
            for namespace in self.namespaces
        }
        self.subscriptions = [
            self.create_subscription(
                Odometry,
                f"/{namespace}/odom",
                lambda msg, namespace=namespace: self._on_odom(namespace, msg),
                odom_qos,
            )
            for namespace in self.namespaces
        ]
        self.get_logger().info(
            "Publishing odom TF from /robot1/odom and /robot2/odom to global and namespaced TF"
        )

    def _on_odom(self, namespace, msg):
        transform = TransformStamped()
        transform.header = msg.header
        transform.header.frame_id = f"{namespace}/odom"
        transform.child_frame_id = f"{namespace}/body"
        transform.transform.translation.x = msg.pose.pose.position.x
        transform.transform.translation.y = msg.pose.pose.position.y
        transform.transform.translation.z = msg.pose.pose.position.z
        transform.transform.rotation = msg.pose.pose.orientation

        tf_msg = TFMessage(transforms=[transform])
        self.global_tf_pub.publish(tf_msg)
        self.namespaced_tf_pubs[namespace].publish(tf_msg)


def main(args=None):
    rclpy.init(args=args)
    node = OdomToTFBridge()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

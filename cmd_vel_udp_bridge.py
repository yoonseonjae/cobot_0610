import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import socket
import struct
import sys
import argparse

class CmdVelUDPBridge(Node):
    def __init__(self, namespace, udp_port):
        super().__init__(f'cmd_vel_udp_bridge_{namespace}')
        topic_name = f'/{namespace}/cmd_vel' if namespace else '/cmd_vel'
        self.subscription = self.create_subscription(
            Twist,
            topic_name,
            self.listener_callback,
            10)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp_ip = "127.0.0.1"
        self.udp_port = udp_port
        self.get_logger().info(f"UDP Bridge started. Listening on {topic_name} and forwarding to UDP port {udp_port}")

    def listener_callback(self, msg):
        # Pack x, y, and z_angular into 3 floats (12 bytes)
        data = struct.pack('fff', msg.linear.x, msg.linear.y, msg.angular.z)
        self.sock.sendto(data, (self.udp_ip, self.udp_port))

def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--namespace', type=str, default='', help='Namespace for cmd_vel topic')
    parser.add_argument('--port', type=int, default=9876, help='UDP port')
    parsed_args, ros_args = parser.parse_known_args(sys.argv)

    rclpy.init(args=ros_args)
    bridge = CmdVelUDPBridge(parsed_args.namespace, parsed_args.port)
    rclpy.spin(bridge)
    bridge.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

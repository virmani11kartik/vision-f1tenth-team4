#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

import numpy as np
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped


class SafetyNode(Node):
    """
    The class that handles emergency braking.
    """
    def __init__(self):
        super().__init__('safety_node')
        """
        One publisher should publish to the /drive topic with a AckermannDriveStamped drive message.

        You should also subscribe to the /scan topic to get the LaserScan messages and
        the /ego_racecar/odom topic to get the current speed of the vehicle.

        The subscribers should use the provided odom_callback and scan_callback as callback methods

        NOTE that the x component of the linear velocity in odom is the speed
        """
        self.speed =0.0
        self.ttc_threshold = 1.2
        
        self.odom_subscriber = self.create_subscription(Odometry, 'ego_racecar/odom', self.odom_callback, 10)
        self.get_logger().info("Odom Called")
        self.laser_subscriber = self.create_subscription(LaserScan, 'scan', self.scan_callback, 10)
        self.get_logger().info("Scan Called")
        self.publisher_ = self.create_publisher(AckermannDriveStamped, 'drive',10)
        self.get_logger().info("Ackermann Called")
        

    def odom_callback(self, odom_msg):
        self.speed = odom_msg.twist.twist.linear.x

    def scan_callback(self, scan_msg):
        if self.speed == 0:
            self.get_logger().info("Vehicle Not Moving")
            return  
        
        ranges = np.array(scan_msg.ranges)  
        angles = np.linspace(scan_msg.angle_min, scan_msg.angle_max, len(ranges))

        range_rates = -self.speed * np.cos(angles)

        with np.errstate(divide='ignore', invalid='ignore'):
            ttc_values = np.where(range_rates < 0, ranges / -range_rates, np.inf)

        min_ttc = np.min(ttc_values)

        self.get_logger().info(f"iTTC values: {min_ttc}")

        if min_ttc < self.ttc_threshold:
            drive_msg = AckermannDriveStamped()
            drive_msg.drive.speed = 0.0 
            self.publisher_.publish(drive_msg)
            self.get_logger().info("Emergency Brake Activated!")
        pass

def main(args=None):
    rclpy.init(args=args)
    safety_node = SafetyNode()
    rclpy.spin(safety_node)
    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    safety_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
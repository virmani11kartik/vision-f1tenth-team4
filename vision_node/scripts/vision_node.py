#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from ackermann_msgs.msg import AckermannDriveStamped

class LaneFollowerNode(Node):
    """
    Lane following node using RealSense D435i color image and PID control.
    """

    def __init__(self):
        super().__init__('lane_follower_node')

        self.bridge = CvBridge()

        self.subscription = self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.image_callback,
            10)
        self.get_logger().info("Subscribed to camera image")

        self.publisher_ = self.create_publisher(
            AckermannDriveStamped,
            '/drive',
            10)
        self.get_logger().info("Publishing to /drive")

        # PID parameters
        self.kp = 0.005
        self.ki = 0.0
        self.kd = 0.001
        self.prev_error = 0.0
        self.integral = 0.0

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"CVBridge Error: {e}")
            return

        height, width, _ = frame.shape
        roi = frame[int(height / 2):, :]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)

        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=150)
        lane_center = width // 2

        if lines is not None:
            left_lines, right_lines = [], []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                slope = (y2 - y1) / (x2 - x1 + 1e-6)
                if slope < -0.5:
                    left_lines.append(x1)
                    left_lines.append(x2)
                elif slope > 0.5:
                    right_lines.append(x1)
                    right_lines.append(x2)

            if left_lines and right_lines:
                left_avg = int(np.mean(left_lines))
                right_avg = int(np.mean(right_lines))
                lane_center = (left_avg + right_avg) // 2

        error = (width // 2) - lane_center
        self.integral += error
        derivative = error - self.prev_error

        steering_angle = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error

        # Construct Ackermann drive message
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = float(steering_angle)
        drive_msg.drive.speed = 1.5  # Adjust as needed
        self.publisher_.publish(drive_msg)

        # Optional: Display for debugging
        cv2.line(roi, (lane_center, 0), (lane_center, roi.shape[0]), (255, 0, 0), 2)
        cv2.imshow("Lane Detection", roi)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = LaneFollowerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()


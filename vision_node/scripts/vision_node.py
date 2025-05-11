#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

import cv2
import numpy as np
import pytesseract
import time

from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from ackermann_msgs.msg import AckermannDriveStamped


class Vision_Node(Node):
    def __init__(self):
        super().__init__('vision_node')
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(Image, "/camera/color/image_raw", self.image_callback, 10)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, "/drive", 10)
        self.last_stop_time = 0
        self.stop_duration = 5  # seconds
        self.default_speed = 1.5
        self.current_speed = self.default_speed
        self.stop_cooldown = 10  

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        output = frame.copy()

        # Red mask
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)

        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        circle_detected = False
        number_detected = False

        for cnt in contours:
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            if 0.7 < circularity < 1.2 and area > 3000:
                circle_detected = True
                break  # Stop once a red circle is found

        if circle_detected:
            current_time = time.time()
            if current_time - self.last_stop_time > self.stop_cooldown:
                self.get_logger().info("Red circle detected. Stopping for 5 seconds.")
                self.publish_speed(0.0)
                self.last_stop_time = current_time
            return

        if time.time() - self.last_stop_time < self.stop_duration:
            return

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w * h < 500:
                continue
            digit_roi = red_mask[y:y+h, x:x+w]
            text = pytesseract.image_to_string(digit_roi, config='--psm 10 digits').strip()

            if text.isdigit():
                speed_val = int(text)
                self.get_logger().info(f"Red number detected: {speed_val}")
                self.publish_speed(speed_val)  
                number_detected = True
                break

        if not number_detected:
            self.publish_speed(self.current_speed)

    def publish_speed(self, speed):
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.speed = speed
        drive_msg.drive.steering_angle = 0.0  
        self.drive_pub.publish(drive_msg)
        self.current_speed = speed


def main(args=None):
    rclpy.init(args=args)
    node = Vision_Node()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

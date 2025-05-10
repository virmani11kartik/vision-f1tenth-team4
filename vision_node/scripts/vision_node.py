#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

import cv2
import numpy as np
import time
import pytesseract
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from ackermann_msgs.msg import AckermannDriveStamped

class LaneFollowerNode(Node):
    def __init__(self):
        super().__init__('vision_node')
        self.bridge = CvBridge()

        # ROS 2 Subscribers and Publisher
        self.subscription = self.create_subscription(Image, '/camera/color/image_raw', self.image_callback, 10)
        self.publisher = self.create_publisher(AckermannDriveStamped, '/drive', 10)

        # PID Controller Parameters
        self.kp = 0.005
        self.ki = 0.0
        self.kd = 0.001
        self.prev_error = 0.0
        self.integral = 0.0

        self.last_stop_time = 0.0
        self.stop_cooldown = 6.0  # avoid repeated stops within 6 seconds

        self.get_logger().info("Vision node initialized successfully")

    def image_callback(self, msg):
        self.get_logger().info("Image callback triggered")
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        height, width, _ = frame.shape
        roi = frame[int(height * 0.5):, int(width * 0.15):int(width * 0.85)]
        roi_width = roi.shape[1]

        ### Lane Center Detection ###
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY_INV)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        centers = []
        for cnt in contours:
            M = cv2.moments(cnt)
            if M['m00'] > 50:
                cx = int(M['m10'] / M['m00'])
                centers.append(cx)

        # Compute lane center error
        if len(centers) >= 2:
            lane_center = int(np.mean(centers))
            error = (roi_width // 2) - lane_center
        else:
            error = 0

        self.integral += error
        derivative = error - self.prev_error
        steering = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error

        ### Command Detection (OCR + Circle Detection) ###
        gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, bin_full = cv2.threshold(gray_full, 60, 255, cv2.THRESH_BINARY_INV)
        full_contours, _ = cv2.findContours(bin_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        now = time.time()
        stop_detected = False
        speed_command = 1.5

        for cnt in full_contours:
            area = cv2.contourArea(cnt)
            if area < 1000:
                continue

            # Detect black circle (stop signal) using circularity
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            if 0.7 < circularity < 1.2 and area > 3000:
                if now - self.last_stop_time > self.stop_cooldown:
                    self.last_stop_time = now
                    stop_detected = True
                    self.get_logger().info("Stop signal detected. Pausing for 5 seconds...")
                    self.publish_drive_command(0.0, 0.0)
                    time.sleep(5)
                break

            # Try digit recognition
            x, y, w, h = cv2.boundingRect(cnt)
            if w * h < 500:
                continue
            roi_digit = bin_full[y:y+h, x:x+w]
            text = pytesseract.image_to_string(roi_digit, config='--psm 10 digits').strip()
            if text.isdigit():
                speed_command = min(2.0, max(0.0, float(text)))  # map number to speed [0, 2.0]
                self.get_logger().info(f"Digit command detected: speed = {speed_command}")
                break

        if not stop_detected:
            self.publish_drive_command(speed_command, float(steering))

        # Optional: show window for debugging
        cv2.imshow("Lane View", roi)
        cv2.waitKey(1)

    def publish_drive_command(self, speed, steering):
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.speed = speed
        drive_msg.drive.steering_angle = steering
        self.publisher.publish(drive_msg)


def main(args=None):
    rclpy.init(args=args)
    print("vision node started")
    node = LaneFollowerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

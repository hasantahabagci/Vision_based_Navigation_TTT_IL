#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_based_navigation_ttt.msg import OpticalFlow
from cv_bridge import CvBridge, CvBridgeError
import cv2
import sys
import numpy as np

# ROI boundaries
x_init_el = y_init_el = y_end_el = 0
y_init_er = x_end_er = y_end_er = 0
x_end_l = y_end_l = y_init_l = y_end_r = 0
x_init_r = y_init_r = 0

def set_limit(img_width, img_height):
    global x_init_el, y_init_el, y_end_el
    x_init_el = 0
    y_init_el = 0
    y_end_el = int(11 * img_height / 12)

    global x_end_er, y_end_er, y_init_er
    x_end_er = img_width
    y_end_er = int(11 * img_height / 12)
    y_init_er = 0

    global x_end_l, y_end_l, y_init_l
    x_end_l = int(4 * img_width / 12)
    y_end_l = int(7 * img_height / 12)
    y_init_l = int(1 * img_height / 12)

    global x_init_r, y_init_r, y_end_r
    x_init_r = int(8 * img_width / 12)
    y_init_r = int(1 * img_height / 12)
    y_end_r = int(7 * img_height / 12)

def draw_optical_flow_field(gray_image, points_old, points_new, flow, dt):
    color_img = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    for i in range(len(points_new)):
        x0, y0 = int(points_old[i, 0]), int(points_old[i, 1])
        x1, y1 = int(points_new[i, 0]), int(points_new[i, 1])
        cv2.line(color_img, (x0, y0), (x1, y1), (0, 255, 0), 3)

    cv2.imshow('Optical Flow', color_img)
    cv2.waitKey(10)

class OFCalculator(Node):
    def __init__(self, param):
        super().__init__('optical_flow')
        self.image_sub_name = "/camera/image"
        self.show = int(param)
        self.bridge = CvBridge()

        self.prev_image = None
        self.prev_kps = np.array([], dtype='f')
        self.prev_time = 0.0
        self.tracking = False
        self.min_feat_threshold = 0.6
        self.num_ext_features = 250
        self.num_cen_features = 150
        self.min_num_features = (2 * self.num_ext_features + self.num_cen_features) / 8
        #self.min_num_features = 200
        
        self.roi_el = np.array([])
        self.roi_er = np.array([])
        self.roi_c = np.array([])

        self.orb_extreme = cv2.ORB_create(self.num_ext_features)
        self.orb_center = cv2.ORB_create(self.num_cen_features)
        self.lk_params = dict(winSize=(15, 15), maxLevel=3,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        self.image_sub = self.create_subscription(Image, self.image_sub_name, self.callback, 10)
        self.optic_flow_pub = self.create_publisher(OpticalFlow, "optical_flow", 10)

    def callback(self, data):
        self.get_logger().info("Received image")
        try:
            curr_image = self.bridge.imgmsg_to_cv2(data, "mono8")
        except CvBridgeError as e:
            self.get_logger().error(str(e))
            return

        # Time parsing (ROS 2 Jazzy uses .sec, .nanosec)
        secs = data.header.stamp.sec
        nsecs = data.header.stamp.nanosec
        curr_time = float(secs) + float(nsecs) * 1e-9

        if self.prev_time != 0:
            frequency = 1.0 / (curr_time - self.prev_time)
            print("Frequency: {:.2f} Hz".format(frequency))

        if self.prev_image is None:
            self.prev_image = curr_image
            self.prev_time = curr_time
            set_limit(data.width, data.height)
            self.prev_kps = self.detect_features(curr_image)
            self.tracking = len(self.prev_kps) > 0
            return

        if not self.tracking:
            print("new keyframe!")
            self.prev_kps = self.detect_features(curr_image)
            self.tracking = len(self.prev_kps) > 0
            if not self.tracking:
                return

        tracked_features, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_image, curr_image,
                                                               self.prev_kps, None,
                                                               **self.lk_params)
        good_kps_new = tracked_features[status == 1]
        good_kps_old = self.prev_kps[status == 1]

        if len(good_kps_new) < self.min_feat_threshold * len(self.prev_kps) or len(good_kps_new) <= self.min_num_features:
            self.tracking = False
            self.prev_kps = np.array([], dtype='f')
            return
        else:
            self.prev_kps = good_kps_new.reshape(-1, 1, 2)

        dt = curr_time - self.prev_time
        flow = good_kps_new - good_kps_old

        if self.show == 1:
            draw_optical_flow_field(curr_image, good_kps_old, good_kps_new, flow, dt)

        msg = OpticalFlow()
        msg.header.stamp.sec = secs
        msg.header.stamp.nanosec = nsecs
        msg.height = data.height
        msg.width = data.width
        msg.dt = dt
        msg.x = good_kps_old[:, 0]
        msg.y = good_kps_old[:, 1]
        msg.vx = flow[:, 0] / dt
        msg.vy = flow[:, 1] / dt
        self.optic_flow_pub.publish(msg)

        self.prev_image = curr_image
        self.prev_time = curr_time

    def detect_features(self, image):
        self.roi_el = image[y_init_el:y_end_el, x_init_el:x_end_l]
        self.roi_er = image[y_init_er:y_end_er, x_init_r:x_end_er]
        self.roi_c = image[y_init_l:y_end_r, x_end_l:x_init_r]

        keypoints = []

        for roi, x_off, y_off, orb in [
            (self.roi_el, x_init_el, y_init_el, self.orb_extreme),
            (self.roi_er, x_init_r, y_init_er, self.orb_extreme),
            (self.roi_c, x_end_l, y_init_l, self.orb_center)]:

            kp = orb.detect(roi)
            for k in kp:
                x, y = k.pt
                k.pt = (x + x_off, y + y_off)
            keypoints.extend(kp)

        if keypoints:
            print(f"[DEBUG] Detected {len(keypoints)} total keypoints.")

            pts = cv2.KeyPoint_convert(keypoints)
            return np.float32(pts.reshape(-1, 1, 2))
    

        return np.array([], dtype='f')

def main(args=None):
    rclpy.init(args=args)
    param = sys.argv[1] if len(sys.argv) > 1 else 0
    node = OFCalculator(param)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Parameter = 1, verbose mode")
    main()

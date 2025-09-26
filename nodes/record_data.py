#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from vision_based_navigation_ttt.msg import TauComputation
import csv
import os

class DataRecorder(Node):
    def __init__(self):
        super().__init__('data_recorder')

        # --- SUBSCRIBERS ---
        # 1. Tau values (the state)
        self.subscription_tau = self.create_subscription(
            TauComputation,
            '/tau_values',
            self.tau_callback,
            10)
        
        # 2. Expert/Teleop commands (the primary action to record)
        self.subscription_cmd = self.create_subscription(
            TwistStamped,
            '/jackal_velocity_controller/cmd_vel',
            self.cmd_callback,  # This callback triggers the recording
            10)
        
        # 3. Ground truth odometry
        self.subscription_odom = self.create_subscription(
            Odometry,
            '/jackal_velocity_controller/odom',
            self.odom_callback,
            10)
            
        # 4. NEW: Default controller commands
        self.subscription_default_controller = self.create_subscription(
            TwistStamped,
            '/default_controller/cmd_vel', # Listening on the new, remapped topic
            self.default_controller_callback,
            10)
        
        # --- CLASS VARIABLES ---
        self.latest_tau_data = None
        self.latest_odom_data = None
        self.latest_default_controller_data = None # Variable to store controller output
        
        self.csv_file_path = os.path.expanduser('~/recorded_data_full.csv')
        self.csv_file = open(self.csv_file_path, 'a', newline='')
        self.csv_writer = csv.writer(self.csv_file)

        # Write header if the file is new/empty
        if os.stat(self.csv_file_path).st_size == 0:
            self.csv_writer.writerow([
                # State
                'tau_el', 'tau_er', 'tau_l', 'tau_r', 'tau_c', 'angular_z_state',
                # Expert Action
                'expert_linear_x', 'expert_angular_z',
                # Ground Truth Position
                'pos_x', 'pos_y', 'pos_z',
                # Default Controller Action
                'default_linear_x', 'default_angular_z'
            ])
        self.get_logger().info("Data recorder started. Recording to recorded_data_full.csv")

    # --- CALLBACKS to store the latest data from each topic ---
    def tau_callback(self, msg):
        self.latest_tau_data = msg
        
    def odom_callback(self, msg):
        self.latest_odom_data = msg
        
    def default_controller_callback(self, msg):
        self.latest_default_controller_data = msg

    # --- MAIN CALLBACK that triggers writing to the file ---
    def cmd_callback(self, msg):
        # This function is triggered by the expert's (teleop) command
        expert_twist_data = msg.twist

        # Check that we have received at least one message from all other topics
        if self.latest_tau_data and self.latest_odom_data and self.latest_default_controller_data:
            state = [
                self.latest_tau_data.tau_el,
                self.latest_tau_data.tau_er,
                self.latest_tau_data.tau_l,
                self.latest_tau_data.tau_r,
                self.latest_tau_data.tau_c,
                expert_twist_data.angular.z
            ]
            expert_action = [
                expert_twist_data.linear.x,
                expert_twist_data.angular.z
            ]
            position = [
                self.latest_odom_data.pose.pose.position.x,
                self.latest_odom_data.pose.pose.position.y,
                self.latest_odom_data.pose.pose.position.z
            ]
            default_controller_action = [
                self.latest_default_controller_data.twist.linear.x,
                self.latest_default_controller_data.twist.angular.z
            ]
            
            # Record only when the expert is sending a non-zero command
            if expert_action[0] != 0.0 or expert_action[1] != 0.0:
                self.csv_writer.writerow(state + expert_action + position + default_controller_action)
                self.get_logger().info(f"Recorded data point")
        else:
            self.get_logger().warn("Waiting for messages from all topics before recording...")

    def destroy_node(self):
        if self.csv_file:
            self.csv_file.close()
        self.get_logger().info(f"Data saved to {self.csv_file_path}")
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    data_recorder = DataRecorder()
    try:
        rclpy.spin(data_recorder)
    except KeyboardInterrupt:
        pass
    finally:
        data_recorder.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
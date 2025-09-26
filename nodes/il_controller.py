#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped
from vision_based_navigation_ttt.msg import TauComputation
import torch
import torch.nn as nn
import numpy as np
import os

# The model architecture remains the same (8 inputs)
class BehavioralCloningModel(nn.Module):
    def __init__(self):
        super(BehavioralCloningModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(8, 64), 
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.network(x)

class ILController(Node):
    def __init__(self):
        super().__init__('il_controller')
        
        # --- NEW: Blending factor parameter ---
        # A value of 1.0 means 100% IL model, 0.0 means 100% classical controller.
        # 0.8 is a good starting point, trusting the IL model more.
        self.alpha = 0.2
        
        # --- Subscribers ---
        self.subscription_tau = self.create_subscription(
            TauComputation, '/tau_values', self.tau_callback, 10)
            
        self.subscription_default_cmd = self.create_subscription(
            TwistStamped, '/default_controller/cmd_vel', self.default_cmd_callback, 10)
            
        self.publisher_cmd = self.create_publisher(
            TwistStamped, '/jackal_velocity_controller/cmd_vel', 10)
        
        # Load the model trained with guidance
        model_path = os.path.expanduser('~/il_model_full.pth')
        self.model = BehavioralCloningModel()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

        self.last_angular_z = 0.0
        self.latest_default_cmd = None
        
        self.get_logger().info(f"Blended IL Controller started with alpha = {self.alpha}")

    def default_cmd_callback(self, msg):
        self.latest_default_cmd = msg

    def tau_callback(self, msg):
        if self.latest_default_cmd is None:
            self.get_logger().warn("Waiting for default controller guidance...")
            return

        # Prepare state vector for the model (8 features)
        state_np = np.array([[
            msg.tau_el, msg.tau_er, msg.tau_l, msg.tau_r, msg.tau_c,
            self.last_angular_z,
            self.latest_default_cmd.twist.linear.x,
            self.latest_default_cmd.twist.angular.z
        ]])
        
        state_tensor = torch.tensor(state_np, dtype=torch.float32)

        with torch.no_grad():
            # This is the raw prediction from the IL model
            il_action_tensor = self.model(state_tensor)
        
        il_action = il_action_tensor.numpy()[0]
        
        # Get the classical controller's current command
        classical_action = np.array([
            self.latest_default_cmd.twist.linear.x,
            self.latest_default_cmd.twist.angular.z
        ])
        
        # --- NEW: Blend the two actions ---
        final_action = self.alpha * il_action + (1.0 - self.alpha) * classical_action
        
        # Create and publish the final, blended command
        twist_msg = TwistStamped()
        twist_msg.header.stamp = self.get_clock().now().to_msg()
        twist_msg.header.frame_id = "base_link"
        twist_msg.twist.linear.x = float(final_action[0])
        twist_msg.twist.angular.z = float(final_action[1])
        self.publisher_cmd.publish(twist_msg)

        self.last_angular_z = twist_msg.twist.angular.z
        
        self.get_logger().info(f"Published command: linear_x={twist_msg.twist.linear.x:.2f}, angular_z={twist_msg.twist.angular.z:.2f}")

def main(args=None):
    rclpy.init(args=args)
    il_controller = ILController()
    rclpy.spin(il_controller)
    il_controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
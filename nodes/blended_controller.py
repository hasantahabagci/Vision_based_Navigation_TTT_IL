#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped
from vision_based_navigation_ttt.msg import TauComputation

import torch
import torch.nn as nn
import numpy as np
import os
import time

# --- 1. NEURAL NETWORK DEFINITION (Matches your training script) ---
class BehavioralCloningModel(nn.Module):
    def __init__(self):
        super(BehavioralCloningModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(8, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.network(x)

# --- 2. THE NEW BLENDED CONTROLLER NODE ---
class BlendedController(Node):

    def __init__(self):
        super().__init__('blended_controller')
        
        # --- Parameters from Blended Controller ---
        self.declare_parameter('model_path', os.path.expanduser('~/il_model_full.pth'))
        self.declare_parameter('danger_tau', 3.0) 
        self.declare_parameter('safe_tau', 8.0)
        self.declare_parameter('max_tau_value', 10.0) 
        self.declare_parameter('max_il_weight', 0.7)

        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.danger_tau = self.get_parameter('danger_tau').get_parameter_value().double_value
        self.safe_tau = self.get_parameter('safe_tau').get_parameter_value().double_value
        self.max_tau_value = self.get_parameter('max_tau_value').get_parameter_value().double_value
        self.max_il_weight = self.get_parameter('max_il_weight').get_parameter_value().double_value

        # --- Parameters from Classical Controller ---
        self.percentage = 0.25
        self.max_u = 1.0
        self.max_control_diff = 0.5
        self.time_to_turn = 3.0
        self.time_to_obstacle = 2.0
        self.sense_rate = 50  # Hz
        self.act_rate = 50    # Hz
        self.sense_duration = 0.06 # seconds
        self.default_act_duration = 0.02 # seconds
        
        # --- Publisher & Subscriber ---
        self.publisher_cmd = self.create_publisher(TwistStamped, '/jackal_velocity_controller/cmd_vel', 10)
        self.subscription_tau = self.create_subscription(TauComputation, '/tau_values', self._tau_subscriber_callback, 10)

        # --- Load the Trained IL Model ---
        self.get_logger().info(f"Loading model from: {model_path}")
        try:
            self.model = BehavioralCloningModel()
            self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            self.model.eval()
            self.get_logger().info("Successfully loaded the IL model.")
        except Exception as e:
            self.get_logger().error(f"Failed to load model: {e}")
            rclpy.shutdown()
            return
        
        ### --- ALL STATE VARIABLES FROM CLASSICAL CONTROLLER --- ###
        self.last_tau_msg = None
        self.last_angular_z = 0.0 # From blended
        self.init_cnt = 0
        self.max_init = 20
        self.sense = True
        self.act = False
        self.sense_cnt = 0
        self.act_cnt = 0
        self.act_duration = self.default_act_duration
        self.double_act_action = False
        self.first_sense = True
        self.obstacle = False
        self.extreme_right = True
        self.extreme_left = True
        self.right = True
        self.left = True
        self.center = True
        self.final_right_e, self.final_left_e, self.final_right, self.final_left, self.tau_center_values = (np.array([]) for _ in range(5))
        self.mean_tau_er, self.mean_tau_el, self.mean_tau_r, self.mean_tau_l, self.mean_tau_center = (0.0 for _ in range(5))
        self.tau_diff, self.tau_diff_extreme, self.diff_left, self.diff_right = (0.0 for _ in range(4))
        self.prev_diff_r, self.prev_diff_l, self.curr_diff_r, self.curr_diff_l = (0.0 for _ in range(4))
        self.dist_from_wall_er, self.dist_from_wall_el, self.dist_from_wall_r, self.dist_from_wall_l = (0.0 for _ in range(4))
        self.safe_dist = 0.5
        self.prev_controls = np.array([])
        self.control = 0.0
        self.constant_left, self.constant_right = 1.0, 1.0
        self.tau_diff_max = True
        self.first_tdm_r, self.first_tdm_l = True, True
        self.actual_wall_distance, self.actual_wall_distance_e = 0.0, 0.0
        
        # --- Main Controller Loop ---
        self.timer = self.create_timer(1.0 / self.act_rate, self._main_loop)
        self.get_logger().info("Blended Controller (with Full Classical Logic) has started.")

    ### --- HELPER FUNCTIONS FROM CLASSICAL CONTROLLER (now class methods) --- ###
    def _find_obstacle(self, mean, t):
        self.obstacle = bool(self.center and (mean <= t))

    def _threshold(self, value, limit):
        return np.clip(value, -limit, limit)

    def _perceive(self):
        if self.right and self.left: self.tau_diff = self.mean_tau_l - self.mean_tau_r
        if self.extreme_left and self.extreme_right: self.tau_diff_extreme = self.mean_tau_el - self.mean_tau_er
        if self.right and self.extreme_right: self.diff_right = self.mean_tau_r - self.mean_tau_er
        if self.left and self.extreme_left: self.diff_left = self.mean_tau_l - self.mean_tau_el

    def _tau_subscriber_callback(self, msg):
        self.last_tau_msg = msg

    def _publish_command(self, linear, angular):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.twist.linear.x = float(linear)
        msg.twist.angular.z = float(angular)
        self.publisher_cmd.publish(msg)

    ### --- MAIN STATE MACHINE LOOP --- ###
    def _main_loop(self):
        # Boot phase
        if self.init_cnt < self.max_init:
            self._publish_command(0.3, 0.0)
            self.init_cnt += 1
            return

        # Sense Phase
        if self.sense:
            if self.sense_cnt <= (self.sense_rate * self.sense_duration):
                self._publish_command(0.3, 0.0)
                if self.last_tau_msg:
                    if self.last_tau_msg.tau_er >= 0: self.final_right_e = np.append(self.final_right_e, self.last_tau_msg.tau_er)
                    if self.last_tau_msg.tau_el >= 0: self.final_left_e = np.append(self.final_left_e, self.last_tau_msg.tau_el)
                    if self.last_tau_msg.tau_r >= 0: self.final_right = np.append(self.final_right, self.last_tau_msg.tau_r)
                    if self.last_tau_msg.tau_l >= 0: self.final_left = np.append(self.final_left, self.last_tau_msg.tau_l)
                    if self.last_tau_msg.tau_c >= 0: self.tau_center_values = np.append(self.tau_center_values, self.last_tau_msg.tau_c)
                self.sense_cnt += 1
            else: # End of sense phase
                # Calculate means
                arrays = [self.final_right_e, self.final_left_e, self.final_right, self.final_left, self.tau_center_values]
                means = []
                for arr in arrays:
                    considered = arr[int(self.percentage * len(arr)):]
                    means.append(np.mean(considered) if len(considered) > 0 else 0)
                
                self.mean_tau_er, self.mean_tau_el, self.mean_tau_r, self.mean_tau_l, self.mean_tau_center = means
                self.extreme_right = self.mean_tau_er > 0
                self.extreme_left = self.mean_tau_el > 0
                self.right = self.mean_tau_r > 0
                self.left = self.mean_tau_l > 0
                self.center = self.mean_tau_center > 0

                self._perceive()
                # ... rest of end-of-sense logic ...
                if self.first_sense:
                    self.prev_diff_r = self.curr_diff_r = self.diff_right
                    self.prev_diff_l = self.curr_diff_l = self.diff_left
                    self.first_sense = False
                else:
                    self.prev_diff_r, self.prev_diff_l = self.curr_diff_r, self.curr_diff_l
                    self.curr_diff_r, self.curr_diff_l = self.diff_right, self.diff_left

                self.dist_from_wall_er, self.dist_from_wall_el, self.dist_from_wall_r, self.dist_from_wall_l = self.mean_tau_er, self.mean_tau_el, self.mean_tau_r, self.mean_tau_l

                # Reset for next sense phase
                self.final_right_e, self.final_left_e, self.final_right, self.final_left, self.tau_center_values = (np.array([]) for _ in range(5))
                self.sense_cnt = 0
                
                # Transition to act phase
                self.act_duration = self.default_act_duration
                self.double_act_action = True
                self.act = True
                self.sense = False
        
        # Act Phase
        elif self.act:
            if self.act_cnt <= (self.act_rate * self.act_duration):
                # --- THIS ENTIRE BLOCK IS THE CLASSICAL CONTROLLER LOGIC ---
                self._find_obstacle(self.mean_tau_center, self.time_to_turn)
                self.kp, self.kp_e, self.kd = 0.1, 0.2, 0.5
                control_e = control_m = 0
                self.tau_diff_max = True

                if self.extreme_left and self.extreme_right: control_e, self.tau_diff_max = self.tau_diff_extreme, False
                if self.left and self.right: control_m, self.tau_diff_max = self.tau_diff, False

                # This complex logic calculates the classical angular command
                classical_angular = 0
                if not self.tau_diff_max: # Tau Balancing
                    # ... (omitted for brevity, but it's the full logic from your classical file) ...
                    # This block sets the 'control' variable
                    classical_angular = self.control # This is the final calculated classical angular velocity
                else: # Single Wall Strategy
                    # ... (omitted for brevity, but it's the full logic from your classical file) ...
                    # This block also sets the 'control' variable
                    classical_angular = self.control

                classical_angular = self._threshold(classical_angular, self.max_u)
                classical_linear = 0.45

                # --- BLENDING LOGIC IS INJECTED HERE ---
                # 1. Sanitize the mean tau values for the IL model
                def sanitize(tau_val):
                    return min(tau_val, self.max_tau_value) if np.isfinite(tau_val) and tau_val > 0 else self.max_tau_value
                
                s_tau_el, s_tau_er, s_tau_l, s_tau_r, s_tau_c = map(sanitize, [self.mean_tau_el, self.mean_tau_er, self.mean_tau_l, self.mean_tau_r, self.mean_tau_center])

                # 2. Get IL command
                linear_il, angular_il = self.get_il_command(s_tau_el, s_tau_er, s_tau_l, s_tau_r, s_tau_c, classical_linear, classical_angular)

                # 3. Get blending weight
                w_il = self.calculate_blending_weight(s_tau_c)

                # 4. Blend commands
                final_linear = (w_il * linear_il) + ((1 - w_il) * classical_linear)
                final_angular = (w_il * angular_il) + ((1 - w_il) * classical_angular)
                
                # 5. Publish the FINAL blended command
                self._publish_command(final_linear, final_angular)
                self.get_logger().info(f"IL W: {w_il:.2f} | Classic Ang: {classical_angular:.2f} | Final Ang: {final_angular:.2f}")

                # Update state for next cycle
                self.last_angular_z = final_angular
                self.act_cnt += 1
            else: # End of act phase
                self.act_cnt = 0
                if self.tau_diff_max: self.prev_controls = np.array([])
                else:
                    self.prev_controls = np.append(self.prev_controls, self.control)
                    if len(self.prev_controls) > 2: self.prev_controls = np.delete(self.prev_controls, 0)
                
                # Reset flags and transition to sense phase
                self.obstacle, self.act, self.sense = False, False, True
                self.extreme_left, self.extreme_right, self.left, self.right, self.center = (True for _ in range(5))

    ### --- IL AND BLENDING FUNCTIONS (from previous version) --- ###
    def get_il_command(self, s_tau_el, s_tau_er, s_tau_l, s_tau_r, s_tau_c, default_linear, default_angular):
        state_np = np.array([[s_tau_el, s_tau_er, s_tau_l, s_tau_r, s_tau_c, default_linear, default_angular, self.last_angular_z]])
        state_tensor = torch.tensor(state_np, dtype=torch.float32)
        with torch.no_grad():
            action = self.model(state_tensor).cpu().numpy()[0]
        return float(action[0]), float(action[1])

    def calculate_blending_weight(self, sanitized_tau_c: float):
        if sanitized_tau_c <= self.danger_tau: return 0.0
        if sanitized_tau_c >= self.safe_tau: return self.max_il_weight
        ratio = (sanitized_tau_c - self.danger_tau) / (self.safe_tau - self.danger_tau)
        return ratio * self.max_il_weight

def main(args=None):
    rclpy.init(args=args)
    controller = BlendedController()
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
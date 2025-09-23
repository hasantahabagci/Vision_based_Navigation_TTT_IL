#!/usr/bin/env python3
import csv, os, sys
from datetime import datetime
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from geometry_msgs.msg import TwistStamped
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

# We REQUIRE the custom Tau message so the subscription is valid.
# If this import fails, your package wasn't built/sourced.
from vision_based_navigation_ttt.msg import TauComputation as TauMsg

def extract_tau_vals(msg):
    # Primary field set (adjust if your message differs)
    names = ['tau_fl','tau_fr','tau_l','tau_r','tau_c']
    vals = [float(getattr(msg, n, -1.0)) for n in names]
    return vals

class ILDemoLogger(Node):
    def __init__(self):
        super().__init__('il_demo_logger')

        # ---- Parameters ----
        self.declare_parameter('tau_topic', '/tau_computation')
        self.declare_parameter('cmd_topic', 'jackal_velocity_controller/cmd_vel')
        self.declare_parameter('out_csv',   'assets/il_demos.csv')
        self.declare_parameter('flush_every_n', 5)  # flush every N rows
        self.declare_parameter('min_rows_log', 1)    # print summary every N rows

        tau_topic = self.get_parameter('tau_topic').get_parameter_value().string_value
        cmd_topic = self.get_parameter('cmd_topic').get_parameter_value().string_value
        out_csv   = self.get_parameter('out_csv').get_parameter_value().string_value
        self.flush_every_n = int(self.get_parameter('flush_every_n').value)
        self.min_rows_log  = int(self.get_parameter('min_rows_log').value)

        os.makedirs(os.path.dirname(out_csv) or '.', exist_ok=True)
        # line-buffered writes so you can watch the file grow while running
        self.f = open(out_csv, 'w', newline='', buffering=1)
        self.w = csv.writer(self.f)
        self.w.writerow(['t','tau_fl','tau_fr','tau_l','tau_r','tau_c','u','v'])

        self.rows = 0
        self.latest_cmd = Twist()

        # Sensor-style QoS tends to work best for perception topics
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Subscriptions
        self.sub_tau = self.create_subscription(TauMsg, tau_topic, self.cb_tau, sensor_qos)
        self.sub_cmd = self.create_subscription(TwistStamped,   cmd_topic, self.cb_cmd, 10)

        self.get_logger().info(
            f"Recording τ from [{tau_topic}] and cmd from [{cmd_topic}] → {out_csv}"
        )

        # Periodic heartbeat so you know it's alive, even if no τ arrives
        self.timer = self.create_timer(2.0, self.heartbeat)

    def heartbeat(self):
        self.get_logger().debug(f"Rows written: {self.rows}")

    def cb_cmd(self, msg: TwistStamped):
        self.latest_cmd = msg

    def cb_tau(self, msg: TauMsg):
        try:
            tau = extract_tau_vals(msg)
        except Exception as e:
            self.get_logger().error(f"τ parse error: {e}")
            return

        # timestamp
        t = 0.0
        if hasattr(msg, 'header'):
            t = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9

        row = [
            t, tau[0], tau[1], tau[2], tau[3], tau[4],
            float(self.latest_cmd.angular.z), float(self.latest_cmd.linear.x)
        ]
        self.w.writerow(row)
        self.rows += 1

        if (self.rows % self.min_rows_log) == 0:
            self.get_logger().info(f"Wrote {self.rows} rows (last u={row[6]:.3f}, v={row[7]:.3f})")

        if (self.rows % self.flush_every_n) == 0:
            self.f.flush()

    def destroy_node(self):
        try:
            if not self.f.closed:
                self.f.flush()
                self.f.close()
        except Exception:
            pass
        super().destroy_node()

def main():
    rclpy.init()
    try:
        node = ILDemoLogger()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

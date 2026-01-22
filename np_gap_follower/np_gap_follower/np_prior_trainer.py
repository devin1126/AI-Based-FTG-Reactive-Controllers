#!/usr/bin/env python3
import rclpy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from collections import deque


# -----------------------------------------------------------
# Bin distance readings into 'num_bins' equal angular regions
# -----------------------------------------------------------
def bin_laser_scans(proc_ranges, num_bins=32, max_range=6.0):
    """
    Condenses raw LiDAR readings into N averaged distance bins and returns
    the bin with the largest distance along with its FOV angle AND its distance.

    Returns:
        binned (torch.Tensor): [num_bins] normalized in [0,1]
        phi_star (torch.Tensor): [1] angle in radians (NOT normalized here)
        d_star (torch.Tensor): [1] normalized max distance in [0,1]
    """
    laser_len = proc_ranges.shape[0]
    segment_size = laser_len // num_bins

    binned = torch.zeros(num_bins)
    for i in range(num_bins):
        start = i * segment_size
        end = (i + 1) * segment_size if i < num_bins - 1 else laser_len
        binned[i] = torch.mean(proc_ranges[start:end])

    # Clip + normalize
    binned = torch.clamp(binned, 0.0, max_range)
    binned_norm = binned / max_range

    # Find max bin and compute its angle
    max_idx = torch.argmax(binned_norm).item()
    d_star = binned_norm[max_idx].unsqueeze(0)  # [1]

    # Angle calculations (from +135 to -135 degrees)
    fov_rad = np.deg2rad(270.0)
    angle_step = fov_rad / num_bins
    angle_left = np.deg2rad(135.0)
    phi_star = angle_left - (max_idx + 0.5) * angle_step  # scalar
    phi_star = torch.tensor([phi_star], dtype=torch.float32)  # [1]

    return binned_norm, phi_star, d_star



class PriorMLP(nn.Module):
    def __init__(self, in_dim=4, hidden_dim=96, delta_max=0.6981):
        super().__init__()
        self.delta_max = delta_max

        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 1)

        self.act = nn.ReLU()

    def forward(self, x):
        x = self.act(self.ln1(self.fc1(x)))
        x = self.act(self.ln2(self.fc2(x)))
        x = self.act(self.ln3(self.fc3(x)))
        x = torch.tanh(self.fc4(x)) * self.delta_max
        return x



class PriorMLPTrainer(Node):
    def __init__(self):
        super().__init__("prior_mlp_trainer_node")

        # -----------------------------
        # Parameters
        # -----------------------------
        self.declare_parameter("max_lidar_dist", 6.0)
        self.declare_parameter("forward_crop", 135)
        self.declare_parameter("num_bins", 256)

        self.declare_parameter("angle_magnitude", 3.0)   # normalize phi* by /angle_magnitude
        self.declare_parameter("max_linear_vel", 5.0)    # normalize v by /max_linear_vel
        self.declare_parameter("max_angular_vel", 5.0)   # normalize w by /max_angular_vel

        self.declare_parameter("batch_size", 100)
        self.declare_parameter("iters_before_optimize", 50)
        self.declare_parameter("max_buffer_size", int(5e5))
        self.declare_parameter("purge_threshold", 20000)

        self.declare_parameter("lr", 1e-3)
        self.declare_parameter("train_mode", False)
        self.declare_parameter("save_path", "prior_mlp_weights.pth")
        self.declare_parameter("load_path", "/home/devin1126/cavrel_racer/racer_ws/src/np_gap_follower/np_gap_follower/prior_mlp_weights_updated.pth")

        self.declare_parameter("report_period_sec", 5.0)
        self.declare_parameter("window", 2000)

        self.max_lidar_dist = float(self.get_parameter("max_lidar_dist").value)
        self.forward_crop = int(self.get_parameter("forward_crop").value)
        self.num_bins = int(self.get_parameter("num_bins").value)

        self.angle_magnitude = float(self.get_parameter("angle_magnitude").value)
        self.max_linear_vel = float(self.get_parameter("max_linear_vel").value)
        self.max_angular_vel = float(self.get_parameter("max_angular_vel").value)

        self.batch_size = int(self.get_parameter("batch_size").value)
        self.iters_before_optimize = int(self.get_parameter("iters_before_optimize").value)
        self.max_buffer_size = int(self.get_parameter("max_buffer_size").value)
        self.purge_threshold = int(self.get_parameter("purge_threshold").value)

        self.lr = float(self.get_parameter("lr").value)
        self.train_mode = bool(self.get_parameter("train_mode").value)
        self.save_path = str(self.get_parameter("save_path").value)
        self.load_path = str(self.get_parameter("load_path").value)
        self.report_period_sec = float(self.get_parameter("report_period_sec").value)
        self.window = int(self.get_parameter("window").value)

        # -----------------------------
        # Replay buffer
        # Each entry: (x=[4], y=[1])
        # -----------------------------
        self.replay_buffer = deque(maxlen=self.max_buffer_size)

        # -----------------------------
        # Model + optimizer
        # -----------------------------
        self.device = torch.device("cpu")
        self.model = PriorMLP(in_dim=4, hidden_dim=64).to(self.device)
        if not self.train_mode:
            self.model.load_state_dict(torch.load(self.load_path, map_location=self.device))
            self.get_logger().info(f"Loaded Prior-MLP weights from: {self.load_path}")

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()

        # -----------------------------
        # ROS interfaces
        # -----------------------------
        self.scan_sub = self.create_subscription(
            LaserScan, "/scan", self.scan_callback, 10
        )
        self.odom_sub = self.create_subscription(
            Odometry, "/ego_racecar/odom", self.odom_callback, 10
        )
        self.drive_sub = self.create_subscription(
            AckermannDriveStamped, "/drive", self.drive_callback, 10
        )

        # -----------------------------
        # Cached latest messages
        # -----------------------------
        self.latest_scan = None
        self.latest_odom = None
        self.latest_drive = None

        # -----------------------------
        # Training stats
        # -----------------------------
        self.internal_counter = 0
        self.train_steps = 0

        self.loss_window = deque(maxlen=self.window)
        self.mae_window = deque(maxlen=self.window)

        self.last_report_time = self.get_clock().now()

        self.get_logger().info("PriorMLPTrainer node initialized.")

        if self.train_mode:
            self.get_logger().info("TRAIN MODE ENABLED.")
        else:
            self.get_logger().info("TEST MODE ENABLED.")

    def odom_callback(self, msg: Odometry):
        self.latest_odom = msg

    def drive_callback(self, msg: AckermannDriveStamped):
        self.latest_drive = msg

    def preprocess_scan(self, scan_msg: LaserScan):
        ranges = np.array(scan_msg.ranges, dtype=np.float32)

        ranges = np.nan_to_num(
            ranges,
            nan=0.0,
            posinf=self.max_lidar_dist,
            neginf=0.0
        )
        ranges = np.clip(ranges, 0.0, self.max_lidar_dist)

        return ranges

    def normalize_features(self, phi_star, d_star, v, w):
        """
        Desired normalization:
          phi*  -> [-1,1] via /angle_magnitude (angle_magnitude ~ 3.0 rad)
          d*    -> [0,1] already
          v     -> [0,1] via /max_linear_vel
          w     -> [-1,1] via /max_angular_vel
        """
        phi_n = torch.clamp(phi_star / self.angle_magnitude, -1.0, 1.0)
        d_n = torch.clamp(d_star, 0.0, 1.0)
        v_n = torch.clamp(v / self.max_linear_vel, 0.0, 1.0)
        w_n = torch.clamp(w / self.max_angular_vel, -1.0, 1.0)
        return phi_n, d_n, v_n, w_n

    def sample_batch(self):
        idxs = np.random.choice(len(self.replay_buffer), size=self.batch_size, replace=False)
        xs = []
        ys = []
        for i in idxs:
            x, y = self.replay_buffer[i]
            xs.append(x)
            ys.append(y)
        x_batch = torch.stack(xs, dim=0).to(self.device)  # [B,4]
        y_batch = torch.stack(ys, dim=0).to(self.device)  # [B,1]
        return x_batch, y_batch

    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return None

        self.model.train()
        x_batch, y_batch = self.sample_batch()

        pred = self.model(x_batch)
        loss = self.loss_fn(pred, y_batch)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            mae = torch.mean(torch.abs(pred - y_batch)).item()

        return loss.item(), mae

    def maybe_report(self):
        now = self.get_clock().now()
        dt = (now - self.last_report_time).nanoseconds * 1e-9
        if dt < self.report_period_sec:
            return

        self.last_report_time = now

        if len(self.loss_window) == 0:
            return

        mean_loss = float(np.mean(self.loss_window))
        std_loss = float(np.std(self.loss_window))
        mean_mae = float(np.mean(self.mae_window))
        std_mae = float(np.std(self.mae_window))

        self.get_logger().info(
            f"[TRAIN MODE] steps={self.train_steps} | "
            f"buffer={len(self.replay_buffer)} | "
            f"loss={mean_loss:.6f} ± {std_loss:.6f} | "
            f"mae={mean_mae:.6f} ± {std_mae:.6f}"
        )

    def scan_callback(self, scan_msg: LaserScan):
        # Need odom + drive to form aligned sample
        if self.latest_odom is None or self.latest_drive is None:
            return

        # -----------------------------
        # Extract current-step features
        # -----------------------------
        ranges = self.preprocess_scan(scan_msg)
        forward_ranges = ranges[self.forward_crop:-self.forward_crop]

        scan_tensor = torch.tensor(forward_ranges, dtype=torch.float32)

        _, phi_star, d_star = bin_laser_scans(
            scan_tensor,
            num_bins=self.num_bins,
            max_range=self.max_lidar_dist
        )

        v = torch.tensor([self.latest_odom.twist.twist.linear.x], dtype=torch.float32)
        w = torch.tensor([self.latest_odom.twist.twist.angular.z], dtype=torch.float32)

        # Label = steering command at the "current" step
        delta = torch.tensor([self.latest_drive.drive.steering_angle], dtype=torch.float32)

        # Normalize
        phi_n, d_n, v_n, w_n = self.normalize_features(phi_star, d_star, v, w)

        x = torch.cat([phi_n, d_n, v_n, w_n], dim=0)  # [4]
        y = delta  # [1]

        # Store in replay buffer
        self.replay_buffer.append((x.detach(), y.detach()))

       # self.get_logger().info(f'[TEST MODE]: True steering: {delta.item():.4f}, Prior steering: {self.model(x.unsqueeze(0).to(self.device)).item():.4f}')

        # Purge if needed
        if len(self.replay_buffer) == self.max_buffer_size:
            self.get_logger().info(f"Purging {self.purge_threshold} oldest samples...")
            for _ in range(self.purge_threshold):
                if len(self.replay_buffer) > 0:
                    self.replay_buffer.popleft()



        # -----------------------------
        # Training step
        # -----------------------------
        self.internal_counter += 1
        if self.internal_counter % self.iters_before_optimize == 0 and self.train_mode:
            out = self.train_step()
            if out is not None:
                loss_val, mae_val = out
                self.train_steps += 1
                self.loss_window.append(loss_val)
                self.mae_window.append(mae_val)

        if self.train_mode: 
            self.maybe_report()
        else:
            self.get_logger().info(f'[TEST MODE]: True steering: {delta.item():.4f}, Prior steering: {self.model(x.unsqueeze(0).to(self.device)).item():.4f}')




def main(args=None):
    rclpy.init(args=args)
    node = PriorMLPTrainer()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("KeyboardInterrupt: shutting down.")

        if node.train_mode:

            # Save weights
            torch.save(node.model.state_dict(), node.save_path)
            node.get_logger().info(f"Saved Prior-MLP weights to: {node.save_path}")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

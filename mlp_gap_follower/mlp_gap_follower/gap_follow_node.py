#!/usr/bin/env python3
import rclpy
import torch
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from mlp_gap_follower.mlp_model_structure import ResMLPPolicy
from mlp_gap_follower.mlp_model_utils import (
    train_mlp_model, 
    bin_laser_scans, 
    compute_steering_rate,
    MLPLoss,
    LoopTimerROS
)
from ackermann_msgs.msg import AckermannDriveStamped
from collections import deque
import matplotlib.pyplot as plt 


class MLPGapFollower(Node):
    def __init__(self):
        super().__init__('mlp_gap_follower_node')

        # --- MLP Model Parameters ---
        self.batch_size = 100  # Batch size for training
        self.iters_before_optimize = 50  # Number of iterations before each optimization step
        self.max_buffer_size = int(5e5)  # Maximum size of the replay buffer
        self.purge_threshold = 20000  # Number of oldest samples to remove when purging buffer

        # --- Initialize Replay Buffer ---
        self.replay_buffer = deque(maxlen=self.max_buffer_size)

        # --- Load Controller NP Model ---
        self.declare_parameter('gap_dim', 256)
        self.declare_parameter('vel_embed_dim', 32)
        self.declare_parameter('odom_dim', 2)
        self.gap_dim = self.get_parameter('gap_dim').value
        self.vel_embed_dim = self.get_parameter('vel_embed_dim').value
        self.odom_dim = self.get_parameter('odom_dim').value
        self.model = ResMLPPolicy(gap_dim=self.gap_dim,
                                  vel_embed_dim=self.vel_embed_dim,
                                  num_layers=4,
                                  output_size=1
                                )

        # Load pre-trained weights (paths can be parameterized as needed)
        self.declare_parameter('model_weights_path', '/home/devin1126/cavrel_racer/racer_ws/src/mlp_gap_follower/mlp_gap_follower/mlp_model_weights_updated.pth')   # Enter path to pre-trained model weights to enable INFERENCE mode
        mlp_weights_path = self.get_parameter('model_weights_path').value
        if mlp_weights_path:
            self.get_logger().info(f'MLP Gap Follower Node initialized in INFERENCE mode.')
            self.train_mode = False
            self.model.load_state_dict(torch.load(mlp_weights_path))
            self.model.eval()
        else:
            self.train_mode = True
            self.get_logger().info("MLP Gap Follower Node initialized in TRAINING mode.")

        # --- Optimizer and Loss Function ---
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.loss_function = MLPLoss()

        # --- ROS2 Subscribers and Publisher ---
        self.drive_sub = self.create_subscription(
            AckermannDriveStamped,
            '/drive',
            self.drive_callback,
            10)
        
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10)
        
        self.odom_sub = self.create_subscription(
            Odometry,
            '/ego_racecar/odom',
            self.odom_callback,
            10)
        
        self.drive_pub = self.create_publisher(
            AckermannDriveStamped,
            '/drive',
            10)
        
        # Internal counters and state
        self.internal_counter = 0
        self.training_steps = 0

        # Control parameters
        self.declare_parameter('max_lidar_dist', 6.0)
        self.declare_parameter('max_steer', 0.6981) # 40 degrees in radians
        self.declare_parameter('forward_crop', 135)
        self.declare_parameter('preprocess_conv_size', 3)
        self.declare_parameter('straights_steering_angle', 0.1396)
        self.declare_parameter('fast_steering_angle', 0.0698)
        self.declare_parameter('corners_speed', 1.5)
        self.declare_parameter('straights_speed', 3.0)
        self.declare_parameter('fast_speed', 5.0)
        self.max_lidar_dist = self.get_parameter('max_lidar_dist').value
        self.max_steer = self.get_parameter('max_steer').value
        self.forward_crop = self.get_parameter('forward_crop').value
        self.preprocess_conv_size = self.get_parameter('preprocess_conv_size').value
        self.straights_steering_angle = self.get_parameter('straights_steering_angle').value
        self.fast_steering_angle = self.get_parameter('fast_steering_angle').value
        self.corners_speed = self.get_parameter('corners_speed').value
        self.straights_speed = self.get_parameter('straights_speed').value
        self.fast_speed = self.get_parameter('fast_speed').value

        # Current state/control variables
        self.drive_label = None
        self.odom_msg = None

        # Logging arrays
        self.mae_array = []
        self.nll_array = []
        self.delta_rate_array = []
        self.prev_steer = None

        # Loop time logger
        self.ftg_timer = LoopTimerROS(self, window=1000, report_period=5.0)

    # Callback function to process drive commands
    def drive_callback(self, drive_msg):
        """Process drive commands and publish NP-based drive commands"""
        self.drive_label = drive_msg

        if self.prev_steer is None:
            self.prev_steer = drive_msg.drive.steering_angle
        
    # Callback function to process odometry messages
    def odom_callback(self, odom_msg):
        self.odom_msg = odom_msg

    def preprocess_scan(self, scan):
        # Convert to numpy
        ranges = np.array(scan.ranges)

        # Replace inf, -inf, nan
        ranges = np.nan_to_num(
            ranges,
            nan=0.0,
            posinf=self.max_lidar_dist,
            neginf=0.0
        )

        # Clip distances
        ranges = np.clip(ranges, 0.0, self.max_lidar_dist)

        # 1. Crop forward-facing arc
        ranges = ranges[self.forward_crop:-self.forward_crop]

        # 2. Moving average smoothing
        kernel = np.ones(self.preprocess_conv_size) / self.preprocess_conv_size
        ranges = np.convolve(ranges, kernel, mode='same')

        # 3. Clip again
        ranges = np.clip(ranges, 0.0, self.max_lidar_dist)

        return ranges

    
    # Callback function to publish np odometry data at a fixed rate
    def scan_callback(self, scan_msg):

        t0 = self.get_clock().now()

        # Convert to numpy
        ranges = self.preprocess_scan(scan_msg)

        """Obtain raw LiDAR scan and process through CNN encoder for gap representation"""
        # Convert processed scan (810) → PyTorch tensor
        scan_tensor = torch.tensor(ranges, dtype=torch.float32)

        # Get condensed distance readings
        gap_msg = bin_laser_scans(scan_tensor, num_bins=self.gap_dim, max_range=self.max_lidar_dist)

        """Process learned gap data and publish drive commands"""
        if self.train_mode:
            if self.odom_msg is None or self.drive_label is None:
                self.get_logger().info("Waiting for initial velocity and drive data...")
                return
            # Create training data tensor
            data = torch.cat((gap_msg, 
                              torch.tensor([self.odom_msg.twist.twist.linear.x,
                                            self.odom_msg.twist.twist.angular.z,
                                            self.drive_label.drive.steering_angle])), dim=0).unsqueeze(0)
        else:
            if self.odom_msg is None:
                self.get_logger().info("Waiting for initial velocity data...")
                return
            # Create inference data tensor
            data = torch.cat((gap_msg, 
                              torch.tensor([self.odom_msg.twist.twist.linear.x,
                                            self.odom_msg.twist.twist.angular.z,
                                            0.0])), dim=0).unsqueeze(0)

        # Append data to replay buffer (only in TRAINING mode)
        if self.train_mode: self.replay_buffer.append(data)

        # Check if buffer size exceeds the purge threshold
        if len(self.replay_buffer) == self.max_buffer_size:
            self.get_logger().info(f"Purging {self.purge_threshold} oldest tensors from replay buffer...")
            for _ in range(self.purge_threshold):
                self.replay_buffer.popleft()

        # Predicting the next steering angle with uncertainty estimate
        input_dim = self.gap_dim + self.odom_dim
        input_data = data[:,:input_dim]
        pred_y, std = self.model(input_data, is_testing=True)

        # Training the model (only done after every 'self.batch_size' iterations to lower computation costs)
        if self.internal_counter % self.iters_before_optimize == 0 and self.internal_counter > 0 and self.train_mode:
            self.training_steps += 1
            self.get_logger().info('Training model...')
            self.model, loss, MAE, MAE_std, NLL = train_mlp_model(self.model, 
                                                                  self.replay_buffer, 
                                                                  self.optimizer, 
                                                                  self.loss_function, 
                                                                  self.batch_size,
                                                                  input_dim)

            self.get_logger().info(f"Weight Update Complete! Step: {self.training_steps}, Current loss: {loss.item()}, Current MAE: {MAE.item()}, Current MAE Std: {MAE_std.item()}")
            self.get_logger().info(f'Current NLL: {NLL.item()}, Steering Angle Error: {abs(pred_y.item() - self.drive_label.drive.steering_angle)}\n')
            self.mae_array.append(MAE.item())
            self.nll_array.append(NLL.item())

            # Set the model back to testing mode
            self.model.eval() 

        # Extracting predicted odometry state into ROS2 message to be published (only in INFERENCE mode)
        """ Create and publish drive message """
        pred_drive = AckermannDriveStamped()
        timestamp = scan_msg.header.stamp.sec + scan_msg.header.stamp.nanosec * 1e-9
        pred_drive.header.stamp.sec = int(timestamp)
        pred_drive.header.stamp.nanosec = int((timestamp - int(timestamp)) * 1e9)
        pred_drive.header.frame_id = "base_link"

        # Clamp steering angle to max limits
        steer_command = torch.clamp(pred_y, -self.max_steer, self.max_steer)
        steer_command = steer_command.item()

        # Determine speed based on steering angle
        if abs(steer_command) > self.straights_steering_angle:
            speed = self.corners_speed
        elif abs(steer_command) > self.fast_steering_angle:
            speed = self.straights_speed
        else:
            speed = self.fast_speed

        # Populate and publish drive message (only in INFERENCE mode)
        pred_drive.drive.speed = speed
        pred_drive.drive.steering_angle = steer_command   #[-max_steer, max_steer] range
        if not self.train_mode: self.drive_pub.publish(pred_drive)

        # Log steering rate
        if self.prev_steer is not None:
            self.delta_rate_array.append(compute_steering_rate(self.prev_steer, pred_drive.drive.steering_angle))

        # Update previous states
        self.prev_steer = steer_command
        
        # Increment internal counter (for training purposes)
        if self.train_mode: self.internal_counter += 1

        # Loop time logging
        t1 = self.get_clock().now()
        self.ftg_timer.push_time_pair(t0, t1)

        if self.ftg_timer.should_report():
            stats = self.ftg_timer.stats()
            if stats is not None:
                mean_dt, std_dt, min_dt, max_dt, hz, n = stats
                self.get_logger().info(
                    f"[FTG LOOP] N={n} | "
                    f"mean={mean_dt*1000:.3f} ms ± {std_dt*1000:.3f} ms | "
                    f"min={min_dt*1000:.3f} ms | max={max_dt*1000:.3f} ms | "
                    f"~{hz:.1f} Hz"
                )

        

def main(args=None):
    try:
        rclpy.init(args=args)

        # Now pass the model to the ROS2 node
        node = MLPGapFollower()
        rclpy.spin(node)
        rclpy.shutdown()
    except KeyboardInterrupt:
        print("Keyboard interrupt received. Shutting down node.")
        print(f'Average steering rate during operation: {np.abs(np.mean(node.delta_rate_array))} rad/s')
        if node.train_mode:
            # Save model after shutdown
            save_path = 'mlp_model_weights.pth'
            torch.save(node.model.state_dict(), save_path)
            print(f"Trained model weights saved to {save_path}")

            # Plot RMSE and NLL over time
            plt.figure(figsize=(12, 5))

            plt.subplot(1, 2, 1)
            plt.plot(node.mae_array, '-ro')
            plt.title('MAE Convergence Dynamics')
            plt.xlabel('Training Step')
            plt.ylabel('MAE')

            plt.subplot(1, 2, 2)
            plt.plot(node.nll_array, '-bo')
            plt.title('NLL Convergence Dynamics')
            plt.xlabel('Training Step')
            plt.ylabel('NLL')

            plt.tight_layout()
            plt.savefig('mae_nll_plot.png')
            plt.show()
            print("Plot saved to mae_nll_plot.png")
            np.save('mae_array_mlp.npy', np.array(node.mae_array))
            np.save('nll_array_mlp.npy', np.array(node.nll_array))
            print("MAE and NLL arrays saved as .npy files.")
    


if __name__ == '__main__':
    main()

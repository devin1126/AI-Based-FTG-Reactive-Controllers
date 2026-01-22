#!/usr/bin/env python3
import rclpy
import torch
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from np_gap_follower.np_model_structure import AttLNP, PriorMLP
from np_gap_follower.np_model_utils import (
    train_np_model, 
    compute_control_prior,
    compute_steering_rate,
    cbf_filter,
    bin_laser_scans,
    ELBO,
    LoopTimerROS
)
from ackermann_msgs.msg import AckermannDriveStamped
from collections import deque
import matplotlib.pyplot as plt 
import time


class NPGapFollower(Node):
    def __init__(self):
        super().__init__('np_gap_follower_node')

        # --- NP Model Parameters ---
        self.batch_size = 100  # Batch size for training
        self.iters_before_optimize = 50  # Number of iterations before each optimization step
        self.max_buffer_size = int(5e5)  # Maximum size of the replay buffer
        self.purge_threshold = 20000  # Number of oldest samples to remove when purging buffer

        # --- Initialize Replay Buffer ---
        self.replay_buffer = deque(maxlen=self.max_buffer_size)

        # --- Load Controller NP Model ---
        self.declare_parameter('gap_dim', 256)
        self.declare_parameter('vel_dim', 2)
        self.declare_parameter('vel_embed_dim', 32)
        self.declare_parameter('enable_ode_model',True)
        self.gap_dim = self.get_parameter('gap_dim').value
        self.vel_dim = self.get_parameter('vel_dim').value
        self.vel_embed_dim = self.get_parameter('vel_embed_dim').value
        self.enable_ode_model = self.get_parameter('enable_ode_model').value
        self.model = AttLNP(control_dim=1, 
                            gap_dim=self.gap_dim, 
                            vel_embed_dim=self.vel_embed_dim,
                            R_dim=128, 
                            enable_ode_model=self.enable_ode_model,  # Toggle PI model usage
                            device='cpu')

        # Load pre-trained weights (paths can be parameterized as needed)
        self.declare_parameter('model_weights_path', '/home/devin1126/cavrel_racer/racer_ws/src/np_gap_follower/np_gap_follower/pinp_model_weights_updated.pth')   # Enter path to pre-trained model weights to enable INFERENCE mode
        np_weights_path = self.get_parameter('model_weights_path').value
        if np_weights_path:
            self.get_logger().info(f'NP Gap Follower Node initialized in INFERENCE mode.')
            self.train_mode = False
            self.model.load_state_dict(torch.load(np_weights_path))
            self.model.eval()
        else:
            self.train_mode = True
            self.get_logger().info("NP Gap Follower Node initialized in TRAINING mode.")

        # --- Log whether PI model is enabled ---
        if self.enable_ode_model:
            self.get_logger().info("PI model ENABLED; using approximate control prior for NP control estimation.")
        else:
            self.get_logger().info("PI model DISABLED; using end-to-end NP control estimation.")

        # --- Optimizer and Loss Function ---
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.loss_function = ELBO()


        # --- Pre-Trained Prior Model Initialization (for PI model control prior) ---
        if self.enable_ode_model:
            self.prior_model = PriorMLP(in_dim=self.vel_dim + 2,  # max_gap_angle + d_max + (v,w)
                                        hidden_dim=64)
            
            # Load pre-trained prior model weights
            self.declare_parameter('prior_model_weights_path', '/home/devin1126/cavrel_racer/racer_ws/src/np_gap_follower/np_gap_follower/prior_mlp_weights_updated.pth')
            prior_weights_path = self.get_parameter('prior_model_weights_path').value
            if prior_weights_path:
                self.prior_model.load_state_dict(torch.load(prior_weights_path))
                self.prior_model.eval()
                self.get_logger().info("Prior MLP model weights loaded successfully.")
            else:
                self.get_logger().error("Prior model weights path not provided; PI model cannot compute control prior.")

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
        self.declare_parameter('angle_magnitude', 3.0)
        self.max_lidar_dist = self.get_parameter('max_lidar_dist').value
        self.max_steer = self.get_parameter('max_steer').value
        self.forward_crop = self.get_parameter('forward_crop').value
        self.preprocess_conv_size = self.get_parameter('preprocess_conv_size').value
        self.straights_steering_angle = self.get_parameter('straights_steering_angle').value
        self.fast_steering_angle = self.get_parameter('fast_steering_angle').value
        self.corners_speed = self.get_parameter('corners_speed').value
        self.straights_speed = self.get_parameter('straights_speed').value
        self.fast_speed = self.get_parameter('fast_speed').value
        self.angle_magnitude = self.get_parameter('angle_magnitude').value

        # Previous state/control variables
        self.prev_steering_angle = None
        self.prev_gap = None
        self.prev_linear_velocity = None
        self.prev_angular_velocity = None
        self.prev_max_angle = None

        # Logging arrays
        self.mae_array = []
        self.nll_array = []
        self.delta_rate_array = []

        # Loop timer for performance monitoring
        self.declare_parameter('log_times', False)
        self.log_times = self.get_parameter('log_times').value
        if self.log_times:
            self.ftg_timer = LoopTimerROS(self, window=5000, report_period=25.0)
            self.cbf_timer = LoopTimerROS(self, window=5000, report_period=30.0)


    # Callback function to process drive commands
    def drive_callback(self, drive_msg):
        """Process drive commands and publish NP-based drive commands"""
        self.drive_label = drive_msg

        if self.prev_steering_angle is None:
            self.prev_steering_angle = self.drive_label.drive.steering_angle
            return
        
    # Callback function to process odometry messages
    def odom_callback(self, odom_msg):
        self.odom_msg = odom_msg

        """Process odometry messages if needed for future extensions"""
        if self.prev_linear_velocity is None or self.prev_angular_velocity is None:
            self.prev_linear_velocity = self.odom_msg.twist.twist.linear.x
            self.prev_angular_velocity = self.odom_msg.twist.twist.angular.z
            return
        
    
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

        # 1. Moving average smoothing
        kernel = np.ones(self.preprocess_conv_size) / self.preprocess_conv_size
        ranges = np.convolve(ranges, kernel, mode='same')

        # 2. Clip again
        ranges = np.clip(ranges, 0.0, self.max_lidar_dist)

        return ranges

    
    # Callback function to publish np odometry data at a fixed rate
    def scan_callback(self, scan_msg):

        # Start loop timer
        t0 = self.get_clock().now()

        # Preprocess LiDAR scan and crop forward-facing arc
        ranges = self.preprocess_scan(scan_msg)
        forward_ranges = ranges[self.forward_crop:-self.forward_crop]

        """Obtain raw LiDAR scan and process through CNN encoder for gap representation"""
        # Convert processed scan (810) → PyTorch tensor
        scan_tensor = torch.tensor(forward_ranges, dtype=torch.float32)

        # Get condensed distance readings
        gap_msg, max_gap_angle, d_max = bin_laser_scans(scan_tensor, num_bins=self.gap_dim, max_range=self.max_lidar_dist)

        """Process learned gap data and publish drive commands"""
        if self.train_mode:
            if self.prev_gap is None or self.prev_linear_velocity is None \
                  or self.prev_angular_velocity is None or self.prev_steering_angle is None \
                  or self.prev_max_angle is None:
                self.prev_gap = gap_msg #torch.tensor(gap_msg.data, dtype=torch.float32)
                self.prev_max_angle = max_gap_angle / self.angle_magnitude
                self.get_logger().info("Waiting for initial gap, velocity, and drive data...")
                return
            # Forming data tensors for training context/target sets
            tensor1 = torch.cat((self.prev_gap, 
                                 torch.tensor([self.prev_linear_velocity, 
                                               self.prev_angular_velocity, 
                                               self.prev_steering_angle])), dim=0).unsqueeze(0)
            tensor2 = torch.cat((gap_msg, 
                                 torch.tensor([self.odom_msg.twist.twist.linear.x, 
                                               self.odom_msg.twist.twist.angular.z,
                                               self.drive_label.drive.steering_angle])), dim=0).unsqueeze(0)  
        else:
            if self.prev_gap is None or self.prev_linear_velocity is None \
                  or self.prev_angular_velocity is None or self.prev_max_angle is None:
                self.prev_gap = gap_msg #torch.tensor(gap_msg.data, dtype=torch.float32)
                self.prev_steering_angle = 0.0
                self.prev_max_angle = max_gap_angle / self.angle_magnitude
                self.get_logger().info("Waiting for initial gap and velocity data...")
                return
            # Forming data tensors for inference context/target sets
            tensor1 = torch.cat((self.prev_gap, 
                                 torch.tensor([self.prev_linear_velocity, 
                                               self.prev_angular_velocity, 
                                               self.prev_steering_angle])), dim=0).unsqueeze(0)
            tensor2 = torch.cat((gap_msg, 
                                 torch.tensor([self.odom_msg.twist.twist.linear.x, 
                                               self.odom_msg.twist.twist.angular.z,
                                               0.0])), dim=0).unsqueeze(0)
              


        # Predict a priori next steering angle using learned prior model
        with torch.no_grad():
            pred_steer = self.prior_model(torch.cat((
                    max_gap_angle.unsqueeze(0) / self.angle_magnitude,
                    d_max[None, None], # For size [1,1]
                    torch.tensor([self.odom_msg.twist.twist.linear.x / 5.0, 
                                self.odom_msg.twist.twist.angular.z / 5.0]).unsqueeze(0)
                ), dim=1))
            
        pred_steer = torch.cat((
                        torch.tensor([self.prev_steering_angle]).unsqueeze(0),
                        pred_steer
                    ), dim=1).unsqueeze(-1)    

        # Creating new sample and adding to replay buffer (if in TRAINING mode)
        sample = torch.cat((tensor1.unsqueeze(1), tensor2.unsqueeze(1)), dim=1) # Shape: [1, 2, input_dim + 1]
        sample = torch.cat((sample, pred_steer), dim=-1)  # Append predicted steering angle sequence

        if self.train_mode: self.replay_buffer.append(sample)

        # Check if buffer size exceeds the purge threshold
        if len(self.replay_buffer) == self.max_buffer_size:
            self.get_logger().info(f"Purging {self.purge_threshold} oldest tensors from replay buffer...")
            for _ in range(self.purge_threshold):
                self.replay_buffer.popleft()

        # Defining context/target sets for NP model
        input_dim = self.gap_dim + self.vel_dim   # gap_dim + vel_dim (v,w)
        context_x = sample[:,:1,:input_dim].clone()
        context_y = sample[:,:1,input_dim:input_dim+1].clone()
        target_x = sample[:,:,:input_dim].clone()

        # Initializing query object for NP model
        query = ((context_x, context_y), target_x)

        # Predicting the next steering angle with uncertainty estimate
        if self.enable_ode_model:
            #pred_control = sample[:,:,input_dim+1:].clone()  # Max gap angle as control prior
            #print(f'Pred Control Shape: {pred_control.shape}')
            #pred_control = compute_control_prior(scan_msg.ranges, prev_steering_angle=self.prev_steering_angle)
            pred_y, var = self.model(query, pred_control=pred_steer, is_testing=True)
        else:
            pred_y, var = self.model(query, is_testing=True)

        # Training the model (only done after every 'self.batch_size' iterations to lower computation costs)
        if self.internal_counter % self.iters_before_optimize == 0 and self.internal_counter > 0 and self.train_mode:
            self.training_steps += 1
            self.get_logger().info('Training model...')
            self.model, loss, MAE, MAE_std, NLL = train_np_model(self.model, 
                                                                 self.replay_buffer, 
                                                                 self.optimizer, 
                                                                 self.loss_function, 
                                                                 self.batch_size,
                                                                 input_dim,
                                                                 self.enable_ode_model)

            self.get_logger().info(f"Weight Update Complete! Step: {self.training_steps}, Current loss: {loss.item()}, Current MAE: {MAE.item()}, Current MAE Std: {MAE_std.item()}")
            self.get_logger().info(f'Current NLL: {NLL.item()}, Steering Angle Error: {abs(pred_y[0,-1,0].item() - self.drive_label.drive.steering_angle)}, Steering Rate: {compute_steering_rate(self.prev_steering_angle, self.drive_label.drive.steering_angle)} \n')
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
        steer_command = torch.clamp(pred_y[0,-1, 0], -self.max_steer, self.max_steer)
        steer_command = steer_command.item()

        # Apply CBF filter on steering angle for safety (during INFERENCE)
        if not self.train_mode and self.prev_linear_velocity > 0.0:
            t2 = self.get_clock().now()

            steer_command = cbf_filter(
                ranges=ranges,
                raw_steer=steer_command,
                forward_velocity=self.prev_linear_velocity,
                max_steer=self.max_steer,
                d_safe=0.1,
                alpha=2.0,
                lf=0.15875,
                lr=0.17145,
                max_lidar_dist=self.max_lidar_dist
            )

            t3 = self.get_clock().now()
            if self.log_times: self.cbf_timer.push_time_pair(t2, t3)

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

        #print(f'True Steering: {self.drive_label.drive.steering_angle:.5f}, Predicted Steering: {pred_steer[:, -1, 0].item():.5f}')
        #print(f'Steering Command: {self.drive_label.drive.steering_angle:.5f}, Max Gap Angle: {max_gap_angle:.5f}')

        # Log steering rate
        self.delta_rate_array.append(compute_steering_rate(self.prev_steering_angle, pred_drive.drive.steering_angle))

        # Update previous predicted steering angle and gap info along with velocities
        self.prev_steering_angle = pred_drive.drive.steering_angle
        self.prev_gap = gap_msg
        self.prev_linear_velocity = self.odom_msg.twist.twist.linear.x
        self.prev_angular_velocity = self.odom_msg.twist.twist.angular.z
        self.prev_max_angle = max_gap_angle
        
        # Increment internal counter (for training purposes)
        if self.train_mode: self.internal_counter += 1

        # End loop timer and report in seconds
        t1 = self.get_clock().now()
        if self.log_times:
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

            if self.cbf_timer.should_report():
                stats = self.cbf_timer.stats()
                if stats is not None:
                    mean_dt, std_dt, min_dt, max_dt, hz, n = stats
                    self.get_logger().info(
                        f"[CBF FILTER] N={n} | "
                        f"mean={mean_dt*1000:.3f} ms ± {std_dt*1000:.3f} ms | "
                        f"min={min_dt*1000:.3f} ms | max={max_dt*1000:.3f} ms | "
                    )

def main(args=None):
    try:
        rclpy.init(args=args)

        # Now pass the model to the ROS2 node
        node = NPGapFollower()
        rclpy.spin(node)
        rclpy.shutdown()
    except KeyboardInterrupt:
        print("Keyboard interrupt received. Shutting down node.")
        print(f'Average steering rate during operation: {np.mean(node.delta_rate_array)} rad/s')
        if node.train_mode:
            # Save model after shutdown
            save_path = 'np_model_trained_weights.pth'
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
            np.save('mae_array.npy', np.array(node.mae_array))
            np.save('nll_array.npy', np.array(node.nll_array))
            print("MAE and NLL arrays saved as .npy files.")
    


if __name__ == '__main__':
    main()

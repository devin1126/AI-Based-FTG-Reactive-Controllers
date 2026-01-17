import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque

###   NLL Loss function for MLP model  ###
class MLPLoss(nn.Module):
    def __init__(self):
        super(MLPLoss, self).__init__()

    def forward(self, dist, target):
        """
        Computes the negative log-likelihood loss.

        Args:
            dist (torch.distributions.Normal): A Normal distribution object from model output.
            target (torch.Tensor): Ground truth steering signals.

        Returns:
            torch.Tensor: The NLL loss.
        """
        # Ensure both tensors are on the same device
        device = target.device
        target = target.to(device)

        # Negative log-likelihood loss
        nll = -dist.log_prob(target)

        # Mean over batch
        return torch.mean(nll)


###  Function that computes the Average Absolute Error (AAE) between target and predicted control signal(s)  ###
def compute_MAE(target_y, pred_y):

  # Computing Mean Absolute Error (MAE) and its standard deviation at final time step
  state_diff = torch.abs(target_y - pred_y)
  state_diff = torch.sum(state_diff, axis=1)
  MAE = torch.mean(state_diff)
  MAE_std = torch.std(state_diff)
  
  return MAE, MAE_std


###  Function that computes the Root Mean Squared Error (RMSE) between target and predicted odometry states  ###
def compute_RMSE(target_y, pred_y):
    """
    Compute the batch-averaged sum of RMSEs for each state at the final time step.
    
    Args:
        target_y (torch.Tensor): Shape (batch_size, time_steps, state_dim)
        pred_y (torch.Tensor): Shape (batch_size, time_steps, state_dim)
    
    Returns:
        float: Scalar RMSE value averaged over the batch.
    """
    # Compute squared error at the final time step
    state_diff_squared = (target_y - pred_y) ** 2  # Shape: (batch_size, state_dim)
    
    # RMSE for each state in each sample
    rmse_per_state = torch.sqrt(state_diff_squared)  # Shape: (batch_size, state_dim)

    # Sum RMSE across states for each sample
    summed_rmse_per_sample = torch.sum(rmse_per_state, axis=1)  # Shape: (batch_size,)

    # Compute average and standard deviation of RMSE across the batch
    batch_rmse = torch.mean(summed_rmse_per_sample)
    batch_rmse_std = torch.std(summed_rmse_per_sample)

    return batch_rmse, batch_rmse_std

###  Function that computes the Average Negative Log-Likelihood (NLL) between target and predicted odometry states  ###
def compute_avg_nll(target_y, pred_y, std):
    """
    Compute the Average Negative Log-Likelihood (NLL) for a batch of data.

    Args:
    - target_y (torch.Tensor): Ground truth states of shape [batch_size, num_states].
    - pred_y (torch.Tensor): Predicted means of shape [batch_size, num_states].
    - std (torch.Tensor): Predicted standard deviations of shape [batch_size, num_states].

    Returns:
    - avg_nll (torch.Tensor): Average Negative Log-Likelihood value.
    """
    # Ensure that variances are positive (avoid numerical issues with zero/negative variances)
    epsilon = 1e-8  # Small value to prevent division by zero
    var = torch.maximum(torch.pow(std, 2), torch.tensor(epsilon, device=std.device))  # Make sure variance is positive

    # Calculate the NLL for each state in the batch
    nll_per_state = 0.5 * torch.log(2 * torch.pi * var) + 0.5 * ((target_y - pred_y) ** 2) / var

    # Compute the average NLL over all samples and states
    avg_nll = torch.mean(nll_per_state)

    return avg_nll


###   Function that computes the marginal conformal quantiles for each state   ###
def compute_marginal_conformal_quantiles(target_y, pred_y, var, alpha=0.95, batch_divide=100):
    """
    Computes the conformal quantiles for the nonconformity scores of each state in the dynamical system.

    Args:
    - target_y (torch.Tensor): True target states of the system (batch_size, time_steps, num_states).
    - pred_y (torch.Tensor): Predicted states of the system (batch_size, time_steps, num_states).
    - var (torch.Tensor): Model variance for each state (batch_size, time_steps, num_states, num_states).
    - alpha (float): The quantile value to compute, default is 0.95.
    - batch_divide (int): Number of batches to divide the computation (default is 100).

    Returns:
    - quantiles (torch.Tensor): The computed quantiles for each state (num_states,).
    """
    num_states = target_y.shape[2]  # The number of states corresponds to the size of the last dimension
    nonconformity_scores_per_state = torch.zeros((num_states, target_y.shape[0]))  # (num_states, batch_size)

    # Compute nonconformity scores for each state individually
    for i in range(num_states):
        # Extract the data for the current state (index 2 corresponds to states)
        target_state = target_y[:, -1, i]  # Shape: (batch_size,)
        pred_state = pred_y[:, -1, i]      # Shape: (batch_size,)
        var_state = var[:, -1, i]          # Shape: (batch_size,) (the variance of state i)

        # Compute Mahalanobis distance for each state
        batch_len = int(var_state.size(0) / batch_divide) if var_state.size(0) >= batch_divide else var_state.size(0)
        nonconformity_scores = torch.zeros(var_state.shape[0])

        for j in range(0, var_state.shape[0], batch_len):

            # Slice batch data
            cov_batch = torch.diag_embed(var_state[j:j+batch_len])  # Covariance matrix for each batch
            x_true_batch = target_state[j:j+batch_len]  # True target values
            x_pred_batch = pred_state[j:j+batch_len]    # Predicted values

            # Compute Mahalanobis distance for the batch
            diff = x_true_batch - x_pred_batch  # Shape: (batch_len,)
            diff = diff.unsqueeze(-1)  # Shape: (batch_len, 1)
            inv_covs = torch.linalg.inv(cov_batch)  # Shape: (batch_len, batch_len)

            # Mahalanobis distance calculation using matrix multiplication
            mahalanobis_dist = torch.sqrt(torch.sum(diff * torch.matmul(inv_covs, diff), dim=1))  # Shape: (batch_len,)

            # Store nonconformity scores in the array for this state
            nonconformity_scores[j:j+batch_len] = mahalanobis_dist

        # Store nonconformity scores for the current state
        nonconformity_scores_per_state[i, :] = nonconformity_scores

    # Compute quantiles for each state separately
    quantiles = torch.quantile(nonconformity_scores_per_state, alpha, dim=1)  # Shape: (num_states,)

    return quantiles


# -----------------------------------------------------------
# Bin distance readings into 'num_bins' equal angular regions
# -----------------------------------------------------------
def bin_laser_scans(proc_ranges, num_bins=32, max_range=6.0):
    """
    Condenses raw LiDAR readings into N averaged distance bins and returns
    the bin with the largest distance along with its FOV angle.

    Angle convention:
        - Leftmost bin = +135 degrees (≈ +2.356 rad)
        - Center bin  = 0 degrees
        - Rightmost bin = -135 degrees (≈ -2.356 rad)

    Args:
        proc_ranges (torch.Tensor): Raw LiDAR distance readings (1D tensor)
        num_bins (int): Number of bins to condense into
        max_range (float): Max range to clip lidar distances

    Returns:
        Tuple[torch.Tensor, int, float]: 
            - Condensed distance array (shape = [num_bins])
            - Index of bin with maximum distance
            - Angle (radians) corresponding to that bin center
    """
    # Clip distances
    laser_len = proc_ranges.shape[0]
    segment_size = laser_len // num_bins

    binned = torch.zeros(num_bins)
    for i in range(num_bins):
        start = i * segment_size
        end = (i + 1) * segment_size if i < num_bins - 1 else laser_len
        binned[i] = torch.mean(proc_ranges[start:end])

    # Normalize
    binned /= max_range

    return binned


###   Function that computes steering rate   ###
def compute_steering_rate(prev_steer, current_steer, dt=1.0/200.0):
    """
    Compute steering rate assuming fixed control loop timing.

    Args:
        prev_steer (float): previous steering command [rad]
        current_steer (float): current steering command [rad]
        dt (float): fixed timestep [s]

    Returns:
        float or None: steering rate [rad/s]
    """
    steer_rate = (current_steer - prev_steer) / dt

    return steer_rate


###   Loop timing utility class using ROS time   ###
class LoopTimerROS:
    """
    Rolling window timer using ROS Clock timestamps (rclpy.time.Time).
    Stores dt in seconds as float.
    """
    def __init__(self, node, window: int = 200, report_period: float = 1.0):
        self.node = node
        self.dt_buf = deque(maxlen=window)

        self.report_period = float(report_period)
        self.last_report_t = node.get_clock().now()

    def push_dt(self, dt_sec: float):
        self.dt_buf.append(float(dt_sec))

    def push_time_pair(self, t_start, t_end):
        """
        t_start, t_end are rclpy.time.Time objects.
        """
        dt = (t_end - t_start).nanoseconds * 1e-9
        self.push_dt(dt)

    def stats(self):
        if len(self.dt_buf) == 0:
            return None

        arr = np.asarray(self.dt_buf, dtype=np.float64)
        mean_dt = float(arr.mean())
        std_dt  = float(arr.std(ddof=0))
        min_dt  = float(arr.min())
        max_dt  = float(arr.max())
        hz = float(1.0 / mean_dt) if mean_dt > 0 else 0.0

        return mean_dt, std_dt, min_dt, max_dt, hz, len(arr)

    def should_report(self):
        now = self.node.get_clock().now()
        dt_report = (now - self.last_report_t).nanoseconds * 1e-9
        if dt_report >= self.report_period:
            self.last_report_t = now
            return True
        return False



###   NP model training function   ###
def train_mlp_model(model, replay_buffer, optimizer, loss_function, batch_size, input_dim):
    # Setting the model to training mode
    model.train()

    # Randomly sample a batch of data from the buffer
    batch = [replay_buffer[i] for i in torch.randint(0, len(replay_buffer), (batch_size,))]

    # Converting batch to tensor and normalizing 
    batch = torch.stack([item for item in batch]).squeeze(1)  # Extract batch dataset

    # Defining context/target sets for the NP model
    input_data = batch[:,:input_dim].clone()
    target_y = batch[:,input_dim:].clone()

    # Predicting the next odometry state and capturing loss info
    dist, pred_y, std = model(input_data, is_testing=False)

    # Calculate the loss and providing recurrent performance feedback
    loss = loss_function(dist, target_y)

    # Backpropagate and optimize
    optimizer.zero_grad()  # Clear gradients from previous step
    loss.backward()  # Compute gradients
    optimizer.step()  # Update the model weights

    # Compute batch root mean squared error (RMSE) and negative log-likelihood (NLL)
    MAE, MAE_std = compute_MAE(target_y, pred_y)
    NLL = compute_avg_nll(target_y, pred_y, std)

    return model, loss, MAE, MAE_std, NLL
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from collections import deque
#from cvxopt import matrix, solvers

###   Minmax normalization function   ###
'''
def minmax_norm(dataset, gap_dim=128, odom_dim=2):

    minmax_data = torch.tensor([[0,5],
                                [-5,5]])
    minmax_idx = 0

    # Parsing relevant data shapes
    dynamical_data_dim = dataset.shape[2]
    prev_dim = 0
    next_dim = 1
    prev_dim_skips = list(range(gap_dim)) + [gap_dim+odom_dim]
    next_dim_skips = list(range(dynamical_data_dim))  # all skips for next state (no normalization required)

    for dim in range(dynamical_data_dim):
        # Skipping sampling rate dimension to avoid normalization
        if dim in prev_dim_skips:
            continue
        
        curr_sequence = dataset[:,prev_dim,dim:dim+1]
        curr_min = minmax_data[minmax_idx,0]
        curr_max = minmax_data[minmax_idx,1]

        dataset[:,prev_dim,dim:dim+1] = (curr_sequence - curr_min) / (curr_max - curr_min)
        minmax_idx += 1
        
        
    for dim in range(dynamical_data_dim):
        # Skipping sampling rate dimension to avoid normalization
        if dim in next_dim_skips:
            continue

        curr_sequence = dataset[:,next_dim,dim:dim+1]
        curr_min = minmax_data[dim,0]
        curr_max = minmax_data[dim,1]

        dataset[:,next_dim,dim:dim+1] = (curr_sequence - curr_min) / (curr_max - curr_min)
      
    return dataset
'''

###   ELBO Loss function for NP models  ###
class ELBO(nn.Module):
  def __init__(self):
    super(ELBO, self).__init__()

  def forward(self, log_p, kl):
    # Get the device of log_p (assuming both log_p and kl should be on the same device)
    device = log_p.device
    
    # Ensure that both tensors are on the same device
    log_p = log_p.to(device)
    kl = kl.to(device)
    
    # Computing total loss
    loss = -torch.mean((log_p - kl))
    
    return loss


###  Function that computes the Average Absolute Error (AAE) between target and predicted control signal(s)  ###
def compute_MAE(target_y, pred_y):

  # Computing Mean Absolute Error (MAE) and its standard deviation at final time step
  state_diff = torch.abs(target_y[:,-1,:] - pred_y[:,-1,:])
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
    state_diff_squared = (target_y[:, -1, :] - pred_y[:, -1, :]) ** 2  # Shape: (batch_size, state_dim)
    
    # RMSE for each state in each sample
    rmse_per_state = torch.sqrt(state_diff_squared)  # Shape: (batch_size, state_dim)

    # Sum RMSE across states for each sample
    summed_rmse_per_sample = torch.sum(rmse_per_state, axis=1)  # Shape: (batch_size,)

    # Compute average and standard deviation of RMSE across the batch
    batch_rmse = torch.mean(summed_rmse_per_sample)
    batch_rmse_std = torch.std(summed_rmse_per_sample)

    return batch_rmse, batch_rmse_std

###  Function that computes the Average Negative Log-Likelihood (NLL) between target and predicted odometry states  ###
def compute_avg_nll(target_y, pred_y, var):
    """
    Compute the Average Negative Log-Likelihood (NLL) for a batch of data.

    Args:
    - target_y (torch.Tensor): Ground truth states of shape [batch_size, num_states].
    - pred_y (torch.Tensor): Predicted means of shape [batch_size, num_states].
    - var (torch.Tensor): Predicted variances of shape [batch_size, num_states].

    Returns:
    - avg_nll (torch.Tensor): Average Negative Log-Likelihood value.
    """
    # Ensure that variances are positive (avoid numerical issues with zero/negative variances)
    epsilon = 1e-8  # Small value to prevent division by zero
    var = torch.maximum(var, torch.tensor(epsilon, device=var.device))  # Make sure variance is positive

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


def compute_control_prior(
    lidar_ranges,
    prev_steering_angle,
    forward_crop=135,
    preprocess_conv_size=3,
    max_lidar_dist=6.0,
    bubble_radius=30,
    safe_threshold=15,
    best_point_conv_size=400,
    max_steer=0.6981            # 40 deg 
):
    """
    Full Follow-The-Gap control prior using ONLY the raw lidar scan
    (correct teacher signal for NP steering training).

    lidar_ranges: torch.Tensor [1080] OR numpy array
    Returns:
        speed, steering_angle  (torch.tensor scalars)
    """
    # ----------------------------------------
    # 0. Convert to numpy
    # ----------------------------------------
    if isinstance(lidar_ranges, torch.Tensor):
        lidar_ranges = lidar_ranges.detach().cpu().numpy()

    # ----------------------------------------
    # 1. Replace NaN/inf and clip
    # ----------------------------------------
    ranges = np.nan_to_num(
        lidar_ranges,
        nan=0.0,
        posinf=max_lidar_dist,
        neginf=0.0
    )
    ranges = np.clip(ranges, 0.0, max_lidar_dist)

    # ----------------------------------------
    # 2. Crop forward-facing region
    #    1080 → ~810 points
    # ----------------------------------------
    ranges = ranges[forward_crop:-forward_crop]

    # ----------------------------------------
    # 3. Moving average smoothing
    # ----------------------------------------
    kernel = np.ones(preprocess_conv_size) / preprocess_conv_size
    ranges = np.convolve(ranges, kernel, mode='same')
    ranges = np.clip(ranges, 0.0, max_lidar_dist)

    # ----------------------------------------
    # 4. Find closest obstacle
    # ----------------------------------------
    closest = ranges.argmin()

    # ----------------------------------------
    # 5. Create the "bubble" (zero out close obstacles)
    # ----------------------------------------
    min_i = max(0, closest - bubble_radius)
    max_i = min(len(ranges) - 1, closest + bubble_radius)
    ranges[min_i:max_i] = 0.0

    # ----------------------------------------
    # 6. Identify max free-space gap
    # ----------------------------------------
    masked = np.ma.masked_where(ranges == 0, ranges)
    slices = np.ma.notmasked_contiguous(masked)

    if not slices:
        # fallback → drive straight
        best_idx = len(ranges) // 2

    else:
        # Choose longest gap
        best_slice = max(slices, key=lambda s: s.stop - s.start)
        start, end = best_slice.start, best_slice.stop

        gap_len = end - start

        # ----------------------------------------
        # 7. Select best point in gap
        # ----------------------------------------
        if gap_len < best_point_conv_size:
            # if gap too small, use center
            best_idx = start + gap_len // 2
        else:
            # sliding-window averaging
            window = np.ones(best_point_conv_size) / best_point_conv_size
            smoothed = np.convolve(ranges[start:end], window, mode='same')
            best_idx = start + smoothed.argmax()

    # ----------------------------------------
    # 8. Convert best index → steering angle
    # ----------------------------------------
    radians_per_elem = (2 * np.pi) / 1080  # full lidar resolution
    lidar_angle = (best_idx - (len(ranges) / 2)) * radians_per_elem
    steering_angle = np.clip(lidar_angle / 2.0, -max_steer, max_steer)

    # ----------------------------------------
    # 9. Apply small noise to steering angle
    # ----------------------------------------
    steering_angle += np.random.normal(0.0, 0.005)  # small Gaussian noise
    steering_angle = np.clip(steering_angle, -max_steer, max_steer) 

    # ----------------------------------------
    # 10. Return tensor (for NP integration)
    # ----------------------------------------
    return torch.tensor([prev_steering_angle, steering_angle], dtype=torch.float32)[None,:,None]


###    CBF-based steering filter function    ###
def cbf_filter(
    ranges: np.ndarray,
    raw_steer: float,
    forward_velocity: float,
    max_steer: float = 0.6981,
    d_safe: float = 0.2,
    alpha: float = 1.5,
    lf: float = 0.15875,
    lr: float = 0.17145,
    max_lidar_dist: float = 6.0,
    side_offset: float = 0.16,
    front_offset: float = 0.055,
    eps: float = 1e-2
) -> float:
    """
    Sensor-based CBF steering filter using full LiDAR scan (1080 beams, fov=4.7 rad).

    Approx body-clearance model (piecewise offset):
      - For side-ish beams near +/- 90 deg (e.g., |phi| in [90,135] deg): subtract ~0.16 m (half-width).
      - Otherwise (more forward-ish): subtract 0.055 m (LiDAR to front bumper).

    Hard min selection:
      - Take the minimum LiDAR range beam and its corresponding angle.
      - Convert to approximate body clearance d_phi = d_laser - body_offset(phi).
      - Use h = d_phi - d_safe in the CBF constraint.

    Sign fix:
      - Flip sign of Lgh so that (typically) +phi pushes steer negative and -phi pushes steer positive.
    """
    # -----------------------------
    # Geometry
    # -----------------------------
    L = lf + lr
    v = forward_velocity

    num_beams = len(ranges)
    fov_rad = 4.7
    angles = np.linspace(+fov_rad / 2.0, -fov_rad / 2.0, num_beams)

    # -----------------------------
    # Preprocess LiDAR
    # -----------------------------
    clipped_ranges = np.clip(ranges, 0.0, max_lidar_dist)

    # -----------------------------
    # Hard minimum selection
    # -----------------------------
    min_idx = int(np.argmin(clipped_ranges))
    d_laser = float(clipped_ranges[min_idx])
    phi = float(angles[min_idx])
    phi_deg = float(np.rad2deg(phi))

    # -----------------------------
    # Approximate body offset (piecewise)
    # -----------------------------
    # Side-ish region: |phi| in [90, 135] deg -> subtract ~half-width
    # Else: treat as forward-ish -> subtract lidar-to-front
    abs_phi = abs(phi_deg)
    if 90.0 <= abs_phi <= 135.0:
        body_offset = side_offset
        offset_region = "SIDE"
    else:
        body_offset = front_offset
        offset_region = "FRONT"

    # Clearance from robot body along that beam
    d_phi = d_laser - body_offset
    d_phi = max(d_phi, 0.0)  # don’t allow negative “clearance” to blow up behavior

    h = d_phi - d_safe

    # -----------------------------
    # Lie derivatives (Ackermann, sign-fixed)
    # -----------------------------
    Lfh = -v * np.cos(phi)

    # Use body clearance d_phi (not raw d_laser)
    # Sign flip to encourage "steer away" consistency
    Lgh = (v / L) * d_phi * np.sin(phi)

    cbf_rhs = -Lfh - alpha * h
    #print(f"CBF RHS: {cbf_rhs:.5f}, Lfh: {Lfh:.5f}, Lgh: {Lgh:.5f}, alpha*h: {alpha*h:.5f}")

    # -----------------------------
    # Active-set 1D QP solve
    # -----------------------------
    delta_nom = float(np.clip(raw_steer, -max_steer, max_steer))

    if Lfh + Lgh * delta_nom + alpha * h >= 0.0:
       # print(f"Nominal steering feasible: {delta_nom:.5f}\n")
        return delta_nom

    if abs(Lgh) < eps:
        #print("Ill-conditioned CBF constraint, using nominal steering\n")
        return delta_nom

    delta_cbf = cbf_rhs / Lgh
    delta_cbf = float(np.clip(delta_cbf, -max_steer, max_steer))
    ''' 
    print(
        f"d_laser: {d_laser:.5f}, phi (deg): {phi_deg:.2f}, "
        f"offset_region: {offset_region}, body_offset: {body_offset:.3f}, "
        f"d_phi: {d_phi:.5f}, h: {h:.5f}, v: {v:.5f}"
    )
    print(f"CBF RHS: {cbf_rhs:.5f}, Lfh: {Lfh:.5f}, Lgh: {Lgh:.5f}, alpha*h: {alpha*h:.5f}")
    print(f"CBF-corrected steering: {delta_cbf:.5f}, Original steering: {delta_nom:.5f}\n")
    '''
    return delta_cbf



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

    # Find max bin/distance and compute its angle
    max_idx = torch.argmax(binned).item()
    d_max = binned[max_idx]

    # Angle calculations (from +135 to -135 degrees)
    fov_rad = np.deg2rad(270)  # 270 degrees
    angle_step = fov_rad / num_bins
    angle_left = np.deg2rad(135)
    angle = angle_left - (max_idx + 0.5) * angle_step  # center of bin

    return binned, torch.tensor([-angle], dtype=torch.float32), d_max

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


###   Loop timing utility class   ###
class LoopTimer:
    def __init__(self, window: int = 200, report_period: float = 1.0):
        """
        window: number of recent loop times to keep (batch size N)
        report_period: how often to print stats (seconds)
        """
        self.dt_buf = deque(maxlen=window)
        self.report_period = float(report_period)
        self.last_report_t = time.perf_counter()

    def push(self, dt: float):
        self.dt_buf.append(float(dt))

    def stats(self):
        if len(self.dt_buf) == 0:
            return None

        arr = np.asarray(self.dt_buf, dtype=np.float64)

        mean_dt = float(arr.mean())
        std_dt  = float(arr.std(ddof=0))   # population std
        min_dt  = float(arr.min())
        max_dt  = float(arr.max())

        hz = float(1.0 / mean_dt) if mean_dt > 0 else 0.0

        return mean_dt, std_dt, min_dt, max_dt, hz, len(arr)

    def should_report(self):
        now = time.perf_counter()
        if (now - self.last_report_t) >= self.report_period:
            self.last_report_t = now
            return True
        return False

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
def train_np_model(model, replay_buffer, optimizer, loss_function, batch_size, input_dim, enable_np_model=True):
    # Setting the model to training mode
    model.train()

    # Randomly sample a batch of data from the buffer
    batch = [replay_buffer[i] for i in torch.randint(0, len(replay_buffer), (batch_size,))]

    # Converting batch to tensor and normalizing 
    batch = torch.stack([item for item in batch]).squeeze(1)  # Extract main dataset

    # Defining context/target sets for the NP model
    context_x = batch[:,:1,:input_dim].clone()
    context_y = batch[:,:1,input_dim:input_dim+1].clone()
    target_x = batch[:,:,:input_dim].clone()
    target_y = batch[:,:,input_dim:input_dim+1].clone()

    # Initializing query object for model
    query = ((context_x, context_y), target_x)

    # Predicting the next odometry state and capturing loss info
    if enable_np_model:
        pred_control = batch[:,:,input_dim+1:].clone()  # Max gap angle as control prior
        log_p, kl, pred_y, var = model(query, pred_control=pred_control, target_y=target_y, is_testing=False)
    else:
        log_p, kl, pred_y, var = model(query, target_y=target_y, is_testing=True)

    # Calculate the loss and providing recurrent performance feedback
    loss = loss_function(log_p, kl)

    # Backpropagate and optimize
    optimizer.zero_grad()  # Clear gradients from previous step
    loss.backward()  # Compute gradients
    optimizer.step()  # Update the model weights

    # Compute batch root mean squared error (RMSE) and negative log-likelihood (NLL)
    MAE, MAE_std = compute_MAE(target_y, pred_y)
    q_marginal = compute_marginal_conformal_quantiles(target_y,pred_y,var)
    var[:,-1,:] *= q_marginal
    NLL = compute_avg_nll(target_y[:,-1,:], pred_y[:,-1,:], var[:,-1,:])

    return model, loss, MAE, MAE_std, NLL
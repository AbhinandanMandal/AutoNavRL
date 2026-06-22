import gym
from gym import spaces
import numpy as np
from real_env import REAL_ENV


class RobotNavEnv(gym.Env):
    """
    Custom Gym environment that wraps the REAL_ENV simulator for robot navigation.

    This environment:
    - Converts the simulator's outputs into a fixed-size observation space
    - Defines a continuous action space for linear and angular velocities
    - Handles state normalization and preprocessing
    - Manages episode termination conditions

    Attributes:
        action_space (gym.spaces.Box): Continuous action space for linear and angular velocities
        observation_space (gym.spaces.Box): Fixed-size observation space
        state_dim (int): Dimension of the state vector
        sim (REAL_ENV): Instance of the real environment simulator
        time (int): Step counter for episode termination


    Observation space (51-D)
    ------------------------
    [0:42]  Binned min-value LiDAR (42 bins, normalised to [0, 1])
    [42]    Distance to current waypoint / 10
    [43]    cos(angle: robot heading → waypoint)
    [44]    sin(angle: robot heading → waypoint)
    [45]    Normalised linear velocity  ((v + 0.6) / 1.2)
    [46]    Normalised angular velocity ((w + 1.2) / 2.4)
    [47]    cos(diff_rad)   heading vs final-goal orientation
    [48]    sin(diff_rad)
    [49]    Waypoint progress  (waypoints completed / total)
    [50]    cos(waypoint direction, final-goal direction)

    """

    def __init__(self, use_cv:bool=True,cv_use_midas:bool=True ):
        super(RobotNavEnv, self).__init__()
        # Action space: [linear_velocity, angular_velocity]
        # Linear velocity range: [-0.6, 0.6] m/s
        # Angular velocity range: [-1.2, 1.2] rad/s
        self.action_space = spaces.Box(
            low=np.array([-0.6, -1.2]),
            high=np.array([0.6, 1.2]),
            dtype=np.float32
        )

        # Observation space: 49-dimensional vector containing:
        # - Binned LIDAR scan data (42 dimensions)
        # - Distance to goal (1 dimension)
        # - Goal direction cos/sin (2 dimensions)
        # - Current linear/angular velocities (2 dimensions)
        # - Goal angle difference cos/sin (2 dimensions)
        self.state_dim = 51
        self.observation_space = spaces.Box(
            low=-1,
            high=1,
            shape=(self.state_dim,),
            dtype=np.float32
        )

        # Initialize simulator with default goal
        self.goal = [0.0, 0.0, 0.0]
        self.sim = REAL_ENV(goal_pose=self.goal, use_cv=use_cv, cv_use_midas=cv_use_midas)
        self.time = 0

    # waypoint features
    def _waypoint_features(self):
        """
        Compute the two extra waypoint-progress observation features.

        Returns
        -------
        waypoint_progress : float  [0, 1]
        wp_cos            : float  [-1, 1]
        """
        n = len(self.sim.waypoints)
        if n == 0:
            return 0.0, 1.0

        idx = self.sim.current_waypoint_index
        waypoint_progress = idx / max(n - 1, 1)

        rx, ry = self.sim.robot_pose[0], self.sim.robot_pose[1]
        tx, ty = self.sim.current_target
        gx, gy = self.sim.robot_goal[0], self.sim.robot_goal[1]

        v_wp = np.array([tx - rx, ty - ry])
        v_goal = np.array([gx - rx, gy - ry])

        norm_wp = np.linalg.norm(v_wp) + 1e-9
        norm_goal = np.linalg.norm(v_goal) + 1e-9
        wp_cos = float(np.dot(v_wp / norm_wp, v_goal / norm_goal))

        return waypoint_progress, wp_cos

    # state preparation
    def prepare_state(self, data):
        """
        Process raw environment data into a normalized state vector.

        Args:
            data (tuple): Raw environment data containing:
                - LIDAR scan data
                - Distance to goal
                - Goal direction cos/sin
                - Collision flag
                - Goal reached flag
                - Angle difference
                - Last action
                - Reward

        Returns:
            tuple: (normalized_state, terminal_flag)
        """
        latest_scan, distance, cos, sin, collision, goal, diff_rad, action, reward = data
        latest_scan = np.array(latest_scan)

        # Handle infinite values in LIDAR data
        # Clip infinite values
        latest_scan[np.isinf(latest_scan)] = 10.0
        latest_scan = np.clip(latest_scan, 0.0, 10.0)

        # Bin LIDAR data to reduce dimensionality
        max_bins = self.state_dim - 9

        # Extend latest_scan to make its length divisible by max_bins
        remainder = len(latest_scan) % max_bins
        if remainder != 0:
            # Calculate the number of elements to add
            elements_to_add = max_bins - remainder
            # Create an array of the last element repeated elements_to_add times
            extension = np.full(elements_to_add, latest_scan[-1])
            # Concatenate the original array with the extension
            latest_scan = np.concatenate((latest_scan, extension))

        bin_size = int(len(latest_scan) / max_bins)
        min_values = []
        for i in range(0, len(latest_scan), bin_size):
            bin = latest_scan[i: i + min(bin_size, len(latest_scan) - i)]
            # Find the minimum value in the current bin and append it to the min_values list
            min_values.append(min(bin) / 10.0)

        # Normalize distance and velocities
        distance_norm = distance / 10.0
        lin_vel = (action[0] + 0.6) / 1.2
        ang_vel = (action[1] + 1.2) / 2.4

        # Convert angle difference to cos/sin representation
        rad_cos = float(np.cos(diff_rad))
        rad_sin = float(np.sin(diff_rad))

        # waypoint progress features
        wp_progress, wp_cos = self._waypoint_features()

        # Combine all state components
        state = min_values + [distance_norm, cos, sin] + \
            [lin_vel, ang_vel] + [rad_cos, rad_sin] + [wp_progress, wp_cos]
        assert len(state) == self.state_dim, (
            f"State length mismatch: expected {self.state_dim}, got {len(state)}"
        )

        terminal = 1 if collision or goal else 0
        return state, terminal

    # gym interface
    def reset(self, goal):
        """
        Reset the environment with a new goal.

        Args:
            goal (list): [x, y, yaw] target pose

        Returns:
            numpy.ndarray: Initial observation
        """
        self.goal = goal
        sim_data = self.sim.reset(goal_pose=self.goal)
        obs, _ = self.prepare_state(sim_data)
        self.current_obs = np.array(obs, dtype=np.float32)
        self.time = 0
        return self.current_obs

    def step(self, action):
        """
        Execute one step in the environment.

        Args:
            action (numpy.ndarray): [linear_velocity, angular_velocity]

        Returns:
            tuple: (observation, reward, done, info)
        """
        lin_velocity, ang_velocity = float(action[0]), float(action[1])

        # Execute action in simulator
        sim_data = self.sim.step(
            lin_velocity=lin_velocity, ang_velocity=ang_velocity)
        obs, terminal = self.prepare_state(sim_data)
        reward = float(sim_data[-1])

        # Check termination conditions
        done = bool(terminal)
        self.time += 1
        if self.time >= 5000:  # Time limit
            done = True
            reward = 0.0
            print("Episode terminated due to time limit")

        info = {
            "waypoints_total":   len(self.sim.waypoints),
            "waypoints_reached": self.sim.current_waypoint_index,
            "cv_active":         self.sim._cv is not None,
            "cv_calibrated":     self.sim._cv_calibrated,
        }
        self.current_obs = np.array(obs, dtype=np.float32)
        return self.current_obs, reward, done, info

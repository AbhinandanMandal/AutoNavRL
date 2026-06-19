import gym
from gym import spaces
import numpy as np
import argparse

from sim import SIM_ENV
from stable_baselines3 import TD3
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy


class RobotNavEnv(gym.Env):
    """
    Custom Gym environment that wraps the SIM_ENV simulator.

    This environment converts the simulator's outputs into a fixed-size observation,
    defines an action space, and scales actions as required for reinforcement learning.


    Observation vector changes to 51-D
    [0:42]   Downsampled min-binned lidar scan (42 bins, normalised 0 -> 1)
    [42]     Distance to current waypoint (normalised / 10)
    [43]     cos(angle between robot heading and waypoint direction)
    [44]     sin(angle between robot heading and waypoint direction)
    [45]     Normalised linear velocity
    [46]     Normalised angular velocity
    [47]     cos(diff_rad)  — heading vs final goal orientation
    [48]     sin(diff_rad)
    [49]     Waypoint progress  (completed / total)
    [50]     cos(angle from waypoint direction to final-goal direction)
    """

    def __init__(self, render=False):
        """
        Initialize the robot navigation environment.

        Args:
            render (bool): Whether to enable visualization
        """
        super(RobotNavEnv, self).__init__()

        # Environment configuration
        self.render = render
        self.state_dim = 51  # Dimension of the observation space
        self.max_steps = 150  # Maximum number of steps per episode

        # Define action space (linear and angular velocity)
        self.action_space = spaces.Box(
            low=np.array([-0.6, -1.2]),  # [min_linear_vel, min_angular_vel]
            high=np.array([0.6, 1.2]),   # [max_linear_vel, max_angular_vel]
            dtype=np.float32
        )

        # Define observation space (normalized to [-1, 1])
        self.observation_space = spaces.Box(
            low=-1,
            high=1,
            shape=(self.state_dim,),
            dtype=np.float32
        )

        # Initialize simulator
        self.sim = SIM_ENV(render=render)

        # Initialize episode tracking
        self._reset_episode_tracking()

        # Get initial observation
        initial_data = self.sim.reset()
        self.current_obs, _ = self.prepare_state(initial_data)

    # Episode tracking of RL policy
    def _reset_episode_tracking(self):
        """Reset all episode tracking variables."""
        self.time = 0
        self.last_position = None
        self.total_distance = 0
        self.total_velocity = 0
        self.steps = 0

    def _calculate_metrics(self, current_position, action):
        """
        Calculate and update episode metrics.

        Args:
            current_position: Current robot position [x, y]
            action: Current action [linear_vel, angular_vel]
        """
        if self.last_position is not None:
            step_distance = np.linalg.norm(
                current_position - self.last_position)
            self.total_distance += step_distance
            self.total_velocity += np.linalg.norm(action)
        self.last_position = current_position
        self.steps += 1

    def _get_episode_info(self, terminal, reward):
        """
        Generate episode information dictionary.

        Args:
            terminal (bool): Whether episode is terminal
            reward (float): Final reward

        Returns:
            dict: Episode information
        """
        avg_velocity = self.total_velocity / self.steps if self.steps > 0 else 0
        return {
            'success': terminal and reward > 0,
            'collision': terminal and reward < 0,
            'steps': self.steps,
            'total_distance': self.total_distance,
            'average_velocity': avg_velocity,
            'time_limit_reached': self.time >= self.max_steps
        }

    # Waypoint features

    def _waypoint_features(self):
        """
        Compute the two extra waypoint-progress features.

        Returns
        -------
        waypoint_progress : float in [0, 1] = Fraction of waypoints already completed.
        wp_cos : float in [-1, 1]
            Cosine similarity between (robot -> current_waypoint) and
            (robot -> final_goal) direction vectors.  Tells the policy whether
            the current waypoint is roughly on the way to the goal.
        """

        n = len(self.sim.waypoints)
        if n == 0:
            return 0.0, 1.0

        idx = self.sim.current_waypoint_index
        waypoint_progress = idx / max(n - 1, 1)

        # Direction vectors in world frame
        robot_state = self.sim.env.get_robot_state()
        rx, ry = robot_state[0].item(), robot_state[1].item()

        tx, ty = self.sim.current_target
        gx = self.sim.robot_goal[0].item()
        gy = self.sim.robot_goal[1].item()

        v_wp = np.array([tx - rx, ty - ry])
        v_goal = np.array([gx - rx, gy - ry])

        norm_wp = np.linalg.norm(v_wp) + 1e-9
        norm_goal = np.linalg.norm(v_goal) + 1e-9

        wp_cos = float(np.dot(v_wp / norm_wp, v_goal / norm_goal))
        return waypoint_progress, wp_cos

    def prepare_state(self, data):
        """
        Process raw simulator data into a normalized 51-D observation vector.

        Args:
            data: Raw simulator data tuple

        Returns:
            tuple: (processed_state, terminal_flag)
        """
        latest_scan, distance, cos, sin, collision, goal, diff_rad, action, reward = data
        latest_scan = np.array(latest_scan)

        # Handle infinite values in laser scan by clipping
        inf_mask = np.isinf(latest_scan)
        latest_scan[inf_mask] = 10.0

        # Downsample laser scan data
        max_bins = self.state_dim - 9  # 51 - 9 = 42

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

        # Create bins and get minimum values
        for i in range(0, len(latest_scan), bin_size):
            # bin = latest_scan[i : i + min(bin_size, len(latest_scan) - i)]
            bin = latest_scan[i: i + bin_size]
            # Find the minimum value in the current bin and append it to the min_values list
            min_values.append(float(min(bin)) / 10.0)

        # Normalize values to [0, 1] range
        distance_norm = distance / 10.0
        lin_vel = (action[0] + 0.6) / 1.2
        ang_vel = (action[1] + 1.2) / 2.4

        # Convert angle difference to cos/sin representation
        rad_cos = float(np.cos(diff_rad))
        rad_sin = float(np.sin(diff_rad))

        # Waypoint progress features
        wp_progress, wp_cos = self._waypoint_features()

        # Combine all features into state vector
        state = min_values + [distance_norm, cos, sin] + [lin_vel,
                                                          ang_vel] + [rad_cos, rad_sin] + [wp_progress, wp_cos]

        assert len(state) == self.state_dim, (
            f"State dimension mismatch: expected {self.state_dim}, got {len(state)}"
        )
        terminal = 1 if collision or goal else 0
        return state, terminal

    # Gym interface
    def reset(self):
        """
        Reset the environment and return initial observation.

        Returns:
            numpy.ndarray: Initial observation
        """
        sim_data = self.sim.reset()
        obs, _ = self.prepare_state(sim_data)
        self.current_obs = obs
        self._reset_episode_tracking()
        return obs

    def step(self, action):
        """
        Execute one time step within the environment.

        Args:
            action: [linear_velocity, angular_velocity]

        Returns:
            tuple: (observation, reward, done, info)
        """
        # Process actions with deadzone
        lin_velocity = 0 if abs(action[0]) < 0.15 else action[0]
        ang_velocity = 0 if abs(action[1]) < 0.15 else action[1]

        # Step simulation
        sim_data = self.sim.step(
            lin_velocity=lin_velocity, ang_velocity=ang_velocity)
        obs, terminal = self.prepare_state(sim_data)
        reward = sim_data[-1]

        # Update metrics
        current_position = self.sim.env.get_robot_state()[:2]
        self._calculate_metrics(current_position, action)

        # Check termination conditions
        done = bool(terminal)
        self.time += 1
        if self.time >= self.max_steps:
            done = True
            reward = -100

        # Generate info dictionary
        info = self._get_episode_info(terminal, reward)
        self.current_obs = obs
        return obs, reward, done, info

# Vectorized environment factory


def make_env(render=False):
    """
    Utility function for creating new instances of RobotNavEnv.
    This is used to create multiple parallel environments.

    Args:
        render (bool): Whether to enable visualization

    Returns:
        function: Environment initialization function
    """
    def _init():
        env = RobotNavEnv(render)
        return env
    return _init


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train TD3 model for robot navigation")
    parser.add_argument("--num-envs", type=int, default=7)
    parser.add_argument("--total-timesteps", type=int, default=200_000)
    parser.add_argument("--model-path", type=str,
                        default="models/td3_robot_nav_model")
    parser.add_argument("--tensorboard-log", type=str,
                        default="./td3_robot_nav_tensorboard/")
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    env_fns = [make_env() for _ in range(args.num_envs)]
    env_fns.append(make_env(render=args.render))
    env = SubprocVecEnv(env_fns)

    try:
        model = TD3.load(args.model_path, env=env)
        print(f"Loaded existing model from {args.model_path}")
    except Exception:
        model = TD3("MlpPolicy", env, verbose=1,
                    tensorboard_log=args.tensorboard_log)
        print("Created new model")

    model.learn(total_timesteps=args.total_timesteps)

    eval_env = DummyVecEnv([make_env()])
    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=args.eval_episodes
    )
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    model.save(args.model_path)

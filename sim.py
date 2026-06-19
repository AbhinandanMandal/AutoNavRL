
"""
This is for training robot with RL algorithm before testing on physical world
---------------------------------------------------------------------------------
At first we put charactics of physical world into 'robot_world.yaml'
Using 'robot_world.yaml' this sim.py creates a simulation environment and take readings
The output of 'SIM_ENV' (sim.py) goes into 'RobotNavEnv' (train.py) for training specific algorithm
Then in 'run.py' we can train the simulation to actually see the performances. 
"""

"""
1. robot_world.yaml  ->  world geometry / sensor config
2. SIM_ENV (this file)  ->  simulation step + A*-planned waypoints
3. RobotNavEnv (train.py)  ->  Gym wrapper consumed by stable-baselines3
4. run.py  ->  evaluation loop
"""

"""

A* integration
--------------
At every reset() the AStarPlanner rasterises current obstacle positions and
plans a waypoint path from the robot's start to the final goal.  The waypoints
are exposed via ``self.waypoints`` and ``self.current_waypoint_index``.
 
During each step() the *active* target is the next waypoint, not the distant
final goal.  When the robot arrives within ``waypoint_radius`` of the current
waypoint it advances to the next one.  When the last waypoint (== the final
goal) is reached irsim reports ``robot.arrive = True`` as usual.
 
``self.current_target`` always returns the (x, y) the reward shaper should
aim at — train.py reads this instead of the raw goal.
"""

import numpy as np
import random
import shapely
from irsim.lib.handler.geometry_handler import GeometryFactory
from irsim.env import EnvBase

from astar_planner import AStarPlanner


class SIM_ENV:
    """
    Simulator environment wrapper for robot navigation.
    
    New public attributes (A* integration)
    ---------------------------------------
    planner : AStarPlanner
              The grid-based A* planner instance.

    waypoints : list of (float, float)
                World-coordinate waypoints from start → final goal, planned at each
                reset().  Empty until the first reset() call.

    current_waypoint_index : int
                             Index of the waypoint the robot is currently heading toward.

    waypoint_radius : float
                      Distance (metres) at which the robot is considered to have reached a
                      waypoint and advances to the next one.

    current_target : (float, float)
                     The (x, y) position the robot should aim at right now.  Equals
                    ``waypoints[current_waypoint_index]`` if waypoints exist, otherwise
                      falls back to the final goal position.
    """
    
    def __init__(self, world_file="robot_world.yaml", render=False):
        """
        Initialize the simulation environment.
        
        Args:
            world_file (str): Path to the world configuration file
            render (bool): Whether to enable visualization
        """
        # Initialize environment
        self.env = EnvBase(world_file, display=render, disable_all_plot=not render)
        self.robot_goal = self.env.get_robot_info(0).goal

        # A* planner - parameters match the 6x6 robot world and specs 
        self.planner = AStarPlanner(
            world_w=6.0, world_h=6.0, cell_size=0.15, robot_radius=0.34, inflation_margin=0.12
        )

        # waypoint stats
        self.waypoints:list=[]
        self.current_waypoint_index:int=0
        self.waypoint_radius:float = 0.45 
        
        # Initialize tracking variables
        self._reset_tracking()

    # Current target of the robot
    @property
    def current_target(self):
        """ 
        Active (x,y) sub-goal of the robot that it should steer toward.
        It returns the next un-reached waypoint
        """
        if self.waypoints and self.current_waypoint_index < len(self.waypoints):
            return self.waypoints[self.current_waypoint_index]
        
        # If not then returns this
        return (self.robot_goal[0].item(), self.robot_goal[1].item())
    

    def _reset_tracking(self):
        """Reset tracking variables for distance and angle differences."""
        self.prev_distance = None
        self.prev_diff_rad = None


    # Path planning internal helpers to make path from start to goal
    def _plan_path(self, start_xy, goal_xy):
        """
        Build the occupancy grid from current obstacles and run A*.
 
        Called inside reset() after obstacle positions are finalised.
        Populates ``self.waypoints`` and resets ``self.current_waypoint_index``.
        """
        self.planner.build_grid(self.env.obstacle_list)
        self.waypoints = self.planner.plan(start_xy, goal_xy)
        self.current_waypoint_index = 0

        print(
            f"[A*] Planned {len(self.waypoints)} waypoints from "
            f"({start_xy[0]:.2f}, {start_xy[1]:.2f}) → "
            f"({goal_xy[0]:.2f}, {goal_xy[1]:.2f})"
        )


    def _advance_waypoint_if_reached(self, robot_xy):
        """
        Check whether the robot has reached the current waypoint.
 
        If so, advance the index so the next step uses the following waypoint.
        Does nothing once all waypoints are consumed (irsim handles final arrival).
        """
        if not self.waypoints:
            return
        if self.current_waypoint_index >= len(self.waypoints):
            return

        tx, ty = self.current_target
        dist = np.hypot(robot_xy[0] - tx, robot_xy[1] - ty)
        if dist < self.waypoint_radius:
            self.current_waypoint_index = min(
                self.current_waypoint_index + 1, len(self.waypoints) - 1
            )
    

    def _calculate_robot_metrics(self, robot_state):
        """
        Compute goal-related metrics against the current waypoint (subgola)
        
        Args:
            robot_state: Current state of the robot [x, y, theta]
            
        Returns:
            goal_vector : list[float, float]
            distance : float = Distance to the current waypoint.
            cos, sin : float = Angle between robot heading and waypoint.
            diff_rad : float = Heading difference between robot and final goal orientation
        """

        target_x, target_y = self.current_target

        # Calculate goal vector
        # goal_vector = [
        #     self.robot_goal[0].item() - robot_state[0].item(),
        #     self.robot_goal[1].item() - robot_state[1].item(),
        # ]
        goal_vector = [
            target_x - robot_state[0].item(),
            target_y - robot_state[1].item(),
        ]
        
        # Calculate angle difference between robot orientation and goal orientation
        diff_rad = float(((-robot_state[2] + self.env.robot.goal[2] + np.pi) % (2 * np.pi)) - np.pi)
        
        # Calculate distance and pose
        distance = np.linalg.norm(goal_vector)
        pose_vector = [np.cos(robot_state[2]).item(), np.sin(robot_state[2]).item()]
        cos, sin = self._calculate_cossin(pose_vector, goal_vector)
        
        return goal_vector, distance, cos, sin, diff_rad

    @staticmethod
    def _calculate_cossin(vec1, vec2):
        """
        Calculate cosine and sine between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            tuple: (cosine, sine) of the angle between vectors
        """
        vec1 = vec1 / (np.linalg.norm(vec1)+1e-9)
        vec2 = vec2 / (np.linalg.norm(vec2)+1e-9)
        cos = np.dot(vec1, vec2)
        sin = vec1[0] * vec2[1] - vec1[1] * vec2[0]
        return cos, sin

    def _calculate_reward(self, goal, collision, distance_delta, action, laser_scan, delta_rad):
        """
        Calculate reward based on various factors including goal achievement,
        collision avoidance, and movement efficiency.
        
        Args:
            goal (bool): Whether goal is reached
            collision (bool): Whether collision occurred
            distance_delta (float): Change in distance to goal
            action (list): [linear_velocity, angular_velocity]
            laser_scan (list): Laser scan readings
            delta_rad (float): Change in angle difference
            
        Returns:
            float: Calculated reward
        """
        if goal:
            return 100.0
        elif collision:
            return -100.0
        
        # Reward components
        progress_reward = distance_delta * 10  # Reward for moving closer to goal
        dir_progress = delta_rad * 1  # Reward for aligning with goal
        time_penalty = -0.65  # Penalty for each time step
        rotation_penalty = -abs(action[1]) * 0.4  # Penalty for excessive rotation
        
        # Obstacle avoidance
        safe_distance = 1.35
        min_dist = min(laser_scan)
        obstacle_penalty = -(safe_distance - min_dist) if min_dist < safe_distance else 0
        
        return progress_reward + time_penalty + obstacle_penalty + dir_progress + rotation_penalty


    def step(self, lin_velocity=0.0, ang_velocity=0.1):
        """
        Execute one simulation step.
        
        Args:
            lin_velocity (float): Linear velocity
            ang_velocity (float): Angular velocity
            
        Returns:
            tuple: (laser_scan, distance, cos, sin, collision, goal, diff_rad, action, reward)
        """
        # Step simulation
        self.env.step(action_id=0, action=np.array([[lin_velocity], [ang_velocity]]))
        if self.env.display:
            self.env.render()

        # Get sensor data
        scan = self.env.get_lidar_scan()
        robot_state = self.env.get_robot_state()
        robot_xy = [robot_state[0].item(), robot_state[1].item()]

        # Advance waypoint if the robot has reached the current one
        self._advance_waypoint_if_reached(robot_xy)

        # Calculate metrics
        goal_vector, distance, cos, sin, diff_rad = self._calculate_robot_metrics(robot_state)
        
        # Calculate deltas
        if self.prev_distance is None:
            distance_delta = delta_rad = 0
        else:
            distance_delta = self.prev_distance - distance
            delta_rad = abs(self.prev_diff_rad) - abs(diff_rad)
        
        # Update tracking
        self.prev_distance = distance
        self.prev_diff_rad = diff_rad
        
        # Get status and calculate reward
        goal = self.env.robot.arrive
        collision = self.env.robot.collision
        action = [lin_velocity, ang_velocity]
        reward = self._calculate_reward(goal, collision, distance_delta, action, scan["ranges"], delta_rad)
        
        if goal:
            print("Goal reached")

        return scan["ranges"], distance, cos, sin, collision, goal, diff_rad, action, reward

    def reset(self, robot_state=None, robot_goal=None, random_obstacles=True):
        """
        Reset the simulation environment, replan the A* path and return the initial state.
        
        Args:
            robot_state (list): Initial robot state [x, y, theta]
            robot_goal (list): Goal position [x, y, theta]
            random_obstacles (bool): Whether to place random obstacles
            
        Returns:
            tuple: Initial state information
        """
        # Initialize robot state
        if robot_state is None:
            robot_state = [[random.uniform(0.5, 5.5)], 
                          [random.uniform(0.5, 5.5)], 
                          [0]]

        self.env.robot.set_state(state=np.array(robot_state), init=True)

        # Place obstacles
        if random_obstacles:
            self.env.random_obstacle_position(
                range_low=[0, 0, -3.14],
                range_high=[6, 6, 3.14],
                ids=list(range(1, 7)),
                non_overlapping=True
            )

        # Set goal
        if robot_goal is None:
            robot_goal = self._generate_valid_goal()
        
        self.env.robot.set_goal(np.array(robot_goal), init=True)
        self.env.reset()
        self.robot_goal = self.env.robot.goal

        # Plan A* path now tha obstacles and goal are finalised
        start_xy = (robot_state[0][0], robot_state[1][0])
        goal_xy = (robot_goal[0][0], robot_goal[1][0])
        self._plan_path(start_xy, goal_xy)

        self._reset_tracking()
        
        # Get initial state
        action = [0.0, 0.0]
        return self.step(lin_velocity=action[0], ang_velocity=action[1])


    def _generate_valid_goal(self):
        """
        Generate a valid goal position that doesn't overlap with obstacles.
        
        Returns:
            list: Valid goal position [x, y, theta]
        """
        while True:
            goal = [[random.uniform(0.5, 5.5)], 
                   [random.uniform(0.5, 5.5)], 
                   [random.uniform(-3.14, 3.14)]]
            
            # Check if goal overlaps with obstacles
            shape = {"name": "circle", "radius": 0.4}
            state = [goal[0], goal[1], goal[2]]
            gf = GeometryFactory.create_geometry(**shape)
            geometry = gf.step(np.c_[state])
            
            if not any(shapely.intersects(geometry, obj._geometry) for obj in self.env.obstacle_list):
                return goal
            

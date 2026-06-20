
"""
This is for actually physical world testing
-------------------------------------------------------------------------
RL directly dosen't communicate with environment
First it takes the input from 'REAL_ENV' of real_env.py
Secondly the inputs from 'REAL_ENV' goes into 'RobotNavEnv' (env.py) which wraps gym environment, which
- Converts the simulator's outputs into a fixed-size observation space
    - Defines a continuous action space for linear and angular velocities
    - Handles state normalization and preprocessing
    - Manages episode termination conditions

Then output of RobotNavEnv goes into run.py that helps robot to actually move using RL algorithm


General Pipeline
-----------------
robot_world -> real_env.py -> env.py -> run.py

A* integration (mirrors the simulation approach in sim.py)
----------------------------------------------------------
* On reset(), after the initial LiDAR scan is received, AStarPlannerROS
  builds an occupancy grid from that scan and plans a waypoint path from the
  robot's current position to the final goal.
 
* On every step(), the occupancy grid is refreshed from the latest scan and
  the planner can optionally replan (``replan_interval`` steps).  The active
  target (``current_target``) is always the nearest un-reached waypoint.
 
* When the robot comes within ``waypoint_radius`` metres of the current
  waypoint the index advances to the next one.  The final waypoint is the
  original goal; irsim's arrival flag is replaced here by an explicit
  distance + angle check.
 
* ``step()`` returns distance/cos/sin relative to ``current_target`` so the
  RL observation always describes the nearest subgoal, giving denser guidance
  - exactly as in the simulation wrapper.

"""

#!/usr/bin/env python
import rospy
import numpy as np
import math
import time

from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from tf.transformations import euler_from_quaternion
import tf

from astar_planner_ros import AStarPlannerROS




def reduce_lidar_scan(scan_array):
    """
    Reduce LIDAR scan data dimensionality by averaging groups of 4 readings.

    Args:
        scan_array (numpy.ndarray): Raw LIDAR scan data

    Returns:
        numpy.ndarray: Reduced LIDAR scan data
    """
    # Make sure the length is divisible by 4, otherwise trim the excess elements
    if len(scan_array) % 4 != 0:
        scan_array = scan_array[:-(len(scan_array) % 4)]

    # Reshape the array into chunks of 4 elements
    reshaped_array = np.array(scan_array).reshape(-1, 4)

    # Compute the average for each chunk, excluding zeros
    result = []
    for chunk in reshaped_array:
        non_zero_values = chunk[chunk != 0]
        if len(non_zero_values) == 0:
            result.append(10)  # All zeros, use 10
        else:
            # Take the mean of non-zero values
            result.append(np.mean(non_zero_values))

    return np.array(result)


def constrain_lidar_scan(bot_pos, yaw, angles, lidar_ranges, box_limits):
    """
    Constrain LIDAR readings to stay within specified box limits.
    This is usefull if your arena lack proper boundaries.

    Args:
        bot_pos (tuple): (x, y) robot position
        yaw (float): Robot orientation
        angles (numpy.ndarray): LIDAR beam angles
        lidar_ranges (numpy.ndarray): LIDAR range readings
        box_limits (tuple): (min_x, max_x, min_y, max_y) environment boundaries

    Returns:
        numpy.ndarray: Constrained LIDAR ranges
    """
    bot_x, bot_y = bot_pos
    min_x, max_x, min_y, max_y = box_limits
    constrained_ranges = np.empty_like(lidar_ranges)

    for i, (angle, lidar_range) in enumerate(zip(angles, lidar_ranges)):
        adjusted_angle = angle + yaw
        ray_dx = np.cos(adjusted_angle)
        ray_dy = np.sin(adjusted_angle)

        distances = []

        # Check intersections with vertical boundaries
        if ray_dx != 0:
            t1 = (min_x - bot_x) / ray_dx
            t2 = (max_x - bot_x) / ray_dx
            distances.extend([t for t in [t1, t2] if t > 0])

        # Check intersections with horizontal boundaries
        if ray_dy != 0:
            t3 = (min_y - bot_y) / ray_dy
            t4 = (max_y - bot_y) / ray_dy
            distances.extend([t for t in [t3, t4] if t > 0])

        if distances:
            min_boundary_dist = min(distances)
            constrained_ranges[i] = min(lidar_range, min_boundary_dist)
        else:
            constrained_ranges[i] = lidar_range

    return constrained_ranges


class REAL_ENV:
    """
    Real robot environment class that interfaces with ROS, augmented with A* path planning.

    This class handles:
    - LIDAR data processing
    - Robot motion control
    - State tracking and goal progress
    - Collision detection

    Attributes:
        scan_sub (rospy.Subscriber): LIDAR scan subscriber
        tf_listener (tf.TransformListener): TF listener for pose tracking
        cmd_vel_pub (rospy.Publisher): Velocity command publisher
        latest_scan (list): Most recent LIDAR scan data
        robot_pose (list): Current robot position [x, y]
        robot_yaw (float): Current robot orientation
        collision (bool): Collision flag
        goal_reached (bool): Goal reached flag
        robot_goal (list): Target pose [x, y, yaw]

    New parameters for A*
    ---------------------
    replan_interval : int 
        How often (in steps) to rebuild the occupancy grid and replan the path
        using the latest LiDAR scan.  Set to 0 to plan only once at reset().
        20–30 steps (≈ 2–3 s at 10 Hz) is a reasonable value for a dynamic
        environment; use 0 for a mostly static arena.
    
        waypoint_radius : float
            Distance in metres at which the robot is considered to have reached a
            waypoint and the index advances (default 0.45 m).
                      
    """

    def __init__(self, goal_pose=None, replan_interval: int=120, waypoint_radius:float=0.45):
        """Initialize the real robot environment."""
        # ROS interface
        self.scan_sub = rospy.Subscriber(
            '/lidar_scan', LaserScan, self.scan_callback)
        self.tf_listener = tf.TransformListener()
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        # Sensors and State variables
        self.latest_scan = []
        self._raw_scan_msg  =None # full laserscan message for A* grid build
        self.robot_pose = [0.0, 0.0]
        self.robot_yaw = 0.0
        self.collision = False
        self.goal_reached = False
        self.robot_goal = goal_pose

        # Performance metrics
        self.start_time = time.time()
        self.path_length = 0
        self.linear_vel_sum = 0
        self.angular_vel_sum = 0
        self.timestep = 0
        self.prev_pose = None

        # A* planner
        self.planner = AStarPlannerROS(
            world_w=6.0,
            world_h=6.0,
            cell_size=0.15,
            robot_radius=0.34,
            inflation_margin=0.12,
        )
        self.waypoints: list = []
        self.current_waypoint_index: int = 0
        self.waypoint_radius: float = waypoint_radius
        self.replan_interval: int = replan_interval

        rospy.sleep(1)  # Initialization wait


    # current target properties
    @property
    def current_target(self):
        """
        Active (x, y) subgoal for the current step.
 
        Returns the next un-reached waypoint, or the final goal as fallback.
        """
        if self.waypoints and self.current_waypoint_index < len(self.waypoints):
            return self.waypoints[self.current_waypoint_index]
        if self.robot_goal is not None:
            return (self.robot_goal[0], self.robot_goal[1])
        return (0.0, 0.0)
    

    # ROS callback
    def scan_callback(self, data:LaserScan):
        """
        Process incoming LIDAR scan data.

        Args:
            data (LaserScan): Raw LIDAR scan message
        """
        self._raw_scan_msg = data 
        latest_scan = reduce_lidar_scan(data.ranges)

        # bot_position = self.robot_pose
        # bot_yaw = self.robot_yaw

        # Calculate LIDAR position
        lidar_offset = 0.15  # LIDAR is 0.15m ahead of robot center
        lidar_x = self.robot_pose[0] + lidar_offset * np.cos(self.robot_yaw)
        lidar_y = self.robot_pose[1] + lidar_offset * np.sin(self.robot_yaw)

        # Generate LIDAR beam angles
        lidar_angles = np.linspace(0, 2 * np.pi, num=420)
        box_limits = (0, 6, 0, 6)  # Environment boundaries

        # Constrain LIDAR readings to environment boundaries
        latest_scan = constrain_lidar_scan(
            [lidar_x, lidar_y],
            self.robot_yaw,
            lidar_angles,
            latest_scan,
            box_limits
        )

        # Rotate scan data to align with robot orientation
        self.latest_scan = np.roll(latest_scan, int(len(latest_scan) * 0.5))

        # Check for collisions
        # 15cm collision threshold
        # Option 1
        self.collision = bool(np.min(self.latest_scan) < 0.15)
        if self.collision:
            self._stop_robot()
            rospy.logwarn("Collision detected!")

        # Option 2
        # self.collision = min(self.latest_scan) < 0.15
        # if self.collision:
        #     cmd = Twist()
        #     cmd.linear.x = 0
        #     cmd.angular.z = 0
        #     self.cmd_vel_pub.publish(cmd)
        #     print("Collision detected!")


    def get_robot_pose_from_tf(self):
        """Get current robot pose from TF."""
        self.tf_listener.waitForTransform(
            "origin", "base_link", rospy.Time(0), rospy.Duration(0.1))
        (trans, rot) = self.tf_listener.lookupTransform(
            "origin", "base_link", rospy.Time(0))

        self.robot_pose = [trans[0], trans[1]]
        _, _, self.robot_yaw = euler_from_quaternion(rot)

        print("TF Pose:", self.robot_pose, self.robot_yaw)


    # A* helpers
    def _build_and_plan(self):
        """
        Rebuild the occupancy grid from the current LiDAR scan and replan.
 
        Safe to call even if no scan has arrived yet (produces an empty grid
        and falls back to a direct-to-goal path).
        """
        if self._raw_scan_msg is None:
            rospy.logwarn(
                "[A*] No LiDAR scan available yet - skipping grid build")
            self.waypoints = [(self.robot_goal[0], self.robot_goal[1])]
            self.current_waypoint_index = 0
            return

        msg = self._raw_scan_msg
        ranges = np.array(msg.ranges, dtype=float)

        self.planner.build_grid_from_scan(
            robot_xy=self.robot_pose,
            robot_yaw=self.robot_yaw,
            ranges=ranges,
            angle_min=msg.angle_min,
            angle_increment=msg.angle_increment,
            max_range=min(msg.range_max, 7.0),
            lidar_offset=0.15,
        )

        start_xy = (self.robot_pose[0], self.robot_pose[1])
        goal_xy = (self.robot_goal[0], self.robot_goal[1])
        self.waypoints = self.planner.plan(start_xy, goal_xy)

        # Do not reset the index on a replan - keep whichever waypoint the
        # robot was heading toward so we never regress backward along the path.
        # But clamp it in case the new path is shorter.
        self.current_waypoint_index = min(
            self.current_waypoint_index, max(0, len(self.waypoints) - 1)
        )

        rospy.loginfo(
            f"[A*] Planned {len(self.waypoints)} waypoints  "
            f"start={start_xy}  goal={goal_xy}"
        )


    # When bot rached within the waypoint radius 
    def _advance_waypoint_if_reached(self):
        """Advance waypoint index if robot is within waypoint_radius of current target."""
        if not self.waypoints or self.current_waypoint_index >= len(self.waypoints):
            return
        tx, ty = self.current_target
        dist = math.hypot(self.robot_pose[0] - tx, self.robot_pose[1] - ty)
        if dist < self.waypoint_radius:
            self.current_waypoint_index = min(
                self.current_waypoint_index + 1, len(self.waypoints) - 1
            )
            rospy.loginfo(
                f"[A*] Waypoint reached → advancing to "
                f"{self.current_waypoint_index + 1}/{len(self.waypoints)}"
            )


    # Internal utilities for robot
    def _stop_robot(self):
        cmd = Twist()
        self.cmd_vel_pub.publish(cmd)

    @staticmethod
    def cossin(vec1, vec2):
        vec1 = vec1 / (np.linalg.norm(vec1) + 1e-9)
        vec2 = vec2 / (np.linalg.norm(vec2) + 1e-9)
        cos = np.dot(vec1, vec2)
        sin = vec1[0] * vec2[1] - vec1[1] * vec2[0]
        return float(cos), float(sin)

    def _reset_metrics(self):
        self.start_time = time.time()
        self.path_length = 0.0
        self.linear_vel_sum = 0.0
        self.angular_vel_sum = 0.0
        self.timestep = 0
        self.prev_pose = None

    def _log_performance(self, distance, diff_rad):
        rospy.loginfo("Goal reached!")
        rospy.loginfo(f"  Time taken:   {time.time() - self.start_time:.1f} s")
        rospy.loginfo(f"  Distance:     {distance:.3f} m")
        rospy.loginfo(f"  Ang diff:     {diff_rad:.3f} rad")
        rospy.loginfo(f"  Path length:  {self.path_length:.2f} m")
        avg_lin = self.linear_vel_sum / max(self.timestep, 1)
        avg_ang = self.angular_vel_sum / max(self.timestep, 1)
        rospy.loginfo(f"  Avg lin vel:  {avg_lin:.3f} m/s")
        rospy.loginfo(f"  Avg ang vel:  {avg_ang:.3f} rad/s")
        rospy.loginfo(
            f"  Waypoints:    {self.current_waypoint_index + 1}/{len(self.waypoints)} reached"
        )



    # gym interface
    def step(self, lin_velocity=0.0, ang_velocity=0.1):
        """
        Execute one step in the environment.

        Args:
            lin_velocity (float): Linear velocity command
            ang_velocity (float): Angular velocity command

        Returns:
            tuple: (scan_data, distance, cos, sin, collision, goal, diff_rad, action, reward)
        """
        self.timestep += 1
        self.get_robot_pose_from_tf()

        if self.robot_pose is None:
            rospy.logwarn("Waiting for AMCL pose...")
            rospy.sleep(0.1)
            return None
        

        # Periodic replan from latest scan
        if (
            self.replan_interval > 0
            and self.timestep % self.replan_interval == 0
            and not self.goal_reached
        ):
            self._build_and_plan()

        # Advance to next waypoint if close enough
        self._advance_waypoint_if_reached()


        # Publish velocity command
        cmd = Twist()
        cmd.linear.x = lin_velocity
        cmd.angular.z = ang_velocity
        if not self.collision and not self.goal_reached:
            self.cmd_vel_pub.publish(cmd)

        rospy.sleep(0.1)  # Allow time for motion

        # metrics relative to current waypoint
        tx, ty = self.current_target
        goal_vector = [tx - self.robot_pose[0], ty - self.robot_pose[1]]
        distance = np.linalg.norm(goal_vector)
        # goal_vector = [
        #     self.robot_goal[0] - self.robot_pose[0],
        #     self.robot_goal[1] - self.robot_pose[1],
        # ]

        # Calculate angle difference to goal
        diff_rad = float(
            ((-self.robot_yaw + self.robot_goal[2] + math.pi) % (2 * math.pi)) - math.pi)
        # distance = np.linalg.norm(goal_vector)

        # final goal arrival check
        final_dist = math.hypot(
            self.robot_goal[0] - self.robot_pose[0],
            self.robot_goal[1] - self.robot_pose[1],
        )

        # 15cm position and 0.15rad angle threshold
        goal = (distance < 0.15 and abs(diff_rad) < 0.15)

        # Update path length
        # if self.prev_pose is not None:
        #     delta = np.sqrt((self.robot_pose[0] - self.prev_pose[0])**2 +
        #                     (self.robot_pose[1] - self.prev_pose[1])**2)
        #     self.path_length += delta
        # self.prev_pose = self.robot_pose

        if self.prev_pose is not None:
            self.path_length += math.hypot(
                self.robot_pose[0] - self.prev_pose[0],
                self.robot_pose[1] - self.prev_pose[1],
            )
        self.prev_pose = list(self.robot_pose)

        # Update velocity sums for averaging in metrics
        self.linear_vel_sum += abs(lin_velocity)
        self.angular_vel_sum += abs(ang_velocity)

        # if goal:
        #     print(self.robot_yaw, self.robot_goal[2])
        #     rospy.loginfo("Goal reached!")
        #     cmd = Twist()
        #     cmd.linear.x = 0
        #     cmd.angular.z = 0
        #     self.cmd_vel_pub.publish(cmd)
        #     self.goal_reached = True

        #     # Log performance metrics
        #     print("Time Taken:", time.time() - self.start_time)
        #     print("Distance:", distance)
        #     print("Ang_diff:", diff_rad)
        #     print("Path Length:", self.path_length)
        #     print('Avg Linear:', self.linear_vel_sum/self.timestep)
        #     print('Avg Ang:', self.angular_vel_sum/self.timestep)

        if goal and not self.goal_reached:
            self._log_performance(final_dist, diff_rad)
            self._stop_robot()
            self.goal_reached = True

        # Compute observation components
        pose_vector = [math.cos(self.robot_yaw), math.sin(self.robot_yaw)]
        cos, sin = self.cossin(np.array(pose_vector), np.array(goal_vector))
        action = [lin_velocity, ang_velocity]
        reward = 0  # Made for inference, not used

        return self.latest_scan, distance, cos, sin, self.collision, goal, diff_rad, action, reward

    def reset(self, goal_pose=None):
        """
        Reset the environment with a new goal.

        Args:
            goal_pose (list): [x, y, yaw] target pose

        Returns:
            tuple: Initial environment state
        """
        # Reset metrics
        self.start_time = time.time()
        self.path_length = 0
        self.timestep = 0
        self.linear_vel_sum = 0
        self.angular_vel_sum = 0

        # Get initial pose
        self._reset_metrics()
        self.get_robot_pose_from_tf()
        self.prev_pose = None

        rospy.loginfo("Manually reset the robot and localization if needed.")
        rospy.sleep(2)  # Wait for manual reset or AMCL reinitialization

        # Set new goal
        self.robot_goal = goal_pose
        self.collision = False
        self.goal_reached = False

        # wait for a fresh scan before planning
        rospy.sleep(0.3)

        # plan initial A* path from current pose to goal
        self.current_waypoint_index = 0
        self._build_and_plan()

        # Take initial step
        action = [0.0, 0.0]
        return self.step(lin_velocity=action[0], ang_velocity=action[1])

    @staticmethod
    def cossin(vec1, vec2):
        """
        Compute cosine and sine of angle between two vectors.

        Args:
            vec1 (list): First vector [x, y]
            vec2 (list): Second vector [x, y]

        Returns:
            tuple: (cosine, sine) of angle between vectors
        """
        vec1 = vec1 / np.linalg.norm(vec1)
        vec2 = vec2 / np.linalg.norm(vec2)
        cos = np.dot(vec1, vec2)
        sin = vec1[0] * vec2[1] - vec1[1] * vec2[0]
        return cos, sin


if __name__ == "__main__":
    # Test the environment
    rospy.init_node('real_robot_env', anonymous=True)
    env = REAL_ENV(goal_pose=[0, 0, 0])
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        step_result = env.step(0.0, 0.0)
        if step_result:
            scan, distance, cos, sin, collision, goal, diff_rad, action, reward = step_result
            rospy.loginfo(f"Distance: {distance:.2f}, Reward: {reward:.2f}")
            if collision or goal:
                break
        rate.sleep()

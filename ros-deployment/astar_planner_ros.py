"""
Key difference from the simulation version (astar_planner.py)
--------------------------------------------------------------
In simulation, obstacle geometry is available as a list of shapely objects,
so we rasterise them directly.  On the real robot the only obstacle
information is the live LiDAR scan.  This planner accepts a lidar scan
(ranges + angles) and marks occupied cells by ray-casting each beam onto the
grid.  Cells are inflated by robot_radius + inflation_margin before search.


planner = AStarPlannerROS(
    world_w=6, world_h=6,
    cell_size=0.15,
    robot_radius=0.34,
    inflation_margin=0.12,
)

planner.build_grid_from_scan(
    robot_xy=(rx, ry),
    robot_yaw=yaw,
    ranges=np.array([...]),       # one range per beam, metres
    angle_min=0.0,                # first beam angle (rad, robot frame)
    angle_increment=...,          # radians per beam
    max_range=7.0,                # discard readings beyond this
)

waypoints = planner.plan(start_xy, goal_xy)
# returns [(x, y), ...] in world (map) coordinates
"""

import heapq
import math
import numpy as np


class AStarPlannerROS:
    """
    Occupancy-grid A* planner driven by a live LiDAR scan.

    Parameters
    ----------
    world_w, world_h : float = World dimensions in metres (must match the real arena size).
    cell_size : float = Grid resolution in metres.  0.15 m -> 40×40 grid for a 6×6 world.
    robot_radius : float = Robot body radius in metres.
    inflation_margin : float = Extra clearance on top of robot_radius (metres).
    """

    def __init__(
        self,
        world_w: float = 6.0,
        world_h: float = 6.0,
        cell_size: float = 0.15,
        robot_radius: float = 0.34,
        inflation_margin: float = 0.12,
    ):
        self.world_w = world_w
        self.world_h = world_h
        self.cell_size = cell_size
        self.inflation = robot_radius + inflation_margin

        self.cols = math.ceil(world_w / cell_size)
        self.rows = math.ceil(world_h / cell_size)

        # Inflation radius in cells (ceiling so we never under-inflate)
        self._inf_cells = math.ceil(self.inflation / cell_size)

        # Occupancy grid — True = blocked
        self.grid = np.zeros((self.rows, self.cols), dtype=bool)

        # 8-connected movement costs
        self._moves = [
            (1, 0, 1.0), (-1, 0, 1.0),
            (0, 1, 1.0), (0, -1, 1.0),
            (1,  1, math.sqrt(2)), (1, -1, math.sqrt(2)),
            (-1, 1, math.sqrt(2)), (-1, -1, math.sqrt(2)),
        ]

    # world to grid environment
    def _world_to_grid(self, x: float, y: float):
        col = int(x / self.cell_size)
        row = int(y / self.cell_size)
        col = max(0, min(col, self.cols - 1))
        row = max(0, min(row, self.rows - 1))
        return col, row

    # grid to world environment
    def _grid_to_world(self, col: int, row: int):
        return (col + 0.5) * self.cell_size, (row + 0.5) * self.cell_size

    # Grid build from lidar scan for obstacle detection and navigation

    def build_grid_from_scan(
        self,
        robot_xy,
        robot_yaw: float,
        ranges,
        angle_min: float = 0.0,
        angle_increment: float = None,
        max_range: float = 7.0,
        lidar_offset: float = 0.15,
    ):
        """
        Build (or refresh) the occupancy grid from a LiDAR scan.

        Each valid beam endpoint is marked as occupied, then the occupied set
        is inflated by ``self.inflation`` metres to account for robot body size.
        The world boundary is always marked as occupied.

        Parameters
        ----------
        robot_xy : array-like (2,) = Robot position (x, y) in world/map frame metres.
        robot_yaw : float = Robot heading in radians (world frame).
        ranges : array-like = Per-beam distance readings in metres.
        angle_min : float
            Angle of the first beam relative to the robot's forward direction
            (radians).  For a 360° scan centred on 0, pass ``-math.pi``.
        angle_increment : float or None
            Angular step between beams.  If None, 2π / len(ranges) is used
            (assumes evenly spaced 360° scan).
        max_range : float
            Beams longer than this are treated as free (no obstacle detected).
        lidar_offset : float
            Distance of the LiDAR sensor ahead of the robot centre (metres).
            Matches the offset in real_env.py (0.15 m).
        """
        ranges = np.asarray(ranges, dtype=float)
        n = len(ranges)

        if angle_increment is None:
            angle_increment = 2 * math.pi / n

        rx, ry = float(robot_xy[0]), float(robot_xy[1])

        # LiDAR sensor position (slightly ahead of robot centre)
        lx = rx + lidar_offset * math.cos(robot_yaw)
        ly = ry + lidar_offset * math.sin(robot_yaw)

        # Start with a clean grid, then re-mark boundary
        self.grid[:] = False
        self._mark_boundary()

        # Collect raw hit cells before inflation
        hit_cells = set()
        for i, r in enumerate(ranges):
            if not np.isfinite(r) or r <= 0.01 or r >= max_range:
                continue
            beam_angle = robot_yaw + angle_min + i * angle_increment
            hx = lx + r * math.cos(beam_angle)
            hy = ly + r * math.sin(beam_angle)
            # Skip hits outside the world
            if 0 <= hx < self.world_w and 0 <= hy < self.world_h:
                hit_cells.add(self._world_to_grid(hx, hy))

        # Inflate each hit cell
        for (hc, hr) in hit_cells:
            r_inf = self._inf_cells
            for dc in range(-r_inf, r_inf + 1):
                for dr in range(-r_inf, r_inf + 1):
                    if dc * dc + dr * dr <= r_inf * r_inf:
                        nc, nr = hc + dc, hr + dr
                        if 0 <= nc < self.cols and 0 <= nr < self.rows:
                            self.grid[nr, nc] = True

    def _mark_boundary(self):
        """Mark the outermost cell ring as occupied (world walls)."""
        self.grid[0, :] = True
        self.grid[-1, :] = True
        self.grid[:, 0] = True
        self.grid[:, -1] = True

    # A* search implementation

    def _heuristic(self, col, row, gc, gr):
        dx, dy = abs(col - gc), abs(row - gr)
        return max(dx, dy) + (math.sqrt(2) - 1) * min(dx, dy)

    def _nearest_free(self, col, row):
        if not self.grid[row, col]:
            return col, row
        visited = {(col, row)}
        queue = [(col, row)]
        while queue:
            nq = []
            for c, r in queue:
                for dc, dr, _ in self._moves:
                    nc, nr = c + dc, r + dr
                    if (nc, nr) not in visited and \
                       0 <= nc < self.cols and 0 <= nr < self.rows:
                        if not self.grid[nr, nc]:
                            return nc, nr
                        visited.add((nc, nr))
                        nq.append((nc, nr))
            queue = nq
        return col, row

    def _astar(self, sc, sr, gc, gr):
        sc, sr = self._nearest_free(sc, sr)
        gc, gr = self._nearest_free(gc, gr)

        open_heap = []
        g_score = {(sc, sr): 0.0}
        came_from = {}
        heapq.heappush(
            open_heap, (self._heuristic(sc, sr, gc, gr), 0.0, sc, sr))

        while open_heap:
            f, g, col, row = heapq.heappop(open_heap)
            if col == gc and row == gr:
                path = []
                node = (gc, gr)
                while node in came_from:
                    path.append(node)
                    node = came_from[node]
                path.append((sc, sr))
                path.reverse()
                return path

            if g > g_score.get((col, row), float("inf")) + 1e-9:
                continue

            for dc, dr, cost in self._moves:
                nc, nr = col + dc, row + dr
                if not (0 <= nc < self.cols and 0 <= nr < self.rows):
                    continue
                if self.grid[nr, nc]:
                    continue
                ng = g + cost
                if ng < g_score.get((nc, nr), float("inf")):
                    g_score[(nc, nr)] = ng
                    came_from[(nc, nr)] = (col, row)
                    fh = ng + self._heuristic(nc, nr, gc, gr)
                    heapq.heappush(open_heap, (fh, ng, nc, nr))

        return []

    # Waypoint thinning
    @staticmethod
    def _thin_path(path):
        if len(path) <= 2:
            return list(path)
        waypoints = [path[0]]
        for i in range(1, len(path) - 1):
            pc, pr = path[i - 1]
            cc, cr = path[i]
            nc, nr = path[i + 1]
            if (cc - pc, cr - pr) != (nc - cc, nr - cr):
                waypoints.append(path[i])
        waypoints.append(path[-1])
        return waypoints

    # This plan() interact with bot and iteratively helps it to navigate the environment

    def plan(self, start_xy, goal_xy, waypoint_spacing: float = 0.45):
        """
        Plan a path and return world-coordinate waypoints.

        Parameters
        ----------
        start_xy, goal_xy : array-like (2,) = (x, y) positions in metres.
        waypoint_spacing : float = Minimum distance between consecutive waypoints (metres).

        Returns
        -------
        list of (float, float)
            Waypoints from start -> goal.  Always ends at goal_xy.
            Returns [goal_xy] if no path found.
        """
        sx, sy = float(start_xy[0]), float(start_xy[1])
        gx, gy = float(goal_xy[0]), float(goal_xy[1])

        sc, sr = self._world_to_grid(sx, sy)
        gc, gr = self._world_to_grid(gx, gy)

        grid_path = self._astar(sc, sr, gc, gr)
        if not grid_path:
            rospy_warn = getattr(__import__(
                "rospy", errors="ignore"), "logwarn", print)
            rospy_warn(
                "[AStarPlannerROS] No path found — falling back to direct goal")
            return [(gx, gy)]

        thinned = self._thin_path(grid_path)
        world_pts = [self._grid_to_world(c, r) for c, r in thinned]

        # Spacing filter
        filtered = [world_pts[0]]
        for pt in world_pts[1:]:
            if math.dist(filtered[-1], pt) >= waypoint_spacing:
                filtered.append(pt)

        if math.dist(filtered[-1], (gx, gy)) > 0.01:
            filtered.append((gx, gy))

        return filtered

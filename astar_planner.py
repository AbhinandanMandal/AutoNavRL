

import heapq
import math
import numpy as np
import shapely
import shapely.geometry as sg


class AStarPlanner:
    """  
    Occupancy-grid A* planner.
    Grid-based A* path planner for 6x6 robot world (robot_world.yaml)

    The world is discretised into cells of 'cell_size' meters. Obstacles shapes 
    (shapely geometry) are rasterised (to convert a digital image described in a mathematical format)
    into grid. A* finds a collision-free path, which is then thinned to a compact list of
    waypoints by removing collinear intermediate nodes.


    Public interface
    -----------------
    planner = AStarPlanner(world_w=6, world_h=6, cell_size=0.15, robot_radius=0.34)
    planner.build_grid(obstacle_list)                     # call after env.reset()
    waypoints = planner.plan(start_xy, goal_xy)           # returns [(x,y), ...]


    Parameters
    ----------
    world_w, world_h : float     = World dimensions in meters.
    cell_size : float            = Side length of each grid cell in meters.
    robot_radius : float         = Robot body radius in meters. (for effective collision free movement)
    inflation_margin : float     = Extra clearance added on the top of robot_radius in meters.
                                   A safety buffer.

    """

    def __init__(self, world_w: float = 6.0, world_h: float = 6.0,
                 cell_size: float = 0.15, robot_radius: float = 0.34,
                 inflation_margin: float = 0.12):

        self.world_w = world_w
        self.world_h = world_h
        self.cell_size = cell_size
        # combined total inflation for safe navigation
        self.inflation = robot_radius + inflation_margin

        # Grid dimensions (number of cells)
        # 40x40 in this case
        self.cols = math.ceil(world_w/cell_size)
        self.rows = math.ceil(world_h/cell_size)

        # Occupancy grid: True = blocked
        # Initially all are free, later for navigation some of them (obstalces)
        # became true
        self.grid = np.zeros((self.rows, self.cols), dtype=bool)

        # 8 connection movement of robot with cost
        # left, right, up, down, up-left, up-right, down-left, down-right
        self._moves = [
            (1, 0, 1.0), (-1, 0, 1.0),
            (0, 1, 1.0), (0, -1, 1.0),
            (1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)),
            (-1, 1, math.sqrt(2)), (-1, -1, math.sqrt(2)),
        ]

    # World to Grid & Grid to World transformation
    def _world_to_grid(self, x: float, y: float):
        """
        Convert world (x,y) meters to grid (col, row) indices.
        """
        col = int(x/self.cell_size)
        row = int(y/self.cell_size)
        col = max(0, min(col, self.cols-1))
        row = max(0, min(row, self.rows - 1))
        return col, row

    def _grid_to_world(self, col: int, row: int):
        """  
        Convert grid (col, row) to world (x, y) at cell center.
        """
        x = (col+0.5)*self.cell_size
        y = (row+0.5)*self.cell_size
        return x, y

    # Obstacle rasterisation

    def build_grid(self, obstacle_list):
        """
        Rasterise all obstacles onto the occupancy grid.
        Obstacles are inflated by ``self.inflation`` metres so the path
        remains safe for the robot body.  Call this once after each
        ``env.reset()`` because obstacle positions change.

        Parameters
        ----------
        obstacle_list : iterable
            Collection of obstacle objects that expose a ``._geometry``
            attribute (shapely geometry), as used by irsim.
        """

        # The following codeblocks converting continuous obstacle geometry
        # into a discrete occupancy grid that A* can search on.
        # Real world obstacles -> Shapely geometry -> Grid map -> A* path planning
        self.grid[:] = False
        for obj in obstacle_list:
            geom = obj._geometry  # _geometry is a shapely geometry object attached to each obstacle
            if geom is None:  # skipping invalid geometry
                continue

            # If obstacle present the inflate it by robot radius + safety margin
            # So that robot can effectively cover the navigation
            try:
                inflated = geom.buffer(self.inflation)
            except Exception:
                inflated = geom

            # Rasterization
            # For effective navigation
            # if inflation contains points or touches points in grid to world trasnformation
            # then make obstacle as true
            for row in range(self.rows):
                for col in range(self.cols):
                    cx, cy = self._grid_to_world(col, row)
                    point = sg.Point(cx, cy)
                    if inflated.contains(point) or inflated.touches(point):
                        self.grid[row, col] = True

            self.grid[0, :] = True
            self.grid[-1, :] = True
            self.grid[:, 0] = True
            self.grid[:, -1] = True

    def _heuristic(self, col: int, row: int, gc: int, gr: int) -> float:
        """
        A* = g(n)+h(n)
        where h(n) is heuristic for effective A* 
        Octile distance heuristic (admissible for 8-connected grid).

        Args:
            col: int = colums
            row: int = row
            gc : int = goal column
            gr : int = goal row
        """
        dx = abs(col-gc)
        dy = abs(row-gr)
        return max(dx, dy) + (math.sqrt(2)-1)*min(dx, dy)

    # A* path algorithm implementation
    def _astar(self, sc: int, sr: int, gc: int, gr: int):
        """
        Core A* on the occupancy grid.
        Returns list of (col, row) from start to goal, or [] if no path.

        Args:
            sc : int = starting column
            sr : int = starting row
            gc : int = goal (target) column
            gr : int = goal row
        """
        if self.grid[sr, sc] or self.grid[gr, gc]:
            # Start or goal is inside an obstacle - try to nudge
            sc, sr = self._nearest_free(sc, sr)
            gc, gr = self._nearest_free(gc, gr)

        open_heap = []
        g_score = {(sc, sr): 0.0}
        came_from = {}

        h = self._heuristic(sc, sr, gc, gr)
        heapq.heappush(open_heap, (h, 0.0, sc, sr))

        while open_heap:
            f, g, col, row = heapq.heappop(open_heap)

            if col == gc and row == gr:
                # Reconstruct path
                path = []
                node = (gc, gr)
                while node in came_from:
                    path.append(node)
                    node = came_from[node]
                path.append((sc, sr))
                path.reverse()
                return path

            if g > g_score.get((col, row), float("inf")) + 1e-9:
                continue  # Stale entry

            for dc, dr, cost in self._moves:
                nc, nr = col + dc, row + dr
                if nc < 0 or nc >= self.cols or nr < 0 or nr >= self.rows:
                    continue
                if self.grid[nr, nc]:
                    continue

                ng = g + cost
                if ng < g_score.get((nc, nr), float("inf")):
                    g_score[(nc, nr)] = ng
                    came_from[(nc, nr)] = (col, row)
                    fh = ng + self._heuristic(nc, nr, gc, gr)
                    heapq.heappush(open_heap, (fh, ng, nc, nr))

        return []  # No path found

    def _nearest_free(self, col: int, row: int, max_radius: int = 10):
        """Find the nearest free cell to (col, row) via BFS."""
        if not self.grid[row, col]:
            return col, row
        visited = set()
        queue = [(col, row)]
        visited.add((col, row))
        while queue:
            next_queue = []
            for c, r in queue:
                for dc, dr, _ in self._moves:
                    nc, nr = c + dc, r + dr
                    if (nc, nr) in visited:
                        continue
                    if 0 <= nc < self.cols and 0 <= nr < self.rows:
                        if not self.grid[nr, nc]:
                            return nc, nr
                        visited.add((nc, nr))
                        next_queue.append((nc, nr))
            queue = next_queue
        return col, row  # Fallback: return original even if blocked

    # waypoint extraction for robot
    # waypoint can be considered as intermediate target points that the robot follows
    # one by one to eventually each the final target.

    @staticmethod
    def _thin_path(path):
        """
        Remove collinear intermediate nodes to produce a compact waypoint list.
        Keeps start, end, and any node where direction changes.
        """
        if len(path) <= 2:
            return list(path)

        waypoints = [path[0]]
        for i in range(1, len(path) - 1):
            pc, pr = path[i - 1]
            cc, cr = path[i]
            nc, nr = path[i + 1]
            # Direction vectors
            d1 = (cc - pc, cr - pr)
            d2 = (nc - cc, nr - cr)
            if d1 != d2:
                waypoints.append(path[i])
        waypoints.append(path[-1])
        return waypoints

    def plan(self, start_xy, goal_xy, waypoint_spacing: float = 0.5):
        """
        Plan a path from start to goal and return world-coordinate waypoints.

        Parameters
        ----------
        start_xy : array-like, shape (2,)
            Robot start position (x, y) in metres.
        goal_xy : array-like, shape (2,)
            Goal position (x, y) in metres.
        waypoint_spacing : float
            Minimum distance between waypoints (metres).  Collinear thinning
            already compresses the path; this provides a secondary spacing
            guarantee.

        Returns
        -------
        list of (float, float)
            World-coordinate (x, y) waypoints from start → goal.
            Always includes the exact goal as the final entry.
            Returns [goal_xy] if no path is found (degenerate fallback).
        """
        sx, sy = float(start_xy[0]), float(start_xy[1])
        gx, gy = float(goal_xy[0]), float(goal_xy[1])

        sc, sr = self._world_to_grid(sx, sy)
        gc, gr = self._world_to_grid(gx, gy)

        grid_path = self._astar(sc, sr, gc, gr)

        if not grid_path:
            print("[AStarPlanner] No path found - returning direct goal as fallback")
            return [(gx, gy)]

        # Thin collinear nodes - compact turn-point waypoints
        thinned = self._thin_path(grid_path)

        # Convert to world coordinates
        world_pts = [self._grid_to_world(c, r) for c, r in thinned]

        # Secondary spacing filter: drop waypoints that are too close together
        filtered = [world_pts[0]]
        for pt in world_pts[1:]:
            if math.dist(filtered[-1], pt) >= waypoint_spacing:
                filtered.append(pt)

        # Always end exactly at the goal
        if math.dist(filtered[-1], (gx, gy)) > 0.01:
            filtered.append((gx, gy))

        return filtered

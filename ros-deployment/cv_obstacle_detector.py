
"""
Role in the pipeline
--------------------
This module is a SAFETY LAYER, not a replacement for LiDAR.  It runs
alongside the LiDAR pipeline and produces a per-sector distance estimate
from the front camera.  Those estimates are fused into the LiDAR scan
inside REAL_ENV so the RL policy always sees the tightest (most conservative)
distance per angular sector.
 
Additionally, a CV VETO is available: if the camera detects an imminent
obstacle in the forward arc that the LiDAR missed (e.g. a low object below
the LiDAR plane, glass, or a mirror), it can suppress the RL velocity command
and issue a corrective steering impulse.
 
Depth estimation strategy
--------------------------
The Qbot has a standard RGB camera, not a depth camera.  We use MiDaS
(intel-isl/MiDaS) (MiDas : Mixed Dataset Training for Monocular Depth Estimation) for monocular relative depth estimation.  MiDaS outputs
an inverse-depth map (higher value = closer).  A single scale calibration
factor is computed once per reset() using the median LiDAR distance in the
forward arc vs the median MiDaS value in the same arc, so relative depth
becomes metric depth.
 
If torch / MiDaS are unavailable the module falls back gracefully: it uses
simple image-brightness-gradient edge detection to identify near obstacles
and marks those sectors as "unknown / potentially blocked" rather than free.

"""


import math 
import threading
import numpy as np
import rospy 
from sensor_msgs.msg import Image 
from cv_bridge import CvBridge

# Good code practices to handel dependencies
_torch_available = False
_midas_model=False
_midas_transform=False

try:
    import torch
    import torchvision.transforms as T
    _torch_available = True
except ImportError:
    pass

try:
    import cv2
except ImportError:
    cv2 = None


def _load_midas(model_type:str="MiDaS_small"):
    """
    Loading MiDaS small model from torch.hub
    MiDaS_small is around 5Hz on a Raspberry Pi 4 / Jetson Nano for real time
    safety monitoring. In it set model_type to "DPT_Hybrid" for better accuracy 
    in case of having GPU. 

    It returnd (model, transform) or (None, None) on failure.
    """
    global _midas_model, _midas_transform
    if _midas_model is not None:
        return _midas_model, _midas_transform
    if not _torch_available:
        return None, None 
    
    try:
        rospy.loginfo("[CV] Loading MiDaS model …")
        model = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
        model.eval()
        if torch.cuda.is_available():
            model = model.cuda()
            rospy.loginfo("[CV] MiDaS running on GPU")
        else:
            rospy.loginfo("[CV] MiDaS running on CPU")

        transforms = torch.hub.load(
            "intel-isl/MiDaS", "transforms", trust_repo=True)
        if model_type in ("DPT_Large", "DPT_Hybrid"):
            transform = transforms.dpt_transform
        else:
            transform = transforms.small_transform

        _midas_model = model
        _midas_transform = transform
        rospy.loginfo("[CV] MiDaS loaded successfully")
        return model, transform
    except Exception as e:
        rospy.logwarn(f"[CV] MiDaS load failed ({e}) — using edge fallback")
        return None, None



class CVObstacleDetector:
    """
    Monocular-camera obstacle detector with LiDAR-fusion support

    Parameters (tuneable according to physical model)
    ------------------------------------------------------
    hfov_deg : float
        Camera horizontal field of view in degrees.
        Qbot2 / Raspberry Pi Camera V2: ~62.2°
    n_sectors : int
        Number of angular sectors.  Must equal the lidar bin count used
        in env.py (42 by default) so arrays can be element-wise fused.
        Only sectors inside the camera FOV will have finite values;
        sectors outside the FOV remain np.inf.
    img_width, img_height : int
        Expected image dimensions (used for column→angle mapping).
    stop_distance : float
        If any in-FOV sector drops below this distance (metres) the veto
        flag is raised.  Default 0.4 m gives ~60 ms reaction time at
        max linear velocity 0.6 m/s.
    forward_veto_half_angle : float
        Half-angle (radians) of the forward arc checked for the veto.
        Default π/6 = 30° either side of dead-ahead.
    use_midas : bool
        Try to load and use MiDaS.  If False or if MiDaS is unavailable,
        uses the lightweight edge-based fallback.
    scale : float or None
        Metric calibration scale (metres per inverse-depth unit).
        Set by calibrate(); until calibrated all depth values are relative.
    camera_topic : str
        ROS topic for the camera image stream.
    camera_height_m : float
        Camera mounting height above the floor in metres.  Used to compute
        the ground-plane cut-off row so ground pixels are excluded from
        obstacle detection.
    """

    def __init__(
            self,
        hfov_deg: float = 62.2, # Qbot camera horizontal FOC
        n_sectors: int = 42, # LiDAR bin count (must match with env.py)
        img_width: int = 640,
        img_height: int = 480,
        stop_distance: float = 0.4, # CV veto threshold (tunable)
        forward_veto_half_angle: float = math.pi / 6,
        use_midas: bool = True, # False in case of edge-only feedback
        scale: float = None,
        camera_topic: str = "/camera/image_raw",
        camera_height_m: float = 0.20,
    ):
        self.hfov = math.radians(hfov_deg)
        self.n_sectors = n_sectors
        self.img_width = img_width
        self.img_height = img_height
        self.stop_distance = stop_distance
        self.forward_half = forward_veto_half_angle
        self.camera_topic = camera_topic
        self.camera_height_m = camera_height_m

        # Caliberation
        self.scale = scale  # None untill calibrate() is called
        self._calibrated = scale is not None 

        # Latest results (written by camera thread, read by main thread)
        self._lock = threading.Lock()
        self._sector_ranges = np.full(n_sectors, np.inf)  # inf = no obstacle
        self._veto = False
        # suggested corrective angular velocity (rad/s)
        self._veto_turn = 0.0
        # full depth map (for debugging / visualisation)
        self._latest_depth = None

        # ROS
        self._bridge = CvBridge()
        self._sub = None            # created in start()

        # MiDaS
        self._use_midas = use_midas
        self._model, self._transform = None, None

        """
        Angular sector mapping:
        Sector 0 = robot left, sector n_sectors//2 = dead ahead (matches
        the rolled lidar convention in real_env.py where forward is centre).
        Only sectors whose centre angle falls inside ±hfov/2 are active.
        """
        sector_angles = np.linspace(-math.pi, math.pi,
                                    n_sectors, endpoint=False)
        self._active_mask = np.abs(sector_angles) <= self.hfov / 2
        self._sector_angles = sector_angles  # radians, 0 = dead-ahead

        # Pre-compute column -> sector mapping for fast per-pixel assignment
        # Column c (0…W-1) maps to angle: (c / (W-1) - 0.5) * hfov
        col_angles = (np.arange(img_width) / (img_width - 1) - 0.5) * self.hfov
        # Sector index for each column
        self._col_sector = np.searchsorted(
            sector_angles + math.pi,          # shift to [0, 2π]
            col_angles + math.pi,
            side="right",
        ) - 1
        self._col_sector = np.clip(self._col_sector, 0, n_sectors - 1)

        # Ground-plane cut-off: ignore rows below the horizon for the camera
        # pitch=0 assumption.  Approximate: rows > cutoff_row are ground.
        # We compute it from camera height and a 45° depression limit.
        # This is conservative — it keeps a large safety margin.
        focal_len_px = img_width / (2 * math.tan(self.hfov / 2))
        # 0.3 m = min look-ahead
        ground_angle = math.atan2(camera_height_m, 0.3)
        self._ground_row = int(
            img_height / 2 + focal_len_px * math.tan(ground_angle))
        self._ground_row = min(self._ground_row, img_height)

        rospy.loginfo(
            f"[CV] Detector init: FOV={hfov_deg}°  sectors={n_sectors}  "
            f"active={self._active_mask.sum()}  ground_row={self._ground_row}"
        )

        

    # Lifecycle of MiDaS

    def start(self):
        """Subscribe to the camera topic and load MiDaS if requested."""
        if self._use_midas:
            self._model, self._transform = _load_midas()

        self._sub = rospy.Subscriber(
            self.camera_topic, Image, self._image_callback,
            queue_size=1, buff_size=2 ** 24,
        )
        rospy.loginfo(f"[CV] Subscribed to {self.camera_topic}")

    def stop(self):
        if self._sub is not None:
            self._sub.unregister()
            self._sub = None

    def calibrate(self, lidar_forward_median: float, midas_forward_median: float):
        """
        Compute scale factor: metres = scale * (1 / midas_depth_value).
 
        Call this once after the first step() when both a LiDAR reading and a
        MiDaS frame are available for the forward arc.
 
        Parameters
        ----------
        lidar_forward_median : float
            Median LiDAR distance (metres) for beams within ±30° of forward.
        midas_forward_median : float
            Median MiDaS inverse-depth value for columns within ±30° of centre.
            If MiDaS is not available, pass 1.0 and scale will remain None.
        """
        if midas_forward_median < 1e-6:
            rospy.logwarn(
                "[CV] Calibration skipped: midas_forward_median too small")
            return
        # MiDaS output is inverse depth (disparity), so:
        # metric_depth = scale / midas_value
        self.scale = lidar_forward_median * midas_forward_median
        self._calibrated = True
        rospy.loginfo(
            f"[CV] Calibrated: scale={self.scale:.3f}  "
            f"(lidar={lidar_forward_median:.2f} m, midas={midas_forward_median:.3f})"
        )

    def get_sector_ranges(self) -> np.ndarray:
        """
        Return per-sector minimum estimated distance (metres).
 
        Sectors outside the camera FOV have value np.inf (treated as free
        space when fused with LiDAR).  Sectors inside the FOV but with no
        detected obstacle also return np.inf.
 
        Returns
        -------
        np.ndarray shape (n_sectors,)
        """
        with self._lock:
            return self._sector_ranges.copy()


    def get_veto(self):
        """
        Return (veto_flag, suggested_angular_velocity).
 
        veto_flag : bool
            True if a near obstacle was detected in the forward arc and the
            RL command should be overridden.
        suggested_angular_velocity : float
            Sign and magnitude of the corrective turn (rad/s).
            Positive = turn left (away from obstacle cluster on the right),
            negative = turn right.
        """
        with self._lock:
            return self._veto, self._veto_turn
        

    def get_latest_depth_map(self):
        """Return the most recent MiDaS depth map (for RViz / debugging)."""
        with self._lock:
            return self._latest_depth
        
    
    # Image processing (camera subscriber thread)

    def _image_callback(self, msg: Image):
        """ROS callback - runs in a separate thread."""
        try:
            if cv2 is None:
                return
            bgr = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self._process_frame(bgr)
        except Exception as e:
            rospy.logwarn_throttle(5.0, f"[CV] Image callback error: {e}")


    def _process_frame(self, bgr):
        """Full processing pipeline for one camera frame."""
        if self._model is not None and self._transform is not None:
            depth_map = self._midas_depth(bgr)
        else:
            depth_map = self._edge_depth(bgr)

        sector_ranges, veto, veto_turn = self._depth_to_sectors(depth_map)

        with self._lock:
            self._sector_ranges = sector_ranges
            self._veto = veto
            self._veto_turn = veto_turn
            self._latest_depth = depth_map

    
    def _midas_depth(self, bgr) -> np.ndarray:
        """
        Run MiDaS on the BGR frame and return a metric-depth map (H×W, metres).
 
        If not yet calibrated, returns a relative inverse-depth map in
        arbitrary units (still useful for veto triggering after calibration).
        """
        import torch
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        inp = self._transform(rgb)
        if torch.cuda.is_available():
            inp = inp.cuda()
 
        with torch.no_grad():
            inv_depth = self._model(inp)          # shape (1, H', W') or (H', W')
            inv_depth = torch.nn.functional.interpolate(
                inv_depth.unsqueeze(1) if inv_depth.dim() == 3 else inv_depth,
                size=(self.img_height, self.img_width),
                mode="bicubic",
                align_corners=False,
            ).squeeze().cpu().numpy()
 
        # Avoid division by zero; MiDaS values > 0 by construction
        inv_depth = np.clip(inv_depth, 1e-3, None)
 
        if self._calibrated and self.scale is not None:
            # Convert to metric depth: depth = scale / inv_depth
            depth_m = self.scale / inv_depth
        else:
            # Return raw inverse depth normalised to [0, 10]
            # (treated as "unknown" until calibrate() is called)
            depth_m = (1.0 / inv_depth)
            depth_m = depth_m / (depth_m.max() + 1e-9) * 10.0
 
        return depth_m.astype(np.float32)
    

    def _edge_depth(self, bgr) -> np.ndarray:
        """
        Lightweight fallback: use Canny edges as a proxy for near obstacles.
 
        Strong edges close to the image centre -> likely obstacle close by.
        Returns a pseudo-depth map where edge pixels are assigned a small
        fixed distance and non-edge pixels are assigned np.inf.
 
        This is deliberately conservative: it will raise false-positive
        vetoes but will never miss a real near obstacle with strong edges.
        """
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        # Gaussian blur to suppress noise before edge detection
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, threshold1=50, threshold2=150)

        # Pseudo-depth: edge pixels = 0.5 m (conservative); non-edge = inf
        depth_m = np.where(edges > 0, 0.5, np.inf).astype(np.float32)
        return depth_m
    


    def _depth_to_sectors(self, depth_map: np.ndarray):
        """
        Convert a depth map to per-sector minimum distances and veto signal.
 
        Parameters
        ----------
        depth_map : np.ndarray (H, W)
            Per-pixel depth in metres (or pseudo-depth for edge fallback).
 
        Returns
        -------
        sector_ranges : np.ndarray (n_sectors,)
        veto : bool
        veto_turn : float  rad/s
        """
        # Mask out ground-plane rows (below estimated horizon)
        obstacle_depth = depth_map.copy()
        obstacle_depth[self._ground_row:, :] = np.inf

        sector_ranges = np.full(self.n_sectors, np.inf)

        # For each active column take the minimum depth in that column,
        # then assign it to the corresponding sector
        col_min = np.min(obstacle_depth, axis=0)       # shape (W,)

        for col_idx in range(self.img_width):
            s = self._col_sector[col_idx]
            if self._active_mask[s]:
                d = col_min[col_idx]
                if d < sector_ranges[s]:
                    sector_ranges[s] = d

        # Veto Logic
        # Identify forward sectors (within forward_half_angle of dead-ahead)
        forward_sectors = np.where(
            np.abs(self._sector_angles) <= self.forward_half
        )[0]

        # Filter to active (in-FOV) forward sectors
        forward_active = [s for s in forward_sectors if self._active_mask[s]]

        veto = False
        veto_turn = 0.0

        if forward_active:
            fwd_dists = sector_ranges[forward_active]
            min_fwd = np.nanmin(fwd_dists) if not np.all(
                np.isinf(fwd_dists)) else np.inf

            if min_fwd < self.stop_distance:
                veto = True
                # Decide which way to turn: steer away from the side with
                # the closer obstacle cluster
                left_sectors = [
                    s for s in forward_active if self._sector_angles[s] > 0]
                right_sectors = [
                    s for s in forward_active if self._sector_angles[s] < 0]

                left_min = np.nanmin(
                    sector_ranges[left_sectors]) if left_sectors else np.inf
                right_min = np.nanmin(
                    sector_ranges[right_sectors]) if right_sectors else np.inf

                # Turn away from the closer side
                # (positive angular_z = turn left in ROS convention)
                if left_min < right_min:
                    veto_turn = -0.8   # obstacle on left → turn right
                else:
                    veto_turn = +0.8   # obstacle on right → turn left

                rospy.logwarn_throttle(
                    1.0,
                    f"[CV] VETO — forward min_dist={min_fwd:.2f} m  "
                    f"turn={'RIGHT' if veto_turn < 0 else 'LEFT'}"
                )

        return sector_ranges, veto, veto_turn

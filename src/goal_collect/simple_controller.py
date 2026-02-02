
import math
from typing import List, Optional, Tuple
from dg_commons.sim.models.diff_drive_structures import DiffDriveGeometry, DiffDriveParameters
from .structures import Command, CommandType


class SimpleController:
    def __init__(
        self,
        wheel_radius: float,
        wheelbase: float,
        params: DiffDriveParameters,
        max_v: Optional[float] = None,
        max_omega: Optional[float] = None
    ):
        self.wheel_radius = wheel_radius
        self.wheelbase = wheelbase
        self.params = params
        
        # Calculate limits or use provided
        self.max_omega = max_omega if max_omega is not None else params.omega_limits[1]
        self.max_v = max_v if max_v is not None else (self.max_omega * self.wheel_radius)
        
        # Tolerance for drift check
        self.pos_tolerance = 0.5 # meters
        self.angle_tolerance = 0.5 # radians (~30 deg)

    def get_commands(
        self, 
        current_time: float, 
        current_pose: Tuple[float, float, float], # x, y, theta
        commands_queue: List[Command]
    ) -> Tuple[float, float]:
        """
        Returns (wl, wr) by integrating commands over the next control step (0.1s).
        This allows executing commands with arbitrary durations (e.g. 0.034s).
        """
        
        # Ensure current_time is float
        t_start = float(current_time)
        dt_control = 0.1 # Standard control loop in this sim
        t_end = t_start + dt_control
        
        # Accumulate weighted commands
        total_v = 0.0
        total_w = 0.0
        covered_duration = 0.0
        
        # Add small epsilon to handle float precision at boundaries
        EPSILON = 1e-5
        
        for cmd in commands_queue:
            # Check for overlap between [t_start, t_end] and [cmd.start, cmd.end]
            # Overlap start = max(t_start, cmd.start)
            # Overlap end = min(t_end, cmd.end)
            
            overlap_start = max(t_start, cmd.start_time)
            overlap_end = min(t_end, cmd.end_time)
            
            overlap_dur = overlap_end - overlap_start
            
            if overlap_dur > EPSILON:
                # Extract command values
                if cmd.type == CommandType.STRAIGHT:
                    v_val = cmd.value / cmd.duration if cmd.duration > 1e-6 else 0.0
                    w_val = 0.0
                elif cmd.type == CommandType.TURN:
                    v_val = 0.0
                    w_val = cmd.value / cmd.duration if cmd.duration > 1e-6 else 0.0
                elif cmd.type == CommandType.WAIT:
                    v_val = 0.0
                    w_val = 0.0
                else:
                    v_val = 0.0
                    w_val = 0.0
                    
                total_v += v_val * overlap_dur
                total_w += w_val * overlap_dur
                covered_duration += overlap_dur
                
            # Optimization: If we passed t_end, we can stop searching (assuming sorted queue)
            # But the queue might have past commands? Ideally queue is cleaned or we just skip early ones.
            if cmd.start_time > t_end:
                break
                
        # Calculate average
        # If covered_duration < dt_control, it means we ran out of commands (finished mission?)
        # or gap in plan. We just average over the time we have commands for?
        # Or should we assume 0.0 for the rest? - Defaults to stop.
        
        if covered_duration > 1e-6:
            v_ff = total_v / dt_control
            w_ff = total_w / dt_control
        else:
            v_ff = 0.0
            w_ff = 0.0
            
        # --- Feedback Control ---
        # 1. Calculate Expected Pose at current_time
        # We find the command active at current_time for the "target"
        # If we are between commands (time overlaps multiple), which one determines "position"?
        # We can integrate the plan from t_start to find precise expected pose?
        # Simpler: Interpolate based on the command that covers current_time.
        
        target_cmd = None
        for cmd in commands_queue:
            if cmd.start_time <= t_start <= cmd.end_time:
                target_cmd = cmd
                break
        
        v_fb = 0.0
        w_fb = 0.0
        
        if target_cmd and target_cmd.start_pose and target_cmd.end_pose:
            # Interpolate
            dt_cmd = t_start - target_cmd.start_time
            dur = target_cmd.duration
            if dur > 1e-6:
                alpha_t = max(0.0, min(1.0, dt_cmd / dur))
                
                sx, sy, sth = target_cmd.start_pose
                ex, ey, eth = target_cmd.end_pose
                
                # Linear interp for X, Y (Correct for Straight, approximate for Turn)
                # For TURN, X,Y should stay constant.
                # For STRAIGHT, Theta stays constant.
                # Assuming commands are pure (Type check?)
                
                if target_cmd.type == CommandType.TURN:
                    # Position is start pos (turn in place)
                    tgt_x, tgt_y = sx, sy
                    # Angle interp
                    # Handle wrapping? value is diff.
                    # eth should be sth + value
                    tgt_th = sth + target_cmd.value * alpha_t
                else: # STRAIGHT or WAIT
                    tgt_th = sth
                    tgt_x = sx + alpha_t * (ex - sx)
                    tgt_y = sy + alpha_t * (ey - sy)
                    
                # Calculate Error in Robot Frame
                cx, cy, cth = current_pose
                
                dx = tgt_x - cx
                dy = tgt_y - cy
                
                # Rotate into robot frame
                err_x = math.cos(cth) * dx + math.sin(cth) * dy
                err_y = -math.sin(cth) * dx + math.cos(cth) * dy
                err_th = tgt_th - cth
                while err_th > math.pi: err_th -= 2*math.pi
                while err_th < -math.pi: err_th += 2*math.pi
                
                # Gains
                # Time constant goal: ~0.1-0.2s. 
                # k = 1/tau. k=8 -> tau=0.125s.
                k_x = 8.0
                k_w = 10.0
                k_y = 8.0 # Stronger lateral correction
                
                # Control Law
                # v correction based on longitudinal error
                v_fb = k_x * err_x
                
                # w correction based on angular error + lateral error
                # Use standard "steering" logic: w ~ err_y + err_th
                w_fb = k_w * err_th + k_y * err_y * (1.0 if v_ff >= 0 else -1.0)
                
        # Combine
        v_cmd = v_ff + v_fb
        w_cmd = w_ff + w_fb

        # Clamp inputs to safe body limits (if provided)
        # This prevents the controller from trying to execute commands that exceed physical or safety limits
        v_cmd = max(-self.max_v, min(self.max_v, v_cmd))
        w_cmd = max(-self.max_omega, min(self.max_omega, w_cmd))

        # Calculate wheel velocities
        # dg-commons kinematic model:
        #   v_body = (omega_l + omega_r) * r / 2
        #   w_body = (omega_r - omega_l) * r / L
        # Inverting:
        #   omega_l + omega_r = 2 * v / r
        #   omega_r - omega_l = w * L / r
        # Solving:
        #   omega_l = v/r - w*L/(2*r)
        #   omega_r = v/r + w*L/(2*r)
        L = self.wheelbase
        r = self.wheel_radius
        
        wl = v_cmd / r - w_cmd * L / (2.0 * r)
        wr = v_cmd / r + w_cmd * L / (2.0 * r)
        
        # Clamp to limits
        max_w = self.params.omega_limits[1]
        
        wl = max(-max_w, min(max_w, wl))
        wr = max(-max_w, min(max_w, wr))
        
        return wl, wr


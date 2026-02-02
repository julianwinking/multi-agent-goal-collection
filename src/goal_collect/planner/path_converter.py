
import math
from typing import List, Tuple, Optional
from ..structures import Command, CommandType, Point
from .graph import Graph


def calculate_turn_angle(
    p_prev: Tuple[float, float],
    p_curr: Tuple[float, float],
    p_next: Tuple[float, float]
) -> float:
    """
    Calculate the signed turn angle at p_curr.
    p_prev -> p_curr is the incoming vector.
    p_curr -> p_next is the outgoing vector.
    Returns angle in radians (+ is CCW, - is CW).
    """
    # Vector 1: prev -> curr
    dx1, dy1 = p_curr[0] - p_prev[0], p_curr[1] - p_prev[1]
    
    # Vector 2: curr -> next
    dx2, dy2 = p_next[0] - p_curr[0], p_next[1] - p_curr[1]
    
    angle1 = math.atan2(dy1, dx1)
    angle2 = math.atan2(dy2, dx2)
    
    diff = angle2 - angle1
    
    # Normalize to [-pi, pi]
    while diff > math.pi:
        diff -= 2 * math.pi
    while diff < -math.pi:
        diff += 2 * math.pi
        
    return diff


def path_to_commands(
    graph: Graph,
    path_nodes: List[int],
    arrival_times: List[float],
    max_v: float,
    max_omega: float,
    start_heading: float = 0.0, # Heading at the start of the path
    prev_node_id: Optional[int] = None # Node visited before path[0] for continuity
) -> List[Command]:
    """
    Converts a sequence of nodes and arrival times into a list of executable Commands.
    
    Args:
        graph: The roadmap graph
        path_nodes: List of node IDs
        arrival_times: List of arrival times at each node
        max_v: Maximum linear velocity (m/s)
        max_omega: Maximum angular velocity (rad/s)
        start_heading: Robot's heading at the start (used if prev_node_id is None)
        prev_node_id: The ID of the node visited immediately before path_nodes[0].
                      Used to calculate the turn required to start moving along the path.
    
    Returns:
        List of Command objects.
    """
    if not path_nodes or len(path_nodes) < 2:
        return []

    commands = []
    
    # We need to reconstruct the geometry to calculate turns
    # Nodes: [n0, n1, n2, ...]
    # Times: [t0, t1, t2, ...]
    
    current_heading = start_heading
    if prev_node_id is not None:
        n_prev = graph.nodes[prev_node_id]
        n0 = graph.nodes[path_nodes[0]]
        # Check if prev and n0 are different
        if abs(n0.x - n_prev.x) > 1e-6 or abs(n0.y - n_prev.y) > 1e-6:
             current_heading = math.atan2(n0.y - n_prev.y, n0.x - n_prev.x)
             
    print(f"[PATH_CONVERTER] StartHeading={start_heading:.4f}, PrevNode={prev_node_id}, CurrentHeading={current_heading:.4f}")
    current_time = arrival_times[0]
    
    # 2. Iterate Intervals (Segments)
    for i in range(len(path_nodes) - 1):
        u_id = path_nodes[i]
        v_id = path_nodes[i+1]
        
        u_node = graph.nodes[u_id]
        v_node = graph.nodes[v_id]
        
        t_start_interval = arrival_times[i]
        t_end_interval = arrival_times[i+1]
        t_allocated = t_end_interval - t_start_interval
        
        # A. WAIT Segment (u == v or same position)
        u_equals_v = u_id == v_id
        same_position = abs(u_node.x - v_node.x) < 1e-6 and abs(u_node.y - v_node.y) < 1e-6
        
        if u_equals_v or same_position:
            # Simple Wait (or transition between overlapping nodes)
            cmd = Command(
                type=CommandType.WAIT,
                duration=t_allocated,
                value=0.0,
                start_time=current_time,
                end_time=current_time + t_allocated,
                start_pose=(u_node.x, u_node.y, current_heading),
                end_pose=(u_node.x, u_node.y, current_heading)
            )
            commands.append(cmd)
            current_time += t_allocated
            continue
            
        # B. MOVE Segment (u -> v)
        # 1. Determine Turn Required (Turn i)
        dx, dy = v_node.x - u_node.x, v_node.y - u_node.y
        target_heading = math.atan2(dy, dx)
        
        diff = target_heading - current_heading
        while diff > math.pi: diff -= 2 * math.pi
        while diff < -math.pi: diff += 2 * math.pi
        
        t_turn_phys = 0.0
        if abs(diff) > 1e-4:
            t_turn_phys = abs(diff) / max_omega
            
        # 2. Determine Move Required (Move i)
        dist = math.hypot(dx, dy)
        t_move_phys = dist / max_v
        
        # 3. Time Budgeting
        # Allocated time must cover Turn + Move
        # A* ensures t_allocated >= t_turn + t_move
        
        # Use exact turn duration
        t_turn_final = t_turn_phys
        
        # Assign remaining time to move
        t_move_final = t_allocated - t_turn_final
        
        # Safety Check
        if t_move_final < 0:
             # Should not happen if A* logic is correct
             # Fallback: Compress turn? Or just overflow time (will cause delay)
             print(f"[WARN] Time budget overrun in command gen! Alloc={t_allocated:.3f}, TurnFinal={t_turn_final:.3f}")
             # Prioritize Turn limit safety or Timeline safety?
             # If we cut turn time, we might violate max_omega.
             # If we cut move time (negative), we violate physics (teleport backwards?)
             # Force valid duration
             t_turn_final = t_allocated 
             t_move_final = 0.0
        
        # 4. Generate Turn Command
        if t_turn_final > 1e-6:
             cmd_turn = Command(
                 type=CommandType.TURN,
                 duration=t_turn_final,
                 value=diff,
                 start_time=current_time,
                 end_time=current_time + t_turn_final,
                 start_pose=(u_node.x, u_node.y, current_heading),
                 end_pose=(u_node.x, u_node.y, target_heading)
             )
             commands.append(cmd_turn)
             current_time += t_turn_final
        else:
             # Update heading even if no command (already aligned)
             # But if diff was tiny, maybe we ignore?
             # A* calculated turn penalty 0 for aligned.
             pass
             
        current_heading = target_heading
        
        # 5. Generate Straight Command
        # Even if distance is 0 (should use Wait but u!=v implies dist>0), we execute.
        if t_move_final > 1e-6:
             cmd_move = Command(
                 type=CommandType.STRAIGHT,
                 duration=t_move_final,
                 value=dist,
                 start_time=current_time,
                 end_time=current_time + t_move_final,
                 start_pose=(u_node.x, u_node.y, current_heading),
                 end_pose=(v_node.x, v_node.y, current_heading)
             )
             commands.append(cmd_move)
             current_time += t_move_final
        elif dist > 1e-6:
             # Weird case: Distance exists but no time left?
             # Force command with 0 duration (instant)? Physics will break.
             # This implies t_allocated was == t_turn_final.
             # Using quantization, t_allocated is multiple of 0.1.
             pass

    return commands

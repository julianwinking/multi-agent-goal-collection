import heapq
import math
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
from shapely.geometry import LineString

from .graph import Graph, Node, EnvironmentMap
from ..structures import ReservationTable, Point


def multi_target_dijkstra(graph: Graph, start_id: int, target_ids: Set[int]) -> Dict[int, float]:
    """
    Compute shortest distances from start_id to each node in target_ids using Dijkstra.
    Returns a dictionary mapping target_id -> distance.
    """
    if not target_ids:
        return {}

    dist_map: Dict[int, float] = {}
    # If start is a target, distance is 0
    if start_id in target_ids:
        dist_map[start_id] = 0.0

    remaining = set(target_ids)
    if start_id in remaining:
        remaining.remove(start_id)

    if not remaining and start_id in target_ids:
        return dist_map

    visited: Set[int] = set()
    # Priority queue stores (distance, node_id)
    pq: List[Tuple[float, int]] = [(0.0, start_id)]

    while pq and remaining:
        dist_u, u = heapq.heappop(pq)

        if u in visited:
            continue
        visited.add(u)

        if u in remaining:
            dist_map[u] = dist_u
            remaining.remove(u)
            if not remaining:
                break

        # Expand neighbors
        for v, weight in graph.get_neighbors(u).items():
            if v not in visited:
                new_dist = dist_u + weight
                heapq.heappush(pq, (new_dist, v))

    return dist_map


def compute_all_distances(
    graph: Graph,
    agent_start_nodes: Dict[str, int],
    goal_nodes: Dict[str, int],
    cp_nodes: Dict[str, int],
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    """
    Pre-compute all relevant distances:
    - agent_to_goal[agent][goal] = distance from agent start to goal
    - goal_to_cp[goal][cp] = distance from goal to CP
    - cp_to_goal[cp][goal] = distance from CP to goal
    """
    goal_node_set = set(goal_nodes.values())
    cp_node_set = set(cp_nodes.values())

    # Agent to goals
    agent_to_goal: Dict[str, Dict[str, float]] = {}
    for agent, start_node in agent_start_nodes.items():
        dist_map = multi_target_dijkstra(graph, start_node, goal_node_set)
        agent_to_goal[agent] = {gid: dist_map.get(gnode, math.inf) for gid, gnode in goal_nodes.items()}

    # Goal to CPs
    goal_to_cp: Dict[str, Dict[str, float]] = {}
    for gid, gnode in goal_nodes.items():
        dist_map = multi_target_dijkstra(graph, gnode, cp_node_set)
        goal_to_cp[gid] = {cpid: dist_map.get(cpnode, math.inf) for cpid, cpnode in cp_nodes.items()}

    # TODO: Potentially save this operation by just reversing the previous result
    # CP to goals (for chaining: after delivery, go to next goal)
    cp_to_goal: Dict[str, Dict[str, float]] = {}
    for cpid, cpnode in cp_nodes.items():
        dist_map = multi_target_dijkstra(graph, cpnode, goal_node_set)
        cp_to_goal[cpid] = {gid: dist_map.get(gnode, math.inf) for gid, gnode in goal_nodes.items()}

    return agent_to_goal, goal_to_cp, cp_to_goal


def heuristic(node1: Node, node2: Node) -> float:
    """Euclidean distance heuristic."""
    return np.hypot(node1.x - node2.x, node1.y - node2.y)


def get_transition_time_penalty(
    graph: Graph,
    prev_node_id: Optional[int],
    curr_node_id: int,
    next_node_id: int,
    # params: PlannerParams, # Removed
    max_omega: float = 1.0,
    start_heading: Optional[float] = None,
) -> float:
    """
    Calculates the time penalty for traversing curr_node based on the turning angle.
    Uses physics-based estimation: duration = angle / max_omega.
    """
    import math

    # If no previous node (first move of the mission), check if we have an initial heading
    if prev_node_id is None:
        if start_heading is not None:
             # Calculate turn from start_heading to vector (curr -> next)
             # Note: Start node is curr_node. Robot is AT curr_node doing a turn to face next_node.
             n_curr = graph.nodes[curr_node_id]
             n_next = graph.nodes[next_node_id]
             
             dx = n_next.x - n_curr.x
             dy = n_next.y - n_curr.y
             if math.hypot(dx, dy) < 1e-6:
                 return 0.0
                 
             target_angle = math.atan2(dy, dx)
             diff = target_angle - start_heading
             while diff > math.pi: diff -= 2 * math.pi
             while diff < -math.pi: diff += 2 * math.pi
             
             return abs(diff) / max_omega
        return 0.0

    # If prev_node == curr_node, this means we're continuing from the same node
    # (e.g., at path boundaries between goal→CP or between missions).
    # Use 0.0 penalty since the agent is already there moving continuously.
    if prev_node_id == curr_node_id:
        return 0.0

    # Get nodes
    try:
        n_prev = graph.nodes[prev_node_id]
        n_curr = graph.nodes[curr_node_id]
        n_next = graph.nodes[next_node_id]
    except KeyError:
        # Should not happen if graph is consistent
        return 0.0

    # Calculate vectors
    # v1: prev -> curr
    dx1 = n_curr.x - n_prev.x
    dy1 = n_curr.y - n_prev.y

    # v2: curr -> next
    dx2 = n_next.x - n_curr.x
    dy2 = n_next.y - n_curr.y

    len1 = math.hypot(dx1, dy1)
    len2 = math.hypot(dx2, dy2)

    if len1 < 1e-6 or len2 < 1e-6:
        return 0.0

    # Vector angles
    angle1 = math.atan2(dy1, dx1)
    angle2 = math.atan2(dy2, dx2)
    
    diff = angle2 - angle1
    
    # Normalize to [-pi, pi]
    while diff > math.pi:
        diff -= 2 * math.pi
    while diff < -math.pi:
        diff += 2 * math.pi
        
    # Calculate duration
    if max_omega < 1e-6:
        return 0.0
        
    duration = abs(diff) / max_omega
    return duration


def time_aware_a_star(
    graph: Graph,
    start_node_id: int,
    goal_node_ids: Set[int],
    start_time: float,
    reservations: ReservationTable,
    safe_speed: float,
    agent_name: str,
    return_conflict: bool = False,
    forbidden_nodes: Optional[Set[int]] = None,
    max_v: Optional[float] = None,
    planner_params=None,
    cp_node_ids: Optional[Set[int]] = None,
    prev_node: Optional[int] = None,
    max_omega: float = 1.0, # Added max_omega for physics-based turns
    start_heading: Optional[float] = None,
) -> Tuple[List[int], List[float], float, Optional[Dict[str, Any]]]:
    """
    Space-Time A* search that finds a collision-free path given existing reservations.

    **Distance-Optimized Version**:
    - g-score = distance traveled (NOT time)
    - Heuristic = Euclidean distance to goal
    - Wait action costs a tiny epsilon (0.001) to break ties but is MUCH cheaper than moving
    - Time is tracked separately for reservation collision checks

    Uses transition-specific speed factors if max_v and planner_params are provided.

    Args:
        prev_node: The ID of the node visited immediately before start_node_id.
                   Used to calculate turn penalties for the first move.
                   If None, the first move is treated as a startup (from standstill).
        start_heading: The initial heading of the agent (radians). Used if prev_node is None.
    """

    # Constants for distance-based optimization
    WAIT_COST = 0.1  # Increased from 0.001 to prevent OOM (prefer waiting over moving, but not infinitely)
    WAIT_STEP = 0.2  # Time duration of a single wait action (seconds)

    # Check if we have transition-specific speed parameters
    use_transition_speeds = max_v is not None and planner_params is not None

    # 1. Initialization
    # State now includes previous node ID for transition-aware timing
    if prev_node is not None:
        print(f"[A*] Agent {agent_name} starting at Node {start_node_id} with Prev Node {prev_node}")

    initial_h = min([heuristic(graph.nodes[start_node_id], graph.nodes[gid]) for gid in goal_node_ids])
    # (f, g_dist, time, node, path, times, prev_node_id)
    start_item = (initial_h, 0.0, start_time, start_node_id, [start_node_id], [start_time], prev_node)
    pq = [start_item]

    visited: Dict[Tuple[int, int], float] = {}
    best_conflict = None

    while pq:
        f, g_dist, current_time, u, path, times, current_prev_node = heapq.heappop(pq)

        # 2. Goal Check - return distance as final cost
        if u in goal_node_ids:
            return path, times, current_time, None

        # 3. State Pruning
        # CRITICAL: Time bucket granularity MUST match WAIT_STEP!
        # If WAIT_STEP = 0.2s, then bucket = int(t * 5) gives 0.2s granularity.
        # This ensures each wait action moves to a NEW time bucket, preventing
        # wait states from being incorrectly pruned as "already visited".
        time_bucket = int(current_time / WAIT_STEP)  # Match WAIT_STEP granularity
        state_key = (u, time_bucket)

        if state_key in visited and visited[state_key] <= g_dist:
            continue
        visited[state_key] = g_dist

        u_node = graph.nodes[u]

        # 4. Generate Neighbors (Move & Wait)
        neighbors = list(graph.get_neighbors(u).keys())
        neighbors.append(u)  # Add self for "Wait" action

        for v in neighbors:
            is_wait = u == v

            if is_wait:
                # Wait action: 0 actual distance, small epsilon cost, fixed time step
                move_dist = 0.0
                dist_cost = WAIT_COST  # Increased to prevent OOM
                travel_time = WAIT_STEP
                # Maintain the same previous node because we haven't moved to a new specific node location difference
                # Actually, if we wait at u, the "previous node" logic for the NEXT move from u is tricky.
                # If we consider wait as "staying at u", the geometry relative to the NEXT move remains the same
                # as if we just arrived at u.
                # So we propagate current_prev_node.
                new_prev_node = current_prev_node
            else:
                # Move action: actual distance cost
                v_node = graph.nodes[v]
                curr_dx = v_node.x - u_node.x
                curr_dy = v_node.y - u_node.y
                move_dist = np.hypot(curr_dx, curr_dy)
                dist_cost = move_dist

                # Calculate travel time: base time + turn penalty if available
                if use_transition_speeds:
                    base_travel_time = move_dist / safe_speed
                    # Pass the previous node, current node (u), and next node (v) to calculate angle at u
                    # Note: planner_params removed from call
                    turn_penalty = get_transition_time_penalty(
                        graph, current_prev_node, u, v, max_omega=max_omega, start_heading=start_heading
                    )
                    
                    # Use exact float time
                    t_turn = turn_penalty
                    t_move = base_travel_time

                    travel_time = t_turn + t_move
                    
                    # Debug: Log significant turn penalties
                    if turn_penalty > 1.0:
                        print(
                            f"[A*] {agent_name}: Turn penalty = {turn_penalty:.2f}s"
                        )
                else:
                    travel_time = move_dist / safe_speed

                # Update previous node for next state -> we are now at v, moving from u
                new_prev_node = u

                # Check forbidden nodes
                if forbidden_nodes and v in forbidden_nodes and v not in goal_node_ids:
                    continue

            new_g_dist = g_dist + dist_cost
            new_time = current_time + travel_time

            # Add POI time overhead [REMOVED]
            # if planner_params is not None and not is_wait:
            #     if cp_node_ids is not None and v in cp_node_ids:
            #         new_time += planner_params.astar_cp_time_overhead
            #     elif v in goal_node_ids:
            #         new_time += planner_params.astar_goal_time_overhead

            # 5. Collision Check using TIME (not distance)
            has_conflict = False
            conflict_info = None

            if is_wait:
                # Check if we can stay at node u from current_time to new_time
                if not reservations.is_node_free(u, current_time, new_time, agent_name):
                    has_conflict = True
                    conflict_info = {"type": "node_wait", "blocker": "unknown", "time": current_time}
                # Also check geometric reservations for the wait position (Bug #5 fix)
                elif not reservations.is_segment_safe_geom(
                    Point(x=u_node.x, y=u_node.y), Point(x=u_node.x, y=u_node.y), current_time, new_time, agent_name
                ):
                    has_conflict = True
                    conflict_info = {"type": "geom_wait", "blocker": "unknown", "time": current_time}
            else:
                # Check edge traversal and arrival node
                if not reservations.is_path_free(u, v, current_time, new_time, agent_name, graph):
                    has_conflict = True
                    conflict_info = {"type": "edge_move", "blocker": "unknown", "time": current_time}
                # FIX: Check node arrival with proper dwell time (time_buffer)
                elif not reservations.is_node_free(v, new_time, new_time + reservations.time_buffer, agent_name):
                    has_conflict = True
                    conflict_info = {"type": "node_arrival", "blocker": "unknown", "time": new_time}

            if has_conflict:
                if return_conflict and (best_conflict is None or conflict_info["time"] > best_conflict.get("time", -1)):
                    best_conflict = conflict_info
                continue

            # 6. Calculate heuristic (Euclidean distance, NOT time-based)
            h_cost = 0.0
            if v not in goal_node_ids:
                h_cost = min([heuristic(graph.nodes[v], graph.nodes[gid]) for gid in goal_node_ids])

            new_f = new_g_dist + h_cost

            new_path = list(path)
            new_path.append(v)
            new_times = list(times)
            new_times.append(new_time)

            heapq.heappush(pq, (new_f, new_g_dist, new_time, v, new_path, new_times, new_prev_node))

    # Failure
    return [], [], 0.0, best_conflict


def path_nodes_to_points_with_waits(
    graph: Graph, path_nodes: List[int], arrival_times: List[float], agent_name: str = ""
) -> List[Point]:
    """
    Convert A* output (node IDs + times) to Points with wait_duration.

    Detects when consecutive nodes are the same (wait action) and accumulates wait time.
    Returns list of Points where each Point has x, y, and wait_duration.
    """
    if not path_nodes:
        return []

    points: List[Point] = []
    i = 0

    while i < len(path_nodes):
        node_id = path_nodes[i]
        node = graph.get_node(node_id)

        # Calculate wait duration: sum all consecutive identical nodes
        wait_duration = 0.0
        j = i + 1
        while j < len(path_nodes) and path_nodes[j] == node_id:
            # Wait from arrival_times[j-1] to arrival_times[j]
            wait_duration += arrival_times[j] - arrival_times[j - 1]
            j += 1

        if wait_duration > 0.001 and agent_name:
            print(f"[{agent_name}] Agent will wait for {wait_duration:.2f}s at position {node.x:.2f} {node.y:.2f}")

        points.append(Point(x=node.x, y=node.y, wait_duration=wait_duration))
        i = j

    return points


def get_path_length(graph: Graph, path: List[int]) -> float:
    """Calculates the total length of a path."""
    if not path or len(path) < 2:
        return 0.0

    length = 0.0
    for i in range(len(path) - 1):
        u = path[i]
        v = path[i + 1]
        neighbors = graph.get_neighbors(u)
        if v in neighbors:
            length += neighbors[v]
        else:
            n1, n2 = graph.get_node(u), graph.get_node(v)
            length += np.hypot(n1.x - n2.x, n1.y - n2.y)
    return length


def get_shortest_path(graph: Graph, start_id: int, end_id: int) -> List[Point]:
    """
    Finds the shortest path between two nodes using A*.
    Returns a list of Points.
    """
    if start_id not in graph.nodes or end_id not in graph.nodes:
        return []

    # Priority Queue: (f_score, current_id)
    pq = [(0.0, start_id)]
    came_from = {}
    g_score = {start_id: 0.0}

    end_node = graph.get_node(end_id)

    while pq:
        _, current = heapq.heappop(pq)

        if current == end_id:
            # Reconstruct
            path = []
            while current in came_from:
                n = graph.get_node(current)
                path.append(Point(x=n.x, y=n.y))
                current = came_from[current]
            n = graph.get_node(start_id)
            path.append(Point(x=n.x, y=n.y))
            return path[::-1]

        current_node = graph.get_node(current)

        for neighbor, weight in graph.get_neighbors(current).items():
            tentative_g = g_score[current] + weight
            if tentative_g < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                neighbor_node = graph.get_node(neighbor)
                h = math.hypot(neighbor_node.x - end_node.x, neighbor_node.y - end_node.y)
                heapq.heappush(pq, (tentative_g + h, neighbor))

    return []


def smooth_path_raytracing(path: List[int], graph: Graph, env_map: EnvironmentMap) -> List[int]:
    """
    Smooths a path using ray tracing to remove unnecessary waypoints.

    Args:
        path: List of node IDs representing the path
        graph: The graph containing node information
        env_map: Environment map for collision checking

    Returns:
        Smoothed path as list of node IDs
    """
    if len(path) <= 2:
        return path

    smoothed = [path[0]]  # Start with the first node
    current_idx = 0

    while current_idx < len(path) - 1:
        # Try to find the furthest visible node from current
        furthest_visible = current_idx + 1

        for test_idx in range(len(path) - 1, current_idx, -1):
            if _is_line_collision_free(graph.nodes[path[current_idx]], graph.nodes[path[test_idx]], env_map):
                furthest_visible = test_idx
                break

        # Move to the furthest visible node
        if furthest_visible > current_idx + 1:
            smoothed.append(path[furthest_visible])
            current_idx = furthest_visible
        else:
            # Can only see the next node, add it
            smoothed.append(path[current_idx + 1])
            current_idx += 1

    return smoothed


def smooth_path_with_waits(points: List[Point], graph: Graph, env_map: EnvironmentMap) -> List[Point]:
    """
    Smooths a path with wait times using ray tracing.

    When intermediate nodes are removed, their wait times are accumulated
    onto the first node of the new direct edge.

    Args:
        points: List of Points with wait_duration
        graph: The graph containing node information
        env_map: Environment map for collision checking

    Returns:
        Smoothed path as list of Points with accumulated wait times
    """
    if len(points) <= 2:
        return points

    # Build a lookup: (x, y) -> node_id for collision checking
    coord_to_node = {}
    for node_id, node in graph.nodes.items():
        coord_to_node[(node.x, node.y)] = node_id

    smoothed = [Point(x=points[0].x, y=points[0].y, wait_duration=points[0].wait_duration)]
    current_idx = 0

    while current_idx < len(points) - 1:
        # Try to find the furthest visible point from current
        furthest_visible = current_idx + 1

        current_node_id = coord_to_node.get((points[current_idx].x, points[current_idx].y))
        if current_node_id is None:
            # Fallback: can't smooth if we don't have node mapping
            smoothed.append(
                Point(
                    x=points[current_idx + 1].x,
                    y=points[current_idx + 1].y,
                    wait_duration=points[current_idx + 1].wait_duration,
                )
            )
            current_idx += 1
            continue

        for test_idx in range(len(points) - 1, current_idx, -1):
            test_node_id = coord_to_node.get((points[test_idx].x, points[test_idx].y))
            if test_node_id is not None:
                if _is_line_collision_free(graph.nodes[current_node_id], graph.nodes[test_node_id], env_map):
                    furthest_visible = test_idx
                    break

        # Calculate accumulated wait time from skipped nodes
        accumulated_wait = 0.0
        if furthest_visible > current_idx + 1:
            # We're skipping nodes from current_idx+1 to furthest_visible-1
            for skip_idx in range(current_idx + 1, furthest_visible):
                accumulated_wait += points[skip_idx].wait_duration

        # Add the furthest visible point with accumulated wait
        smoothed.append(
            Point(
                x=points[furthest_visible].x,
                y=points[furthest_visible].y,
                wait_duration=points[furthest_visible].wait_duration + accumulated_wait,
            )
        )
        current_idx = furthest_visible

    return smoothed


def _is_line_collision_free(node_a: Node, node_b: Node, env_map: EnvironmentMap) -> bool:
    """
    Checks if a straight line between two nodes is collision-free.

    Args:
        node_a: Starting node
        node_b: Ending node
        env_map: Environment map for collision checking

    Returns:
        True if the line is collision-free, False otherwise
    """
    line = LineString([(node_a.x, node_a.y), (node_b.x, node_b.y)])

    # Buffer the line by robot radius (obstacles are already buffered in env_map)
    # We just need to ensure the line itself doesn't go outside boundaries

    # Check boundary
    if env_map.boundary_poly and not env_map.boundary_poly.contains(line):
        return False

    # Check obstacles (obstacles in env_map are already inflated by robot_radius)
    for prepared_obs in env_map.prepared_obstacles:
        if prepared_obs.intersects(line):
            return False

    return True

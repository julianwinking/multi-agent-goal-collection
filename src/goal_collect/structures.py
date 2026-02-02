from typing import List, Dict, Tuple, Optional, Set, Union
from enum import Enum
from dataclasses import dataclass, field
from pydantic import BaseModel

from .config import config


class Point(BaseModel):
    x: float
    y: float
    theta: float = 0.0  # Heading/Orientation
    wait_duration: float = 0.0  # seconds to wait at this point



class CommandType(str, Enum):
    STRAIGHT = "STRAIGHT"
    TURN = "TURN"
    WAIT = "WAIT"


class Command(BaseModel):
    type: CommandType
    duration: float  # seconds
    value: float = 0.0 # distance for straight (m), angle for turn (radians, + is CCW)
    start_time: float = 0.0 # Expected start time
    end_time: float = 0.0 # Expected end time
    start_pose: Optional[Tuple[float, float, float]] = None # (x, y, theta) for verification
    end_pose: Optional[Tuple[float, float, float]] = None # (x, y, theta) for verification


class Mission(BaseModel):
    """
    Represents a complete cycle of collecting one goal.
    1. Follow path_to_goal.
    2. Automatically pick up goal_id upon arrival.
    3. Follow path_to_collection.
    4. Automatically drop off goal upon arrival.
    """

    goal_id: str
    path_to_goal: List[Point]
    path_to_collection: List[Point]
    commands: List[Command] = [] # The explicit command sequence to execute

    # Optional timing data for speed factor debugging
    euclidean_distance: Optional[float] = None  # Total straight-line distance (start->goal + goal->cp)
    planned_duration: Optional[float] = None    # Planned travel time (excluding pickup/dropoff/wait)
    used_speed: Optional[float] = None          # Speed used for planning
    
    # Planned ETA for runtime comparison
    planned_eta: Optional[float] = None         # Expected completion time from conflict resolution
    mission_start_time: Optional[float] = None  # Planned start time of this mission


class AgentPlan(BaseModel):
    """The sequence of missions for a specific agent."""

    missions: List[Mission]


class GlobalPlanMessage(BaseModel):
    """The master plan containing schedules for all agents."""

    # Key is the agent's name (PlayerName)
    agent_plans: Dict[str, AgentPlan]


@dataclass
class Interval:
    start: float
    end: float


@dataclass
class AgentState:
    node: int
    time: float
    # Previous node ID. Used for turn penalty continuity.
    # - None means "first move" (startup penalty applies)
    prev_node: Optional[int] = None
    heading: Optional[float] = None


class ReservationTable:
    """
    Manages Space-Time reservations for nodes and edges.
    Used to check availability for Time-Aware A*.
    """

    def __init__(self):
        self.time_buffer = 0.0
        self.safety_dist = 2.0 * config.agent.robot_radius + config.planner.safety_dist_buffer

        # Map NodeID -> List of (Interval, agent_id)
        self.node_reservations: Dict[int, List[Tuple[Interval, str]]] = {}
        # Geometric Reservations: List of (p1, p2, Interval, agent_id)
        self.geometric_reservations: List[Tuple[Point, Point, Interval, str]] = []

    def _add_node_res(self, node_id: int, start: float, end: float, agent_id: str):
        if node_id not in self.node_reservations:
            self.node_reservations[node_id] = []
        # Add buffer
        s = start - self.time_buffer
        e = end + self.time_buffer
        self.node_reservations[node_id].append((Interval(s, e), agent_id))

    def add_path_reservation(self, path_nodes: List[int], arrival_times: List[float], graph, agent_id: str):
        """
        Reserves proper intervals for a sequence of nodes and times.
        path_nodes: [n0, n1, n2...]
        arrival_times: [t0, t1, t2...] matching path_nodes
        """
        for i in range(len(path_nodes)):
            u = path_nodes[i]
            t_u = arrival_times[i]

            # Node Reservation
            if i < len(path_nodes) - 1:
                v = path_nodes[i + 1]
                t_v = arrival_times[i + 1]
                self._add_node_res(u, t_u, t_v, agent_id)

                # Block neighbors within safety distance
                if hasattr(graph, "get_nodes_in_radius"):
                    u_node = graph.nodes[u]
                    neighbors = graph.get_nodes_in_radius(u_node.x, u_node.y, self.safety_dist)
                    for nid in neighbors:
                        if nid != u:
                            self._add_node_res(nid, t_u, t_v, agent_id)

                # Geometric Edge Reservation
                # Add buffer
                s = t_u - self.time_buffer
                e = t_v + self.time_buffer

                n_u = graph.nodes[u]
                n_v = graph.nodes[v]
                p1 = Point(x=n_u.x, y=n_u.y)
                p2 = Point(x=n_v.x, y=n_v.y)

                self.geometric_reservations.append((p1, p2, Interval(s, e), agent_id))

            else:
                # Last node - reserve with proper time buffer
                self._add_node_res(u, t_u, t_u + self.time_buffer, agent_id)
                # Block neighbors for last node
                if hasattr(graph, "get_nodes_in_radius"):
                    u_node = graph.nodes[u]
                    neighbors = graph.get_nodes_in_radius(u_node.x, u_node.y, self.safety_dist)
                    for nid in neighbors:
                        if nid != u:
                            self._add_node_res(nid, t_u, t_u + self.time_buffer, agent_id)

                # Geometric point reservation for last node (prevents edge collisions)
                u_node = graph.nodes[u]
                p = Point(x=u_node.x, y=u_node.y)
                s = t_u - self.time_buffer
                e = t_u + self.time_buffer
                self.geometric_reservations.append((p, p, Interval(s, e), agent_id))

    def reserve_node_forever(self, node_id: int, x: float, y: float, start_time: float, agent_id: str, graph=None):
        """Reserves a node from start_time to infinity (for parking), including geometric blocking."""
        self._add_node_res(node_id, start_time, float("inf"), agent_id)

        # Block neighbors within safety distance
        if graph and hasattr(graph, "get_nodes_in_radius"):
            neighbors = graph.get_nodes_in_radius(x, y, self.safety_dist)
            for nid in neighbors:
                if nid != node_id:
                    self._add_node_res(nid, start_time, float("inf"), agent_id)

        # Add Geometric Reservation (Point)
        p = Point(x=x, y=y)
        self.geometric_reservations.append((p, p, Interval(start_time, float("inf")), agent_id))

    def reserve_point_forever_geom(self, point: Point, start_time: float, agent_id: str):
        """Reserve an arbitrary point (not necessarily a graph node) forever."""
        self.geometric_reservations.append((point, point, Interval(start_time, float("inf")), agent_id))

    def add_geometric_reservation(self, p1: Point, p2: Point, start_time: float, end_time: float, agent_id: str):
        """Add a single geometric reservation segment."""
        self.geometric_reservations.append((p1, p2, Interval(start_time, end_time), agent_id))

    def add_geometric_path_reservation(self, path_points: List[Point], arrival_times: List[float], agent_id: str):
        """Reserve a geometric path described purely by coordinates."""
        if len(path_points) != len(arrival_times):
            raise ValueError("points and times must have same length")
        for i in range(len(path_points) - 1):
            p1 = path_points[i]
            p2 = path_points[i + 1]
            t1 = arrival_times[i]
            t2 = arrival_times[i + 1]
            s = t1 - self.time_buffer
            e = t2 + self.time_buffer
            self.geometric_reservations.append((p1, p2, Interval(s, e), agent_id))
        # Reserve final point momentarily to maintain ordering; caller can extend if needed
        last_point = path_points[-1]
        last_time = arrival_times[-1]
        self.geometric_reservations.append(
            (last_point, last_point, Interval(last_time - self.time_buffer, last_time + self.time_buffer), agent_id)
        )

    def is_node_safe(
        self,
        node_id: int,
        time_point: float,
        agent_id: Optional[str] = None,
        return_owner: bool = False,
    ):
        """
        Check if node is free at specific time, ignoring agent's own reservations.
        When return_owner=True, returns (is_safe, blocking_agent_id).
        """
        if node_id not in self.node_reservations:
            return (True, None) if return_owner else True
        for interval, owner_id in self.node_reservations[node_id]:
            if agent_id is not None and owner_id == agent_id:
                continue
            if interval.start <= time_point <= interval.end:
                return (False, owner_id) if return_owner else False
        return (True, None) if return_owner else True

    def _dist_sq_point_segment(self, p, s1, s2):
        """Squared distance from point p to segment s1-s2"""
        x0, y0 = p.x, p.y
        x1, y1 = s1.x, s1.y
        x2, y2 = s2.x, s2.y

        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            return (x0 - x1) ** 2 + (y0 - y1) ** 2

        t = ((x0 - x1) * dx + (y0 - y1) * dy) / (dx * dx + dy * dy)
        t = max(0, min(1, t))

        closest_x = x1 + t * dx
        closest_y = y1 + t * dy
        return (x0 - closest_x) ** 2 + (y0 - closest_y) ** 2

    def _segments_distance_sq(self, p1, p2, p3, p4):
        """Approximate minimum squared distance between two segments."""

        def ccw(A, B, C):
            return (C.y - A.y) * (B.x - A.x) > (B.y - A.y) * (C.x - A.x)

        if (ccw(p1, p3, p4) != ccw(p2, p3, p4)) and (ccw(p1, p2, p3) != ccw(p1, p2, p4)):
            return 0.0

        d1 = self._dist_sq_point_segment(p1, p3, p4)
        d2 = self._dist_sq_point_segment(p2, p3, p4)
        d3 = self._dist_sq_point_segment(p3, p1, p2)
        d4 = self._dist_sq_point_segment(p4, p1, p2)

        return min(d1, d2, d3, d4)

    def is_edge_safe(
        self,
        u_node,
        v_node,
        t_start: float,
        t_end: float,
        agent_id: Optional[str] = None,
        return_owner: bool = False,
    ):
        """
        Check if edge u-v is usable during [t_start, t_end].
        Checks geometric distance > 2*Radius against all reserved segments NOT owned by agent_id.
        When return_owner=True, returns (is_safe, blocking_agent_id).
        """
        p_start = Point(x=u_node.x, y=u_node.y)
        p_end = Point(x=v_node.x, y=v_node.y)

        for r_p1, r_p2, interval, owner_id in self.geometric_reservations:
            if agent_id is not None and owner_id == agent_id:
                continue

            # Check Time Overlap
            if max(t_start, interval.start) < min(t_end, interval.end):

                # Check Geometric Distance
                dist_sq = self._segments_distance_sq(p_start, p_end, r_p1, r_p2)

                if dist_sq < self.safety_dist**2:
                    return (False, owner_id) if return_owner else False

        return (True, None) if return_owner else True

    def is_segment_safe_geom(
        self,
        p_start: Point,
        p_end: Point,
        t_start: float,
        t_end: float,
        agent_id: Optional[str] = None,
        clearance_override: Optional[float] = None,
        time_relaxation: float = 0.0,
    ) -> bool:
        """Check geometric segment safety without relying on graph nodes."""
        clearance = clearance_override if clearance_override is not None else self.safety_dist
        for r_p1, r_p2, interval, owner_id in self.geometric_reservations:
            if agent_id is not None and owner_id == agent_id:
                continue
            effective_start = interval.start + time_relaxation
            effective_end = interval.end - time_relaxation
            if effective_start > effective_end:
                effective_start = interval.start
                effective_end = interval.end
            if max(t_start, effective_start) < min(t_end, effective_end):
                dist_sq = self._segments_distance_sq(p_start, p_end, r_p1, r_p2)
                if dist_sq < clearance**2:
                    return False
        return True

    def is_point_safe_geom(self, point: Point, time_point: float, agent_id: Optional[str] = None) -> bool:
        """Check if a geometric point is free at the given time."""
        for r_p1, r_p2, interval, owner_id in self.geometric_reservations:
            if agent_id is not None and owner_id == agent_id:
                continue
            if interval.start <= time_point <= interval.end:
                dist_sq = self._segments_distance_sq(point, point, r_p1, r_p2)
                if dist_sq < self.safety_dist**2:
                    return False
        return True

    def remove_reservations_by_agent(self, agent_id: str, after_time: float = 0.0):
        """Remove all reservations for a specific agent that start after a certain time."""
        # Adjust threshold to account for time buffer applied during reservation
        threshold = after_time - self.time_buffer - 0.001

        # Filter Node Reservations
        for nid in list(self.node_reservations.keys()):
            self.node_reservations[nid] = [
                (inv, owner)
                for (inv, owner) in self.node_reservations[nid]
                if not (str(owner) == str(agent_id) and inv.start >= threshold)
            ]
            if not self.node_reservations[nid]:
                del self.node_reservations[nid]

        # Filter Geometric Reservations
        original_count = len(self.geometric_reservations)
        self.geometric_reservations = [
            r for r in self.geometric_reservations if not (str(r[3]) == str(agent_id) and r[2].start >= threshold)
        ]
        removed_count = original_count - len(self.geometric_reservations)
        if removed_count > 0:
            print(
                f"      [ReservationTable] Removed {removed_count} geometric reservations for {agent_id} after t={after_time}"
            )

    def clear_all(self):
        """Clear all reservations. Used when restarting planning with new priority order."""
        self.node_reservations.clear()
        self.geometric_reservations.clear()

    def is_node_free(self, node_id: int, start_time: float, end_time: float, agent_id: str) -> bool:
        """
        Wrapper to check if a node is free during a time INTERVAL.
        Used by time_aware_a_star for wait actions.
        """
        if node_id not in self.node_reservations:
            return True
        
        for interval, owner_id in self.node_reservations[node_id]:
            if owner_id == agent_id:
                continue
            # Check for overlap: max(start1, start2) < min(end1, end2)
            if max(start_time, interval.start) < min(end_time, interval.end):
                return False
        return True

    def is_path_free(self, u_id: int, v_id: int, start_time: float, end_time: float, agent_id: str, graph) -> bool:
        """
        Wrapper to check if an edge is free during a time interval.
        Used by time_aware_a_star for move actions.
        """
        if u_id not in graph.nodes or v_id not in graph.nodes:
            return False
            
        u_node = graph.nodes[u_id]
        v_node = graph.nodes[v_id]
        
        # Use the existing geometric edge checker
        return self.is_edge_safe(u_node, v_node, start_time, end_time, agent_id=agent_id)

@dataclass
class GlobalSolution:
    """Result of Global path optimization."""

    sequences: Dict[str, List[str]]  # agent -> ordered list of goal IDs
    cp_assignments: Dict[str, str]  # goal_id -> cp_id
    estimated_times: Dict[str, Dict[str, float]]  # agent -> {goal_id: arrival_time}
    makespan: float
    total_distance: float
    success: bool
    status: str
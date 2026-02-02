from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import math
from ..structures import AgentPlan, AgentState, Mission, Point, ReservationTable, Command, CommandType
from .graph import Graph, EnvironmentMap, get_poly

from ..config import config



class ParkingManager:
    """Handles parking by spiraling out from the goal until a valid spot is found."""

    def __init__(
        self,
        env_map: EnvironmentMap,
        graph: Graph,
        cp_ids: List[str],
        poi_nodes: Dict[str, List[int]],
        reservations: ReservationTable,
        safe_speed: float,
    ):
        self.env_map = env_map
        self.graph = graph
        self.reservations = reservations
        self.safe_speed = safe_speed
        self.poi_nodes = poi_nodes

        # -- Tuning Parameters --
        self.robot_radius = config.agent.robot_radius
        # Tunnel width for moving to spot
        self.segment_clearance = (2 * self.robot_radius) + 0.1
        # Distance to keep from other parked agents
        self.conflict_clearance = (2 * self.robot_radius) + 0.1
        
        self.time_relax = reservations.time_buffer * 0.8

        # Cache CP Data
        self.node_to_cp: Dict[int, str] = {}
        self.cp_center_points: Dict[str, Point] = {}
        self.cp_radius_map: Dict[str, float] = {}
        self._init_cp_data(cp_ids, poi_nodes)

        self.parking_reservations_set: Set[Tuple[str, int]] = set()

    @staticmethod
    def _get_obj_id(obj):
        return getattr(obj, "id", getattr(obj, "goal_id", getattr(obj, "point_id", None)))

    def _init_cp_data(self, cp_ids: List[str], poi_nodes: Dict[str, List[int]]):
        if self.env_map.collection_points:
            for cp in self.env_map.collection_points:
                raw_id = self._get_obj_id(cp)
                if not raw_id: continue
                poly = get_poly(cp)
                area = poly.area
                r = np.sqrt(area / np.pi) if area > 0 else self.robot_radius
                self.cp_radius_map[str(raw_id)] = max(r, self.robot_radius)

        for cp_id_raw in cp_ids:
            cp_id = str(cp_id_raw)
            nodes = poi_nodes.get(cp_id, [])
            if not nodes: continue
            
            center_node = self.graph.nodes[nodes[0]]
            # FIX: Keyword arguments for Point
            self.cp_center_points[cp_id] = Point(x=center_node.x, y=center_node.y)
            for nid in nodes:
                self.node_to_cp[nid] = cp_id

    def _reserve_stay(self, agent: str, node_id: int, start_time: float) -> Dict:
        """Helper to lock a node forever."""
        node_key = ("node", node_id)
        if node_key in self.parking_reservations_set:
            print(f"[PARKING_CRITICAL] Agent {agent} is overwriting reservation at Node {node_id}!")
        
        n = self.graph.nodes[node_id]
        self.reservations.reserve_node_forever(node_id, n.x, n.y, start_time, agent, self.graph)
        self.parking_reservations_set.add(node_key)
        return {"type": "stay", "node": node_id}

    def _check_spot_validity(
        self, 
        agent: str, 
        start_pt: Point, 
        target_pt: Point, 
        start_time: float, 
        latest_arrival: float,
        future_agents: Set[str]
    ) -> Optional[Dict]:
        """
        Validates if we can move from start_pt -> target_pt and park there safely.
        """
        # 1. Check if the spot itself is geometrically valid
        # Check ALL permanent reservations (past parked agents), 
        # not just future_agents, to prevent adjacent spot selection.
        for p1, p2, interval, owner_id in self.reservations.geometric_reservations:
            if owner_id == agent:
                continue  # Skip self
            
            is_permanent = (interval.end == float('inf'))
            
            if is_permanent:
                # ALWAYS check against already-parked agents (permanent reservations)
                dist_sq = self.reservations._segments_distance_sq(target_pt, target_pt, p1, p2)
                if dist_sq < self.conflict_clearance**2:
                    return None
            else:
                # For non-permanent reservations, only check future agents (original behavior)
                if owner_id not in future_agents: 
                    continue
                if interval.end < start_time: 
                    continue
                
                dist_sq = self.reservations._segments_distance_sq(target_pt, target_pt, p1, p2)
                if dist_sq < self.conflict_clearance**2:
                    return None 

        # 2. Check path feasibility
        if not self.env_map.is_segment_free(start_pt.x, start_pt.y, target_pt.x, target_pt.y):
            return None

        dist = np.hypot(target_pt.x - start_pt.x, target_pt.y - start_pt.y)

        travel_time = (dist / self.safe_speed)
        
        wait_step = 0.1
        max_depart = max(start_time, latest_arrival - travel_time + 5.0) 
        
        depart = start_time
        path_found = False
        
        while depart <= max_depart:
            arrival = depart + travel_time
            if self.reservations.is_segment_safe_geom(
                start_pt, target_pt, depart, arrival, agent,
                clearance_override=self.segment_clearance,
                time_relaxation=self.time_relax
            ):
                if self.reservations.is_point_safe_geom(target_pt, arrival, agent_id=agent):
                    path_found = True
                    break
            depart += wait_step
            
        if not path_found:
            return None

        return {
            "start_pt": start_pt, "end_pt": target_pt,
            "depart": depart, "arrival": arrival
        }

    def assign_parking(
        self,
        agent: str,
        start_node: int,
        start_time: float,
        future_agents: Set[str],
        next_cp_time: Optional[float] = None,
    ) -> Optional[Dict]:
        
        cp_id = self.node_to_cp.get(start_node)
        requires_orbit = (cp_id is not None) and (next_cp_time is not None)

        # FIX: Check if ANYONE else visits this CP in the future (even for drop-off)
        if (cp_id is not None) and (not requires_orbit):
            cp_nodes = self.poi_nodes.get(cp_id, [])
            for nid in cp_nodes:
                if requires_orbit: break
                
                # Check node reservations
                if nid in self.reservations.node_reservations:
                    for interval, owner_id in self.reservations.node_reservations[nid]:
                        if owner_id == agent: continue
                        
                        # If someone else uses this node AFTER we park, we must move.
                        if interval.end > start_time:
                            requires_orbit = True
                            print(f"[PARKING] {agent} forced to orbit: {owner_id} visits {cp_id} (Node {nid}) later.")
                            break

        if not requires_orbit:
            if ("node", start_node) not in self.parking_reservations_set:
                return self._reserve_stay(agent, start_node, start_time)
            print(f"[PARKING] {agent} wanted to stay at {start_node} but it's taken. Searching...")

        # FIX: Keyword arguments for Point
        start_pt = Point(x=self.graph.nodes[start_node].x, y=self.graph.nodes[start_node].y)
        center_pt = self.cp_center_points.get(cp_id, start_pt)
        base_radius = self.cp_radius_map.get(cp_id, max(self.robot_radius, 0.5))
        
        # -- LAYER 1: GEOMETRIC RINGS --
        radii_multipliers = [2.0, 2.4, 2.8, 3.2, 3.6, 4.0] 
        n_angles = 12
        
        for mult in radii_multipliers:
            # FIX: Variable name updated to self.conflict_clearance
            radius = max(base_radius + mult * self.robot_radius, self.conflict_clearance + 0.1)
            
            for i in range(n_angles):
                angle = 2 * np.pi * i / n_angles
                tx = center_pt.x + radius * np.cos(angle)
                ty = center_pt.y + radius * np.sin(angle)
                
                if not self.env_map.is_free(tx, ty): continue
                
                target_pt = Point(x=tx, y=ty)
                
                res = self._check_spot_validity(agent, start_pt, target_pt, start_time, next_cp_time or start_time, future_agents)
                if res:
                    print(f"[PARKING] {agent} found Orbital Spot (R={radius:.2f}m).")
                    pts = [res["start_pt"], res["end_pt"]]
                    tms = [res["depart"], res["arrival"]]
                    self.reservations.add_geometric_path_reservation(pts, tms, agent)
                    self.reservations.reserve_point_forever_geom(target_pt, res["arrival"], agent)
                    return {"type": "orbital", "mission_points": pts}

        # -- LAYER 2: GRAPH NEIGHBORS --
        print(f"[PARKING] {agent} Geometric rings failed. Trying graph neighbors...")
        
        neighbors = list(self.graph.get_neighbors(start_node).keys())
        
        for nid in neighbors:
            if ("node", nid) in self.parking_reservations_set: continue
            
            n_node = self.graph.nodes[nid]
            # FIX: Keyword arguments for Point
            target_pt = Point(x=n_node.x, y=n_node.y)
            
            res = self._check_spot_validity(agent, start_pt, target_pt, start_time, next_cp_time or start_time, future_agents)
            if res:
                 print(f"[PARKING] {agent} Moving to Neighbor Node {nid}.")
                 pts = [res["start_pt"], res["end_pt"]]
                 tms = [res["depart"], res["arrival"]]
                 
                 self.reservations.add_geometric_path_reservation(pts, tms, agent)
                 self.reservations.reserve_node_forever(nid, n_node.x, n_node.y, res["arrival"], agent, self.graph)
                 self.parking_reservations_set.add(("node", nid))
                 
                 return {"type": "orbital", "mission_points": pts}

        msg = f"[PARKING_FATAL] {agent} could not find ANY spot. Force Stay (Collision Risk)."
        print(msg)
        raise ValueError(msg)
        # return self._reserve_stay(agent, start_node, start_time)

    def resolve_parking(self, final_plans, agent_states):
        agents = list(final_plans.keys())
        finish_states = sorted(
            ((ag, agent_states[ag].time, agent_states[ag].node) for ag in agents),
            key=lambda item: item[1],
        )

        for idx, (ag, start_time, start_node) in enumerate(finish_states):
            cp_id = self.node_to_cp.get(start_node)
            future_agents = {fs[0] for fs in finish_states[idx + 1 :]}
            future_same_cp = [fs[1] for fs in finish_states[idx + 1 :] if self.node_to_cp.get(fs[2]) == cp_id]
            next_cp_time = min(future_same_cp) if future_same_cp else None

            result = self.assign_parking(ag, start_node, start_time, future_agents, next_cp_time)

            if result and result.get("type") == "orbital":
                # Define limits for physics calculation
                max_v = self.safe_speed
                max_omega = 4.0 # Default
                
                pts = result["mission_points"]
                start_pt, end_pt = pts[0], pts[1]
                t_curr = result.get("depart", start_time)
                t_arrival = result.get("arrival", start_time)
                
                # Simple "Turn then Straight" logic
                cmds = []
                
                # 1. Geometry
                dx = end_pt.x - start_pt.x
                dy = end_pt.y - start_pt.y
                target_psi = math.atan2(dy, dx)
                dist = math.hypot(dx, dy)
                
                curr_psi = agent_states[ag].heading if agent_states[ag].heading is not None else 0.0
                turn_val = target_psi - curr_psi
                while turn_val > math.pi: turn_val -= 2*math.pi
                while turn_val < -math.pi: turn_val += 2*math.pi
                
                # 2. Timing (Physics-based)
                # Maximize velocity utilization
                
                # Turn Duration
                t_turn = 0.0
                if abs(turn_val) > 1e-4:
                     t_turn_phys = abs(turn_val) / max_omega
                     t_turn = math.ceil(t_turn_phys * 10.0) / 10.0
                     # Ensure at least one tick if minimal?
                     if t_turn < 0.1: t_turn = 0.1

                # Move Duration
                t_move = 0.0
                if dist > 1e-4:
                     t_move_phys = dist / max_v
                     t_move = math.ceil(t_move_phys * 10.0) / 10.0
                     if t_move < 0.1: t_move = 0.1

                # 3. Create Commands
                if t_turn > 0:
                    cmds.append(Command(
                        type=CommandType.TURN,
                        start_time=t_curr,
                        end_time=t_curr + t_turn,
                        duration=t_turn,
                        value=turn_val,
                        start_pose=(start_pt.x, start_pt.y, curr_psi),
                        end_pose=(start_pt.x, start_pt.y, target_psi)
                    ))
                    t_curr += t_turn
                    
                if t_move > 0:
                    cmds.append(Command(
                        type=CommandType.STRAIGHT,
                        start_time=t_curr,
                        end_time=t_curr + t_move,
                        duration=t_move,
                        value=dist,
                        start_pose=(start_pt.x, start_pt.y, target_psi),
                        end_pose=(end_pt.x, end_pt.y, target_psi)
                    ))

                final_plans[ag].missions.append(Mission(
                    goal_id="PARKING", 
                    path_to_goal=pts, 
                    path_to_collection=[],
                    commands=cmds
                ))
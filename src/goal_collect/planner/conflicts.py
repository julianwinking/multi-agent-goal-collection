from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
import math

from ..structures import Mission, AgentPlan, Point, ReservationTable, GlobalSolution, Interval
from ..structures import AgentState
from .graph import Graph, Node, EnvironmentMap
from .path import time_aware_a_star, smooth_path_raytracing, path_nodes_to_points_with_waits, smooth_path_with_waits
from .path_converter import path_to_commands
from ..config import config
from ..utils import plot_spacetime_reservations, animate_spacetime_reservations
from .. import planner_data_store


@dataclass
class MissionAttemptResult:
    """
    Result of a mission attempt for an agent.

    Attributes:
        success: Whether the mission was successfully planned and reserved.
        conflict_agent: The name of the agent responsible for a conflict, if any.
        reason: A description of the reason for failure (e.g., 'goal_path', 'cp_path').
    """

    success: bool
    conflict_agent: Optional[str] = None
    reason: str = ""


class ConflictManager:
    """
    Manages multi-agent path planning by resolving conflicts sequentially.

    It prioritizes agents and plans their missions (Goal -> Checkpoint) while avoiding
    collisions with static obstacles and other agents that have already reserved paths.
    """

    def __init__(
        self,
        graph: Graph,
        reservations: ReservationTable,
        safe_speed: float,
        env_map: EnvironmentMap,
        max_v: float = None,
        max_omega: float = None,
        planner_params=None,
    ):
        self.graph = graph
        self.reservations = reservations
        self.safe_speed = safe_speed
        self.env_map = env_map
        self.max_v = max_v
        self.max_omega = max_omega
        self.planner_params = planner_params

    def resolve_conflicts(
        self,
        global_solution: GlobalSolution,
        agents: List[str],
        agent_states: Dict[str, AgentState],
        goal_nodes: Dict[str, int],
        cp_nodes: Dict[str, int],
        goal_to_cp: Dict[str, Dict[str, float]],
        final_plans: Dict[str, AgentPlan],
    ) -> Set[str]:
        """
        Generates collision-free paths for the missions defined in the global solution.

        This method performs sequential path planning. It prioritizes agents based on
        travel distance and attempts to plan their assigned missions one by one.
        It uses a reservation table to ensure that paths do not conflict with previously
        scheduled agents.

        Args:
            global_solution: The high-level task allocation solution.
            agents: List of agent identifiers.
            agent_states: Current state (position, time) of each agent.
            goal_nodes: Map of goal IDs to graph node IDs.
            cp_nodes: Map of checkpoint IDs to graph node IDs.
            goal_to_cp: Distance matrix from goals to checkpoints.
            final_plans: Dictionary to store the resulting execution plans.

        Returns:
            A set of goal IDs that could not be assigned to any agent.
        """

        cfg = config.milp
        valid_goal_ids = set(goal_nodes.keys())
        pending_goals = set(valid_goal_ids)

        # Pre-compute goal-to-cp options
        goal_cp_options: Dict[str, List[Tuple[str, int, float]]] = {}
        for gid, gnode in goal_nodes.items():
            cp_list = []
            for cpid, cpnode in cp_nodes.items():
                d = goal_to_cp.get(gid, {}).get(cpid, math.inf)
                if not math.isinf(d):
                    cp_list.append((cpid, cpnode, d))
            cp_list.sort(key=lambda x: x[2])
            goal_cp_options[gid] = cp_list[: cfg.max_cp_options]

        goal_area_nodes: Dict[str, Set[int]] = {}
        for gid in valid_goal_ids:
            nodes = self.graph.poi_nodes.get(gid, [])
            goal_area_nodes[gid] = set(nodes) if nodes else set()

        # [Initialization Phase]
        # Identify influence areas around goal nodes to prevent premature congestion.
        # Influence Radius = Agent Radius + POI Buffer
        influence_radius = 2 * config.agent.robot_radius + config.graph.poi_buffer

        # Map goal_id -> Set of "danger" nodes.
        # These nodes are essentially "soft" no-go zones for agents not assigned to this goal,
        # to prevent blocking access to the goal.
        goal_influence_nodes: Dict[str, Set[int]] = {}
        unassigned_goal_nodes: Set[int] = set()

        for gid, center_node_id in goal_nodes.items():
            if center_node_id not in self.graph.nodes:
                continue
            center_node = self.graph.nodes[center_node_id]
            # Get all nodes in influence radius
            danger_nodes = self.graph.get_nodes_in_radius(center_node.x, center_node.y, influence_radius)
            goal_influence_nodes[gid] = set(danger_nodes)
            unassigned_goal_nodes.update(danger_nodes)

        print(f"[DEBUG] Two-Layer Safety Initialized. Total Unassigned Goal Nodes: {len(unassigned_goal_nodes)}")

        all_goal_area_nodes: Set[int] = set()
        for nodes in goal_area_nodes.values():
            all_goal_area_nodes.update(nodes)

        # Track which agents have completed their first pickup (for startup overhead)
        agents_first_pickup_done: Set[str] = set()

        # [Idle Agent Reservation] Reserve nodes around agents with no missions
        # This prevents other agents from planning paths through idle agents' positions
        idle_reservation_radius = config.agent.robot_radius + config.planner.parking_dist_buffer
        idle_agents: List[str] = []

        for agent in agents:
            agent_sequence = global_solution.sequences.get(agent, [])
            # Agent is considered idle if it has no missions or only empty missions
            if not agent_sequence or len(agent_sequence) == 0:
                idle_agents.append(agent)

        if idle_agents:
            print(f"\n[IDLE_AGENT_RESERVATION] Found {len(idle_agents)} idle agents: {idle_agents}")
            for agent in idle_agents:
                # Get agent's starting position
                start_node = agent_states[agent].node
                if start_node not in self.graph.nodes:
                    print(f"  > WARN: Agent {agent} start node {start_node} not in graph, skipping reservation")
                    continue

                agent_node = self.graph.nodes[start_node]
                agent_x, agent_y = agent_node.x, agent_node.y

                # Permanently reserve the start node (handles neighbor blocking and geometric reservation internally)
                self.reservations.reserve_node_forever(start_node, agent_x, agent_y, 0.0, agent, graph=self.graph)

                print(f"  > Reserved start node {start_node} for {agent} at ({agent_x:.2f}, {agent_y:.2f})")

        def execute_mission(
            agent: str, goal_id: str, preferred_cp: Optional[str] = None, is_last_mission: bool = False
        ) -> MissionAttemptResult:
            nonlocal unassigned_goal_nodes
            if goal_id not in goal_nodes:
                return MissionAttemptResult(False, reason="invalid_goal")

            start_node = agent_states[agent].node
            start_time = agent_states[agent].time
            goal_node = goal_nodes[goal_id]

            # [Execution Loop] Masking Logic
            # Construct forbidden_mask: Prevent this agent from traversing near other unassigned goals.
            # We allow the agent to approach its OWN target goal (current_target_nodes).
            current_target_nodes = goal_influence_nodes.get(goal_id, set())
            forbidden_mask = unassigned_goal_nodes - current_target_nodes

            # Safety Exception: If the agent currently plans to start AT a forbidden node
            # (e.g., it was spawned there or finished a previous mission there), we must allow it.
            if start_node in forbidden_mask:
                forbidden_mask.remove(start_node)

            print(f"[DEBUG] Agent {agent} targeting Goal {goal_id}. Masking {len(forbidden_mask)} nodes.")

            # [LEG 1] Path to goal
            # Use prev_node from agent state (None if first mission, or 2nd last node of previous mission)
            current_prev_node = agent_states[agent].prev_node
            print(f"[DEBUG] {agent} Leg 1 (Goal {goal_id}) Start: Node {start_node}, Prev Node: {current_prev_node}")

            # Path to goal
            p_ag, t_ag, time_at_goal, goal_conflict = time_aware_a_star(
                self.graph,
                start_node,
                {goal_node},
                start_time,
                self.reservations,
                self.safe_speed,
                agent,
                return_conflict=True,
                forbidden_nodes=forbidden_mask,
                max_v=self.max_v,
                planner_params=self.planner_params,
                cp_node_ids=set(cp_nodes.values()),
                prev_node=current_prev_node,
                max_omega=self.max_omega,
                start_heading=agent_states[agent].heading,
            )
            if not p_ag:
                blocker = goal_conflict.get("blocker") if goal_conflict else None
                if blocker:
                    print(
                        f"    > BLOCKED: {agent} needs goal {goal_id} but {blocker} holds resource ({goal_conflict.get('type')})"
                    )
                else:
                    print(
                        f"    > WARN: No safe path from {agent} (Node {start_node}) to goal {goal_id} at t={start_time:.2f}"
                    )
                return MissionAttemptResult(False, conflict_agent=blocker, reason="goal_path")

            # [Commit Phase] Leg 1: Pickup Margin
            # Reserve the goal node and its influence area during the pickup operation.
            # This ensures the agent has exclusive access while performing the pickup.

            # first_pickup_overhead: [REMOVED]
            # if agent not in agents_first_pickup_done:
            #     time_at_goal += config.planner.first_pickup_overhead
            #     agents_first_pickup_done.add(agent)

            # mission_overhead: [REMOVED]
            # pickup_time = config.planner.pickup_time + config.planner.mission_overhead
            # pickup_time = config.planner.pickup_time # Deprecated
            pickup_time = 0.0  # Instant pickup/dropoff in simplified model?
            # User said "pickup_time: float = 0.8" in the REMOVE list.
            # So pickup time is 0.0.

            pickup_end_time = time_at_goal + pickup_time

            # Dynamic Reservation (Self) for pickup
            self.reservations._add_node_res(goal_node, time_at_goal, pickup_end_time, agent)

            # Get CP options
            cp_options = list(goal_cp_options.get(goal_id, []))
            if preferred_cp:
                cp_options.sort(key=lambda x: (0 if x[0] == preferred_cp else 1, x[2]))

            cp_blocker: Optional[str] = None
            for cp_id, cp_node, _ in cp_options:
                # [LEG 2] Goal -> Checkpoint (CP)
                # Plan from the end of the pickup duration.
                # Note: Masking is DISABLED for Leg 2 to allow full maneuverability to reach the dropoff.

                # Determine previous node for Leg 2 start (continuity from Leg 1)
                # If we moved to goal (len >= 2), the node before goal is the "previous node" for next move.
                # If we didn't move (len < 2), we keep the previous node from start of Leg 1.
                leg2_prev_node = current_prev_node
                if len(p_ag) >= 2:
                    leg2_prev_node = p_ag[-2]

                print(
                    f"[DEBUG] {agent} Leg 2 (CP {cp_id}) Start: Node {p_ag[-1]}, Prev Node: {leg2_prev_node} (Path Len: {len(p_ag)})"
                )

                p_gc, t_gc, finish_at_cp, cp_conflict = time_aware_a_star(
                    self.graph,
                    goal_node,
                    {cp_node},
                    pickup_end_time,  # Start after pickup
                    self.reservations,
                    self.safe_speed,
                    agent,
                    return_conflict=True,
                    forbidden_nodes=None,  # Safety Layer 1: Correction for Leg 2
                    max_v=self.max_v,
                    planner_params=self.planner_params,
                    cp_node_ids=set(cp_nodes.values()),
                    prev_node=leg2_prev_node,
                    max_omega=self.max_omega,
                )
                if not p_gc:
                    blocker = cp_conflict.get("blocker") if cp_conflict else None
                    if blocker:
                        print(
                            f"      > BLOCKED: {agent} cannot reach CP {cp_id} due to {blocker} ({cp_conflict.get('type')})"
                        )
                        if cp_blocker is None:
                            cp_blocker = blocker
                    else:
                        print(
                            f"      > DEBUG: CP {cp_id} unreachable for {agent} from Goal {goal_id} at t={pickup_end_time:.2f}"
                        )
                    continue

                # Reserve paths
                self.reservations.add_path_reservation(p_ag, t_ag, self.graph, agent)

                # Manual reservation for Pickup Wait
                # Note: p_ag ends at time_at_goal. p_gc starts at pickup_end_time.
                # We need to reserve the gap [time_at_goal, pickup_end_time] at goal_node.
                # add_path_reservation doesn't cover the gap automatically if we treat them as separate paths.

                # 1. NODE RESERVATION
                g_n = self.graph.nodes[goal_node]
                for nid in self.graph.get_nodes_in_radius(g_n.x, g_n.y, self.reservations.safety_dist):
                    self.reservations._add_node_res(nid, time_at_goal, pickup_end_time, agent)

                # 2. GEOMETRIC RESERVATION (Blocks edge physics check)
                g_p = Point(x=g_n.x, y=g_n.y)
                self.reservations.geometric_reservations.append(
                    (g_p, g_p, Interval(time_at_goal, pickup_end_time), agent)
                )

                self.reservations.add_path_reservation(p_gc, t_gc, self.graph, agent)

                # dropoff_time: [REMOVED]
                dropoff_time = 0.0
                dropoff_end_time = finish_at_cp + dropoff_time

                # 1. NODE RESERVATION
                cp_n = self.graph.nodes[cp_node]
                for nid in self.graph.get_nodes_in_radius(cp_n.x, cp_n.y, self.reservations.safety_dist):
                    self.reservations._add_node_res(nid, finish_at_cp, dropoff_end_time, agent)

                # 2. GEOMETRIC RESERVATION
                # Reserve the geometric space (Point) of the checkpoint node.
                # This prevents other agents from traversing the edge crossing this point during the wait.
                cp_p = Point(x=cp_n.x, y=cp_n.y)
                self.reservations.geometric_reservations.append(
                    (cp_p, cp_p, Interval(finish_at_cp, dropoff_end_time), agent)
                )

                # [Parking Buffer Logic]
                # If this is the last mission, extend the reservation at the drop-off location
                # to give the agent time to transition to parking logic safely.
                if is_last_mission:
                    buffer_time = config.planner.parking_buffer_time
                    buffer_end_time = dropoff_end_time + buffer_time

                    # Extended Spatial Reservation (2 * Radius + Safety Margin)
                    extended_radius = 2 * config.agent.robot_radius + config.planner.safety_dist_buffer

                    # --- VALIDITY CHECK START ---
                    # Before applying this extended reservation, we MUST check if it conflicts
                    # with any existing reservations (since we might be a lower-priority agent planning later).

                    # 1. Check Node Conflicts
                    radius_nodes = self.graph.get_nodes_in_radius(cp_n.x, cp_n.y, extended_radius)
                    for nid in radius_nodes:
                        if not self.reservations.is_node_free(nid, dropoff_end_time, buffer_end_time, agent):
                            print(
                                f"      > FAIL: Parking buffer conflict at Node {nid} (t={dropoff_end_time:.2f}-{buffer_end_time:.2f})"
                            )
                            return MissionAttemptResult(False, reason="parking_buffer_node_conflict")

                    # 2. Check Geometric Conflicts
                    # Use is_segment_safe_geom with the extended radius as clearance.
                    # This ensures no other agent's path cuts through our extended buffer zone.
                    if not self.reservations.is_segment_safe_geom(
                        cp_p,
                        cp_p,
                        dropoff_end_time,
                        buffer_end_time,
                        agent_id=agent,
                        clearance_override=extended_radius,
                    ):
                        print(
                            f"      > FAIL: Parking buffer geometric conflict (t={dropoff_end_time:.2f}-{buffer_end_time:.2f})"
                        )
                        return MissionAttemptResult(False, reason="parking_buffer_geom_conflict")
                    # --- VALIDITY CHECK END ---

                    # If valid, apply reservations

                    # 1. NODE RESERVATION (Extended)
                    for nid in radius_nodes:
                        self.reservations._add_node_res(nid, dropoff_end_time, buffer_end_time, agent)

                    # 2. GEOMETRIC RESERVATION (Extended Time)
                    # Maintain the geometric block at the center point.
                    # Note: We reserve the point itself. The 'clearance' protection for others comes
                    # from checking against this point with safety_dist.
                    # Implicitly, by checking is_segment_safe_geom above with extended_radius, we proved it's safe.
                    # Now we reserve it so future agents (lower priority) respect it.
                    self.reservations.geometric_reservations.append(
                        (cp_p, cp_p, Interval(dropoff_end_time, buffer_end_time), agent)
                    )

                    print(
                        f"      [DEBUG] Parking Buffer Applied: Reserved CP area (r={extended_radius:.2f}m) until t={buffer_end_time:.2f}s"
                    )

                # Convert to points with wait detection
                pts_ag = path_nodes_to_points_with_waits(self.graph, p_ag, t_ag, agent_name=agent)
                pts_gc = path_nodes_to_points_with_waits(self.graph, p_gc, t_gc, agent_name=agent)

                # Speed factor analysis - compute timing data
                # Euclidean distance: start -> goal + goal -> cp
                start_n = self.graph.nodes[start_node]
                goal_n = self.graph.nodes[goal_node]
                cp_n_coords = self.graph.nodes[cp_node]
                euclidean_to_goal = math.hypot(goal_n.x - start_n.x, goal_n.y - start_n.y)
                euclidean_to_cp = math.hypot(cp_n_coords.x - goal_n.x, cp_n_coords.y - goal_n.y)
                total_euclidean = euclidean_to_goal + euclidean_to_cp

                # Calculate total wait time from path points
                wait_time_to_goal = sum(p.wait_duration for p in pts_ag if p.wait_duration > 0)
                wait_time_to_cp = sum(p.wait_duration for p in pts_gc if p.wait_duration > 0)
                total_wait_time = wait_time_to_goal + wait_time_to_cp

                # Planned duration: travel time only (excluding pickup/dropoff AND wait times)
                travel_to_goal = time_at_goal - start_time
                travel_to_cp = finish_at_cp - pickup_end_time
                planned_travel_time = (travel_to_goal + travel_to_cp) - total_wait_time

                # Compute optimal speed factor if debug enabled
                if config.debug_speed_factor and planned_travel_time > 0:
                    pass  # Debug removed as safe_speed_factor is deprecated

                # Smooth paths
                # Note: Path smoothing is currently handled after path finding if enabled.
                # The commented out lines below were for deprecated smoothing logic.

                # Record planned path for distance error plotting
                if config.debug_speed_factor:
                    # Record path to goal (Leg 1)
                    for i, node_id in enumerate(p_ag):
                        node = self.graph.nodes[node_id]
                        is_last = i == len(p_ag) - 1
                        planner_data_store.record_planned_path_point(
                            agent_name=agent,
                            time=t_ag[i],
                            x=node.x,
                            y=node.y,
                            is_poi=is_last,
                            poi_type="goal" if is_last else "",
                        )

                    # Record path to collection (Leg 2)
                    for i, node_id in enumerate(p_gc):
                        node = self.graph.nodes[node_id]
                        is_last = i == len(p_gc) - 1
                        planner_data_store.record_planned_path_point(
                            agent_name=agent,
                            time=t_gc[i],
                            x=node.x,
                            y=node.y,
                            is_poi=is_last,
                            poi_type="cp" if is_last else "",
                        )

                # Generate Command List
                # Combined Commands = Leg 1 Commands + [Pickup Wait] + Leg 2 Commands + [Dropoff Wait]

                # Leg 1 Commands
                cmds_ag = path_to_commands(
                    graph=self.graph,
                    path_nodes=p_ag,
                    arrival_times=t_ag,
                    max_v=self.max_v,
                    max_omega=self.max_omega,
                    start_heading=agent_states[agent].heading,
                    prev_node_id=current_prev_node,
                )

                # Pickup Wait
                pickup_wait_cmd = []
                if cmds_ag:
                    last_cmd = cmds_ag[-1]
                    end_time_leg1 = last_cmd.end_time
                    end_pose_leg1 = last_cmd.end_pose
                    if end_pose_leg1:
                        agent_states[agent].heading = end_pose_leg1[2]
                else:
                    # Should not happen if path found
                    end_time_leg1 = time_at_goal
                    end_pose_leg1 = None  # TODO: Fill with goal pose?

                pickup_duration = pickup_end_time - time_at_goal
                # Wait command at goal
                from ..structures import Command, CommandType

                # Construct Wait Command
                pickup_cmd = Command(
                    type=CommandType.WAIT,
                    duration=pickup_duration,
                    value=0.0,
                    start_time=time_at_goal,
                    end_time=pickup_end_time,
                    start_pose=end_pose_leg1,
                    end_pose=end_pose_leg1,
                )

                # Leg 2 Commands
                cmds_gc = path_to_commands(
                    graph=self.graph,
                    path_nodes=p_gc,
                    arrival_times=t_gc,
                    max_v=self.max_v,
                    max_omega=self.max_omega,
                    start_heading=agent_states[agent].heading,
                    prev_node_id=leg2_prev_node,
                )

                # Dropoff Wait (optional, purely for alignment with planner dropoff_time)
                # Mission ends at dropoff_end_time, path ends at finish_at_cp
                dropoff_duration = dropoff_end_time - finish_at_cp
                dropoff_cmd = []
                if cmds_gc:
                    last_cmd_gc = cmds_gc[-1]
                    end_pose_gc = last_cmd_gc.end_pose
                else:
                    end_pose_gc = end_pose_leg1  # If Leg 2 empty (start=goal=cp?)

                if dropoff_duration > 0.001:
                    d_cmd = Command(
                        type=CommandType.WAIT,
                        duration=dropoff_duration,
                        value=0.0,
                        start_time=finish_at_cp,
                        end_time=dropoff_end_time,
                        start_pose=end_pose_gc,
                        end_pose=end_pose_gc,
                    )
                    dropoff_cmd = [d_cmd]

                # Update state for next mission
                if cmds_gc:
                    last_cmd_gc = cmds_gc[-1]
                    if last_cmd_gc.end_pose:
                        agent_states[agent].heading = last_cmd_gc.end_pose[2]

                final_plans[agent].missions.append(
                    Mission(
                        goal_id=goal_id,
                        path_to_goal=pts_ag,
                        path_to_collection=pts_gc,
                        commands=cmds_ag + [pickup_cmd] + cmds_gc + dropoff_cmd,
                        euclidean_distance=total_euclidean,
                        planned_duration=planned_travel_time,
                        used_speed=self.safe_speed,
                        planned_eta=dropoff_end_time,
                        mission_start_time=start_time,
                    )
                )

                agent_states[agent].node = cp_node
                agent_states[agent].time = dropoff_end_time

                # Update prev_node for next mission continuity
                # If we moved to CP (len >= 2), the node before CP is the "previous node" for next mission.
                # If we didn't move (len < 2), we keep the prev_node from start of Leg 2.
                if len(p_gc) >= 2:
                    agent_states[agent].prev_node = p_gc[-2]
                else:
                    agent_states[agent].prev_node = leg2_prev_node

                print(f"[DEBUG] {agent} Mission End. Updated Prev Node: {agent_states[agent].prev_node}")

                pending_goals.discard(goal_id)

                # [Commit Phase] Finalizing Assignment
                # Since the goal is now assigned to this agent, we remove it from the "unassigned" set.
                # This allows other agents to traverse this area if necessary, provided they respect reservations.
                if goal_id in goal_influence_nodes:
                    removed_nodes = goal_influence_nodes[goal_id]
                    unassigned_goal_nodes -= removed_nodes

                    # Add Static Reservation for target nodes (t=0 to arrival)
                    # This ensures no OTHER agent plans a path through this goal location
                    # before the current agent arrives to pick it up.
                    for nid in removed_nodes:
                        self.reservations._add_node_res(nid, 0.0, time_at_goal, agent)

                    print(
                        f"[DEBUG] Goal {goal_id} COMMITTED. Removed from Unassigned Set. Reserved static obstacle from t=0.0 to t={time_at_goal:.2f}."
                    )
                    print(
                        f"[DEBUG] Remaining Unassigned Goals: {len(unassigned_goal_nodes)} nodes (Leg 1 masking active for others)."
                    )
                    print(f"[DEBUG] Leg 2 Safety: Masking disabled for {agent} (Full Capacity).")

                print(f"    >>> {agent}: Goal {goal_id} -> CP {cp_id} (ETA: {dropoff_end_time:.2f}s)")
                return MissionAttemptResult(True)

            if cp_blocker:
                return MissionAttemptResult(False, conflict_agent=cp_blocker, reason="cp_path")

            print(f"    > WARN: No feasible CP route for {goal_id} assigned to {agent}")
            return MissionAttemptResult(False, reason="cp_unreachable")

        # Prioritize agents by total travel distance
        agent_distances = []
        for agent in agents:
            total_dist = 0.0
            goals = global_solution.sequences.get(agent, [])
            if not goals:
                agent_distances.append((0.0, agent))
                continue

            curr_node = agent_states[agent].node
            for gid in goals:
                # To Goal
                if gid in goal_nodes:
                    g_node = goal_nodes[gid]
                    # Estimate distance (Euclidean or Graph)
                    # Using Euclidean for speed, or pre-computed if available
                    # We have goal_to_cp, but not arbitrary node to goal easily without lookup
                    # Let's use Euclidean heuristic
                    n1 = self.graph.nodes[curr_node]
                    n2 = self.graph.nodes[g_node]
                    total_dist += math.hypot(n1.x - n2.x, n1.y - n2.y)
                    curr_node = g_node

                # To CP
                cpid = global_solution.cp_assignments.get(gid)
                if cpid and cpid in cp_nodes:
                    cp_node = cp_nodes[cpid]
                    n1 = self.graph.nodes[curr_node]
                    n2 = self.graph.nodes[cp_node]
                    total_dist += math.hypot(n1.x - n2.x, n1.y - n2.y)
                    curr_node = cp_node

            agent_distances.append((total_dist, agent))

        # Sort descending
        agent_distances.sort(key=lambda x: x[0], reverse=True)
        sorted_agents = [a for d, a in agent_distances]

        print("\n  [ConflictManager] Agent Priority (Distance):")
        for d, a in agent_distances:
            print(f"    - {a}: {d:.2f}m")

        # Save initial states for potential retry (including prev_node for turn continuity)
        # Fix: Must capture heading too!
        initial_agent_states = {
            agent: AgentState(node=s.node, time=s.time, heading=s.heading, prev_node=s.prev_node)
            for agent, s in agent_states.items()
        }
        initial_unassigned_goal_nodes = set(unassigned_goal_nodes)

        # Priority reshuffling retry logic
        MAX_PRIORITY_RETRIES = 20
        # Initialize with sorted agents (distance-based)
        current_priority = list(sorted_agents)
        retry_count = 0

        while retry_count < MAX_PRIORITY_RETRIES:
            if retry_count > 0:
                print(f"\n  [PRIORITY_RESHUFFLE] Retry {retry_count}/{MAX_PRIORITY_RETRIES}. New priority order:")
                for idx, a in enumerate(current_priority):
                    print(f"    {idx+1}. {a}")

            # Reset state for this attempt
            self.reservations.clear_all()

            # Re-apply idle agent reservations after clearing
            for idle_agent in idle_agents:
                start_node = initial_agent_states[idle_agent].node
                if start_node not in self.graph.nodes:
                    continue
                agent_node = self.graph.nodes[start_node]
                agent_x, agent_y = agent_node.x, agent_node.y
                self.reservations.reserve_node_forever(start_node, agent_x, agent_y, 0.0, idle_agent, graph=self.graph)

            for agent in agents:
                agent_states[agent].node = initial_agent_states[agent].node
                agent_states[agent].time = initial_agent_states[agent].time
                agent_states[agent].heading = initial_agent_states[agent].heading
                agent_states[agent].prev_node = initial_agent_states[agent].prev_node
                final_plans[agent].missions.clear()
            unassigned_goal_nodes.clear()
            unassigned_goal_nodes.update(initial_unassigned_goal_nodes)
            pending_goals = set(valid_goal_ids)

            # Track which agent fails in this attempt
            failed_agent_this_attempt: Optional[str] = None
            failed_goal_this_attempt: Optional[str] = None

            # Execute missions with current priority
            for agent in current_priority:
                sequence = list(global_solution.sequences.get(agent, []))
                if not sequence:
                    continue

                # Clear any pre-existing reservations for this agent.
                # This is critical for replanning: we clear the slate for the current agent
                # before generating its new plan, while keeping other agents' reservations intact.
                self.reservations.remove_reservations_by_agent(agent)

                # Execute sequence
                for goal_id in sequence:
                    preferred_cp = global_solution.cp_assignments.get(goal_id)
                    # Determine if this is the last mission in the sequence
                    is_last = goal_id == sequence[-1]
                    result = execute_mission(agent, goal_id, preferred_cp=preferred_cp, is_last_mission=is_last)

                    if not result.success:
                        print(f"    > FAIL: Agent {agent} stopped at goal {goal_id} (Reason: {result.reason})")
                        failed_agent_this_attempt = agent
                        failed_goal_this_attempt = goal_id
                        break

                if failed_agent_this_attempt:
                    break

            # Check if we need to retry
            if failed_agent_this_attempt is None:
                # Success! All agents planned successfully
                print(f"\n  [ConflictManager] All agents planned successfully!")
                break

            # Modify priority for next attempt based on failure
            fail_idx = current_priority.index(failed_agent_this_attempt)

            if fail_idx == 0:
                # Agent is already at the top and still failed.
                # Enter Backup Mode (drop goals)
                print(f"\n  [BACKUP_MODE] Agent {failed_agent_this_attempt} failed again at top priority.")

                # Find the index of the failed goal and remove it and all subsequent goals
                agent_sequence = list(global_solution.sequences.get(failed_agent_this_attempt, []))
                if failed_goal_this_attempt in agent_sequence:
                    f_g_idx = agent_sequence.index(failed_goal_this_attempt)
                    removed_goals = agent_sequence[f_g_idx:]
                    remaining_goals = agent_sequence[:f_g_idx]

                    # Update global solution
                    global_solution.sequences[failed_agent_this_attempt] = remaining_goals

                    # Add removed goals back to pending
                    for gid in removed_goals:
                        pending_goals.add(gid)
                        print(f"    > Removed goal {gid} from {failed_agent_this_attempt}, marked as pending")

            else:
                # Bump up by one spot
                prev_agent = current_priority[fail_idx - 1]
                current_priority[fail_idx - 1] = failed_agent_this_attempt
                current_priority[fail_idx] = prev_agent
                print(
                    f"\n  [PRIORITY_RESHUFFLE] Swapped {failed_agent_this_attempt} with {prev_agent}. Moving up to pos {fail_idx}."
                )

            retry_count += 1

        if pending_goals:
            msg = f"CRITICAL: {len(pending_goals)} goals remain unassigned!"
            print(f"  > {msg}")
            raise RuntimeError(msg)

        # Plot reservations for debugging
        if config.advanced_plotting:
            import os

            output_path = os.path.join(config.plots_dir, "reservations_3d.png")
            plot_spacetime_reservations(self.graph, self.reservations, output_path)

            # Generate animated version
            animated_output = os.path.join(config.plots_dir, "reservations_3d_animated.gif")
            animate_spacetime_reservations(
                self.graph,
                self.reservations,
                animated_output,
                max_time=30.0,
                fps=20,  # Reduced from 30
                duration_seconds=6,  # Reduced from 8
            )

        return pending_goals

from typing import Dict, List, Optional

from ..structures import AgentPlan, AgentState, Point, ReservationTable
from .graph import Graph, EnvironmentMap, generate_occupancy_grid, generate_prm, generate_inflated_visibility_graph
from .path import compute_all_distances
from .milp import solve_milp_vrp
from .regret_insertion import solve_regret_insertion
from .optimizer import solve_iterative_greedy
from .conflicts import ConflictManager
from .parking import ParkingManager
from ..config import config
from ..utils import plot_env_graph, plot_all_plans, generate_intermediate_plans
from .. import planner_data_store


class FleetPlanner:
    """
    Hiearachical global planner:
    - Build roadmap (static obstacles)
    - Compute shortest-path distances
    - Assign and order goals (MILP/greedy)
    - Generate collision-free space-time paths
    - Assign parking at the end
    """

    def __init__(self):
        self.graph: Optional[Graph] = None
        self.env_map: Optional[EnvironmentMap] = None
        self.reservations: Optional[ReservationTable] = None

    def plan(
        self,
        agent_starts: Dict[str, Point],
        goals: List[object],
        collection_points: List[object],
        static_obstacles: List[object],
        max_v: float,
        max_omega: float,
        num_agents: int,
    ) -> Dict[str, AgentPlan]:
        plotting = config.enable_plotting
        print("=" * 60)
        print("STARTING FLEET PLANNER")
        print(f"  Agents: {list(agent_starts.keys())}")
        print(f"  Goals: {len(goals)}")
        print(f"  Collection Points: {len(collection_points)}")
        print("=" * 60)

        print("\n--- PHASE 0: Environment Setup ---")
        start_positions = [(p.x, p.y) for p in agent_starts.values()]
        self._initialize_environment(static_obstacles, goals, collection_points, start_positions)
        if self.graph is None or self.env_map is None:
            raise RuntimeError("Environment initialization failed (graph/env_map is None)")
        graph = self.graph
        env_map = self.env_map

        if plotting:
            plot_env_graph(
                static_obstacles,
                goals,
                collection_points,
                agent_starts,
                graph,
                output_file=f"{config.plots_dir}/env_graph.png",
            )

        goal_ids = self._extract_ids(goals)
        cp_ids = self._extract_ids(collection_points)

        safe_speed = max_v
        print(f"  Config: safe_speed={safe_speed:.2f}")

        reservations = ReservationTable()
        self.reservations = reservations
        # Populate global data store for visualization

        agents = list(agent_starts.keys())
        agent_states: Dict[str, AgentState] = {}
        final_plans = {name: AgentPlan(missions=[]) for name in agents}

        agent_start_nodes: Dict[str, int] = {}
        for name, pt in agent_starts.items():
            node_id = graph.get_closest_node(pt.x, pt.y)
            print(f"[DEBUG] Planner mapped {name} at ({pt.x:.2f}, {pt.y:.2f}, theta={pt.theta:.2f}) to Node {node_id}")
            # Use actual initial heading
            agent_states[name] = AgentState(node=node_id, time=0.0, heading=pt.theta)
            agent_start_nodes[name] = node_id

        # Reserve the start position so time-aware planning cannot route another agent through it at t=0.
            start_node = graph.nodes[node_id]
            reservations.reserve_node_forever(node_id, start_node.x, start_node.y, 0.0, name, graph=graph)

        goal_nodes = self._extract_poi_center_nodes(goal_ids)
        cp_nodes = self._extract_poi_center_nodes(cp_ids)

        print("\n--- PHASE 1: Computing Distances ---")
        agent_to_goal, goal_to_cp, cp_to_goal = compute_all_distances(graph, agent_start_nodes, goal_nodes, cp_nodes)

        print("\n--- PHASE 2: Global Optimization ---")
        if num_agents <= 2:
            solver_method = "REGRET"
        else:
            solver_method = "GREEDY"
        print(f"  > Solver Method: {solver_method} (selected based on {num_agents} agent(s))")

        if solver_method == "MILP":
            global_solution = solve_milp_vrp(
                agents=agents,
                goal_ids=goal_ids,
                cp_ids=cp_ids,
                agent_to_goal=agent_to_goal,
                goal_to_cp=goal_to_cp,
                cp_to_goal=cp_to_goal,
                safe_speed=safe_speed,
            )

        elif solver_method == "REGRET":
            # Extract Goal coords for spatial conflict check
            goal_coords = {}
            for gid, nid in goal_nodes.items():
                node = graph.nodes[nid]
                goal_coords[gid] = (node.x, node.y)

            global_solution = solve_regret_insertion(
                agents=agents,
                goal_ids=goal_ids,
                cp_ids=cp_ids,
                agent_to_goal=agent_to_goal,
                goal_to_cp=goal_to_cp,
                cp_to_goal=cp_to_goal,
                safe_speed=safe_speed,
                goal_coords=goal_coords,
            )

        elif solver_method == "GREEDY":
            global_solution = solve_iterative_greedy(
                agents=agents,
                goal_ids=goal_ids,
                cp_ids=cp_ids,
                agent_to_goal=agent_to_goal,
                goal_to_cp=goal_to_cp,
                cp_to_goal=cp_to_goal,
                safe_speed=safe_speed,
            )

        else:
            raise ValueError(f"Unknown solver method: {solver_method}. Options: MILP, REGRET, GREEDY")

        if not global_solution.success:
            print(f"  > Global optimization failed: {global_solution.status}")

        if plotting:
            intermediate_plans = generate_intermediate_plans(
                agents, agent_start_nodes, global_solution, goal_nodes, cp_nodes, graph
            )

            plot_all_plans(
                static_obstacles,
                goals,
                collection_points,
                agent_starts,
                intermediate_plans,
                graph,
                output_file=f"{config.plots_dir}/global_plan.png",
            )

        print("\n--- PHASE 3: Conflict Resolution & Execution ---")
        conflict_manager = ConflictManager(
            graph, reservations, safe_speed, env_map,
            max_v=max_v, max_omega=max_omega, planner_params=config.planner
        )

        unassigned_goals = conflict_manager.resolve_conflicts(
            global_solution=global_solution,
            agents=agents,
            agent_states=agent_states,
            goal_nodes=goal_nodes,
            cp_nodes=cp_nodes,
            goal_to_cp=goal_to_cp,
            final_plans=final_plans,
        )

        if unassigned_goals:
            print(f"  > CRITICAL: {len(unassigned_goals)} goals remain unassigned!")

        print("\n--- PHASE 4: Safe Parking ---")
        parking_mgr = ParkingManager(
            env_map=env_map,
            graph=graph,
            cp_ids=cp_ids,
            poi_nodes=graph.poi_nodes,
            reservations=reservations,
            safe_speed=safe_speed,
        )

        parking_mgr.resolve_parking(final_plans, agent_states)

        if plotting:
            plot_all_plans(
                static_obstacles,
                goals,
                collection_points,
                agent_starts,
                final_plans,
                graph,
                output_file=f"{config.plots_dir}/final_plan.png",
            )


        print("=" * 60)
        print("PLANNING COMPLETE")
        print("=" * 60)

        if config.advanced_plotting:
            # Save populated data for live visualization
            planner_data_store.save_data(self.reservations, self.graph, final_plans)

        # Save planned paths for distance error analysis
        if config.debug_speed_factor:
            planner_data_store.save_planned_paths()

        return final_plans

    def _initialize_environment(self, static_obstacles, goals, collection_points, start_positions):
        self.env_map = EnvironmentMap(
            static_obstacles=static_obstacles,
            goals=goals,
            collection_points=collection_points,
        )

        if config.graph.graph_type == "OCCUPANCY_GRID":
            self.graph = generate_occupancy_grid(
                self.env_map, 
                resolution=config.graph.occupancy_grid_resolution,
                start_positions=start_positions
            )
        elif config.graph.graph_type == "PRM":
            self.graph = generate_prm(
                self.env_map,
                num_samples=config.graph.prm_num_samples,
                connection_radius=config.graph.prm_connection_radius,
            )
        elif config.graph.graph_type == "VISIBILITY_GRAPH":
            self.graph = generate_inflated_visibility_graph(self.env_map, start_positions=start_positions)
        else:
            raise ValueError(f"Unknown graph type: {config.graph.graph_type}")

    def _extract_ids(self, objs: List[object]) -> List[str]:
        ids: List[str] = []
        for obj in objs:
            raw_id = self._get_id(obj)
            if raw_id is None:
                continue
            ids.append(str(raw_id))
        return ids

    def _extract_poi_center_nodes(self, poi_ids: List[str]) -> Dict[str, int]:
        if self.graph is None:
            raise RuntimeError("Graph not initialized")
        center_nodes: Dict[str, int] = {}
        for poi_id in poi_ids:
            nodes = self.graph.poi_nodes.get(poi_id, [])
            if nodes:
                center_nodes[poi_id] = nodes[0]
        return center_nodes

    @staticmethod
    def _get_id(obj):
        return getattr(obj, "id", getattr(obj, "goal_id", getattr(obj, "point_id", None)))

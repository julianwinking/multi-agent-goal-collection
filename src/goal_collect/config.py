import numpy as np
from dataclasses import dataclass, field


@dataclass
class AgentParams:
    robot_radius: float = 0.6

    # Dynamic Obstacles
    dynamic_obstacle_radius: float = 0.35

    # Path Following
    path_end_buffer: float = 0.05

    # Wait point detection (separate from controller waypoint_tolerance)
    wait_check_tolerance: float = 0.5  # Distance to detect waypoints with waits (larger to catch more)


@dataclass
class GraphParams:
    occupancy_grid_resolution: float = 0.4
    prm_num_samples: int = 500
    prm_connection_radius: float = 4.0
    prm_seed: int = 42
    robot_radius_buffer: float = 0.2  # Used for inflated obstacles
    poi_buffer: float = 0.5
    graph_type: str = "OCCUPANCY_GRID"  # Options: OCCUPANCY_GRID, PRM, VISIBILITY_GRAPH
    # Epsilon adds a small buffer to the inflated obstacles to ensure nodes are strictly free
    # and avoid numerical issues at boundaries.
    visibility_graph_epsilon: float = 0.05
    reduced_visibility_graph: bool = True

    # Parking Parameters
    parking_min_dist: float = 2.0
    parking_max_dist: float = 5.0
    parking_samples: int = 20


@dataclass
class PathParams:
    heuristic_weight: float = 0.0


@dataclass
class MILPParams:
    """Configuration for MILP fleet optimizer."""

    # Solver settings
    time_limit: float = 40.0  # Maximum solve time (seconds)
    gap_tolerance: float = 0.05  # MIP gap tolerance (5% optimality)
    threads: int = 4  # Parallel threads for solver

    # Objective weights (aligned with scoring formula)
    # Score formula: +10 pts/second saved (makespan), -0.5 pts/unit distance
    makespan_weight: float = 10.0  # Weight for makespan minimization
    distance_weight: float = 2.5  # Weight for total distance
    load_balance_weight: float = 2.0  # Penalty for unbalanced loads

    # CP selection
    max_cp_options: int = 3  # Max CPs to consider per goal in MILP

    # Pruning
    edge_prune_factor: float = 2.0  # Skip goal-to-goal edges > factor * direct distance


@dataclass
class RegretInsertionParams:
    """Configuration for Regret-Based Insertion heuristic."""

    makespan_weight: float = 0.1
    distance_weight: float = 2.5
    p: int = 1  # Regret power (1 for sum, higher for prioritizing difficult tasks)
    conflict_radius: float = 2.0  # Radius to check for spatial crowding
    conflict_penalty: float = 5.0  # Penalty for inserting near other agents' goals
    refinement_iterations: int = 5  # Number of passes to refine CP selection


@dataclass
class PlannerParams:
    solver_method: str = "GREEDY"  # Options: MILP, REGRET, GREEDY

    # Security
    safety_dist_buffer: float = 0.4
    parking_buffer_time: float = 1.3
    parking_dist_buffer: float = 0.25  # Buffer for idle agent reservation radius


@dataclass
class Pdm4arConfig:
    enable_plotting: bool = False
    advanced_plotting: bool = False
    debug_speed_factor: bool = False  # Enable speed factor analysis and box plots
    verbose_agent_logging: bool = False  # Control verbose agent debug output
    agent: AgentParams = field(default_factory=AgentParams)
    graph: GraphParams = field(default_factory=GraphParams)
    path: PathParams = field(default_factory=PathParams)
    planner: PlannerParams = field(default_factory=PlannerParams)
    milp: MILPParams = field(default_factory=MILPParams)
    regret: RegretInsertionParams = field(default_factory=RegretInsertionParams)
    plots_dir: str = "visualizations"


# Global config instance
config = Pdm4arConfig()

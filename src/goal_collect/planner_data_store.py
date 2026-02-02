import pickle
import os
from typing import Optional, Dict, List, Tuple
from .planner.graph import Graph
from .structures import ReservationTable, AgentPlan

# Global store for visualization (in-memory)
graph: Optional[Graph] = None
reservations: Optional[ReservationTable] = None
plans: Optional[Dict[str, AgentPlan]] = None

DATA_FILE = "src/pdm4ar/exercises/ex14/plots/planner_data.pkl"
TRANSITION_FILE = "src/pdm4ar/exercises/ex14/plots/transition_timing.pkl"
PLANNED_PATHS_FILE = "src/pdm4ar/exercises/ex14/plots/planned_paths.pkl"

# In-memory storage for planned paths during planning phase
# Format: {agent_name: [(time, x, y, is_poi, poi_type), ...]}
_planned_paths: Dict[str, List[Tuple[float, float, float, bool, str]]] = {}


def save_data(res: ReservationTable, g: Graph, p: Dict[str, AgentPlan]):
    """Save planner data to disk."""
    global reservations, graph, plans
    reservations = res
    graph = g
    plans = p

    os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
    try:
        with open(DATA_FILE, "wb") as f:
            pickle.dump((res, g, p), f)
        print(f"Planner data saved to {DATA_FILE}")
    except Exception as e:
        print(f"Failed to save planner data: {e}")


def load_data():
    """Load planner data from disk if not already loaded."""
    global reservations, graph, plans

    if reservations is not None and graph is not None and plans is not None:
        return

    if not os.path.exists(DATA_FILE):
        return

    try:
        with open(DATA_FILE, "rb") as f:
            data = pickle.load(f)
            if len(data) == 3:
                reservations, graph, plans = data
            else:
                reservations, graph = data
                plans = None
                print("Warning: Loaded legacy planner data (no plans)")

        print(f"Planner data loaded from {DATA_FILE}")
    except Exception as e:
        print(f"Failed to load planner data: {e}")


def record_transition(
    agent_name: str,
    category: str,
    distance: float,
    expected_duration: float,
    actual_duration: float,
    from_x: float = 0.0,
    from_y: float = 0.0,
    to_x: float = 0.0,
    to_y: float = 0.0,
):
    """
    Record a single node-to-node transition for timing analysis.
    Uses file-based storage to work across processes.
    """
    os.makedirs(os.path.dirname(TRANSITION_FILE), exist_ok=True)
    
    # Load existing data
    data = []
    if os.path.exists(TRANSITION_FILE):
        try:
            with open(TRANSITION_FILE, "rb") as f:
                data = pickle.load(f)
        except:
            data = []
    
    # Append new transition
    data.append({
        'agent_name': agent_name,
        'category': category,
        'distance': distance,
        'expected_duration': expected_duration,
        'actual_duration': actual_duration,
        'from_x': from_x,
        'from_y': from_y,
        'to_x': to_x,
        'to_y': to_y,
    })
    
    # Save back to file
    with open(TRANSITION_FILE, "wb") as f:
        pickle.dump(data, f)


def clear_transition_timing():
    """Clear transition timing data for new simulation run."""
    if os.path.exists(TRANSITION_FILE):
        os.remove(TRANSITION_FILE)
    print(f"[DATASTORE] Cleared transition timing file")


def get_transition_timing_data():
    """Get collected transition timing data from file."""
    if not os.path.exists(TRANSITION_FILE):
        print(f"[DATASTORE] No transition file found")
        return []
    
    try:
        with open(TRANSITION_FILE, "rb") as f:
            data = pickle.load(f)
        print(f"[DATASTORE] Loaded {len(data)} transitions from file")
        return data
    except Exception as e:
        print(f"[DATASTORE] Failed to load transitions: {e}")
        return []


# ============== Planned Paths for Distance Error Plotting ==============

def record_planned_path_point(
    agent_name: str,
    time: float,
    x: float,
    y: float,
    is_poi: bool = False,
    poi_type: str = "",
):
    """
    Record a single planned path point for an agent.
    Used to build the planned trajectory for distance error analysis.
    """
    global _planned_paths
    if agent_name not in _planned_paths:
        _planned_paths[agent_name] = []
    _planned_paths[agent_name].append((time, x, y, is_poi, poi_type))


def save_planned_paths():
    """Save planned paths data to disk for post-simulation analysis."""
    global _planned_paths
    os.makedirs(os.path.dirname(PLANNED_PATHS_FILE), exist_ok=True)
    
    try:
        with open(PLANNED_PATHS_FILE, "wb") as f:
            pickle.dump(_planned_paths, f)
        total_points = sum(len(pts) for pts in _planned_paths.values())
        print(f"[DATASTORE] Saved planned paths for {len(_planned_paths)} agents ({total_points} points)")
    except Exception as e:
        print(f"[DATASTORE] Failed to save planned paths: {e}")


def get_planned_paths_data() -> Dict[str, List[Tuple[float, float, float, bool, str]]]:
    """Get planned paths data from file."""
    if not os.path.exists(PLANNED_PATHS_FILE):
        print(f"[DATASTORE] No planned paths file found")
        return {}
    
    try:
        with open(PLANNED_PATHS_FILE, "rb") as f:
            data = pickle.load(f)
        total_points = sum(len(pts) for pts in data.values())
        print(f"[DATASTORE] Loaded planned paths for {len(data)} agents ({total_points} points)")
        return data
    except Exception as e:
        print(f"[DATASTORE] Failed to load planned paths: {e}")
        return {}


def clear_planned_paths():
    """Clear planned paths data for new simulation run."""
    global _planned_paths
    _planned_paths = {}
    if os.path.exists(PLANNED_PATHS_FILE):
        os.remove(PLANNED_PATHS_FILE)
    print(f"[DATASTORE] Cleared planned paths data")

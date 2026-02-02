import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from dg_commons.sim import InitSimGlobalObservations
from shapely import plotting
from shapely.geometry import Point, LinearRing

from .planner.graph import Graph, get_poly
from .structures import AgentPlan, Mission
from .planner.path import get_shortest_path
from .config import config
import os
import numpy as np


def generate_intermediate_plans(
    agents,
    agent_start_nodes,
    global_solution,
    goal_nodes,
    cp_nodes,
    graph: Graph,
):
    intermediate_plans = {}
    for agent_name in agents:
        missions = []
        current_node = agent_start_nodes[agent_name]

        if agent_name in global_solution.sequences:
            for gid in global_solution.sequences[agent_name]:
                # Path to Goal
                path_to_goal = []
                if gid in goal_nodes:
                    g_node_id = goal_nodes[gid]
                    path_to_goal = get_shortest_path(graph, current_node, g_node_id)
                    current_node = g_node_id

                # Path to CP
                path_to_collection = []
                if gid in global_solution.cp_assignments:
                    cpid = global_solution.cp_assignments[gid]
                    if cpid in cp_nodes:
                        cp_node_id = cp_nodes[cpid]
                        path_to_collection = get_shortest_path(graph, current_node, cp_node_id)
                        current_node = cp_node_id

                missions.append(Mission(goal_id=gid, path_to_goal=path_to_goal, path_to_collection=path_to_collection))

        intermediate_plans[agent_name] = AgentPlan(missions=missions)
    return intermediate_plans


def plot_env_graph(
    static_obstacles,
    goals,
    collection_points,
    agent_starts,
    graph: Graph,
    output_file: str,
):
    """
    Plots the static environment, agents, goals, collection points, and the generated graph.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect("equal")

    # 1. Plot Obstacles
    for obs in static_obstacles:
        poly = get_poly(obs)
        if isinstance(poly, LinearRing):
            plotting.plot_line(poly, ax=ax, add_points=False, color="gray", alpha=0.5)
        else:
            plotting.plot_polygon(poly, ax=ax, add_points=False, color="gray", alpha=0.5)

    # 2. Plot Goals
    # shared_goals and collection_points might be dicts {id: object}
    _goals = goals.values() if isinstance(goals, dict) else goals
    if _goals:
        for goal in _goals:
            plotting.plot_polygon(get_poly(goal), ax=ax, add_points=False, color="green", alpha=0.3, label="Goal")

    _cps = collection_points.values() if isinstance(collection_points, dict) else collection_points
    if _cps:
        for cp in _cps:
            plotting.plot_polygon(
                get_poly(cp), ax=ax, add_points=False, color="red", alpha=0.3, label="Collection Point"
            )

    # 3. Plot Agents
    for name, state in agent_starts.items():
        ax.plot(state.x, state.y, "bo", label="Agent")
        # simple arrow for heading
        import numpy as np

        # Check if state has psi (DiffDriveState) or just x,y (Point)
        psi = getattr(state, "psi", 0.0)
        arrow_len = 0.5
        ax.arrow(state.x, state.y, arrow_len * np.cos(psi), arrow_len * np.sin(psi), head_width=0.2, color="blue")

    # 4. Plot Graph
    # Nodes
    xs, ys = [], []
    poi_center_ids = set()
    for nodes_list in graph.poi_nodes.values():
        if nodes_list:
            poi_center_ids.add(nodes_list[0])

    xs_normal, ys_normal = [], []
    xs_poi, ys_poi = [], []

    for node in graph.nodes.values():
        if node.id in poi_center_ids:
            xs_poi.append(node.x)
            ys_poi.append(node.y)
        else:
            xs_normal.append(node.x)
            ys_normal.append(node.y)

    ax.scatter(xs_normal, ys_normal, s=5, c="black", zorder=10)
    ax.scatter(xs_poi, ys_poi, s=20, c="yellow", edgecolors="black", zorder=11, label="POI Node")

    # Edges
    # To optimize plotting, we can create a LineCollection or just plot segments
    from matplotlib.collections import LineCollection

    lines = []
    for u_id, neighbors in graph.edges.items():
        u = graph.get_node(u_id)
        for v_id in neighbors:
            # Avoid duplicating edges (undirected)
            if u_id < v_id:
                v = graph.get_node(v_id)
                lines.append([(u.x, u.y), (v.x, v.y)])

    lc = LineCollection(lines, colors="orange", linewidths=0.5, alpha=0.6, zorder=5)
    ax.add_collection(lc)

    ax.autoscale()
    plt.title("Graph Verification")
    # Create directory if it doesn't exist
    os.makedirs(config.plots_dir, exist_ok=True)

    # Construct full path
    # If output_file is just a filename, join it. If it's a path, use it?
    # Usually output_file passed here is just a filename like "env_graph.png".
    # But just in case, let's assume it's a filename or we join it.
    full_path = os.path.join(config.plots_dir, os.path.basename(output_file))

    plt.title("Graph Verification")
    plt.savefig(full_path)
    print(f"Graph verification plot saved to {full_path}")
    plt.close(fig)


def plot_all_plans(
    static_obstacles,
    goals,
    collection_points,
    agent_starts,
    agent_plans,
    graph: Graph,
    output_file: str = "plans_verification.png",
):
    """
    Plots the planned paths for all agents.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect("equal")

    # 1. Plot Environment (Obstacles, Goals, CPs)
    # Obstacles
    for obs in static_obstacles:
        poly = get_poly(obs)
        if hasattr(poly, "geom_type") and poly.geom_type == "LinearRing":
            plotting.plot_line(poly, ax=ax, add_points=False, color="gray", alpha=0.5)
        else:
            plotting.plot_polygon(poly, ax=ax, add_points=False, color="gray", alpha=0.5)

    # Goals
    _goals = goals.values() if isinstance(goals, dict) else goals
    if _goals:
        for goal in _goals:
            plotting.plot_polygon(get_poly(goal), ax=ax, add_points=False, color="green", alpha=0.3)

    # Collection Points
    _cps = collection_points.values() if isinstance(collection_points, dict) else collection_points
    if _cps:
        for cp in _cps:
            plotting.plot_polygon(get_poly(cp), ax=ax, add_points=False, color="red", alpha=0.3)

    # 2. Plot Paths
    colors = ["blue", "red", "purple", "orange", "cyan", "magenta", "brown", "pink"]
    linestyles = ["-", "--", "-.", ":"]

    import numpy as np

    # Pre-compute random offsets for each agent to avoid perfect overlap
    # Or deterministic offsets

    agent_names = list(agent_plans.keys())

    from matplotlib.lines import Line2D

    legend_elements = []

    for idx, name in enumerate(agent_names):
        plan = agent_plans[name]
        color = colors[idx % len(colors)]

        # Plot Start Position
        if name in agent_starts:
            start = agent_starts[name]
            ax.plot(start.x, start.y, marker="o", color=color, label=f"{name} Start")
            legend_elements.append(
                Line2D([0], [0], marker="o", color="w", label=f"{name}", markerfacecolor=color, markersize=8)
            )

        for m_idx, mission in enumerate(plan.missions):
            style = linestyles[m_idx % len(linestyles)]

            # Counter for forward/backward paths to vary shift
            # We want robust separation:
            # 1. Separate Agents (idx)
            # 2. Separate Missions (m_idx)
            # 3. Separate Forward/Backward (0/1)

            # Base separation unit
            sep = 0.08

            # Calculate a unique 'lane' for this path segment
            # Cycle every 10 missions to keep offsets from growing to infinity, but large enough cycle to avoid local conflicts
            mission_cycle = m_idx % 10

            # Direction: 0 for Goal (Forward), 1 for Collection (Backward)

            def get_shift(direction_id):
                # unique index = mission_cycle * 2 + direction
                # This ensures Fwd/Bwd of same mission are adjacent lanes, and next mission is next lane
                lane_id = mission_cycle * 2 + direction_id

                # Combine with agent specific small shift to separate agents in same lane
                # Agent shift is smaller than lane width
                agent_shift = (idx % 4) * 0.01

                total_offset = 0.03 + (lane_id * sep) + agent_shift

                # Apply diagonal shift
                # Alternate direction based on lane_id to distribute around center?
                # Or just monotonic to ensure separation? Monotonic is safer for consistency.
                return total_offset, total_offset

            # Path to Goal (Forward)
            if mission.path_to_goal:
                dx, dy = get_shift(0)
                xs = [p.x + dx for p in mission.path_to_goal]
                ys = [p.y + dy for p in mission.path_to_goal]
                ax.plot(xs, ys, color=color, linestyle=style, linewidth=1.5, alpha=0.8)

                # Plot Wait Positions
                for p_idx, p in enumerate(mission.path_to_goal):
                    if hasattr(p, "wait_duration") and p.wait_duration > 0.001:
                        ax.plot(p.x + dx, p.y + dy, "o", color="purple", markersize=6, zorder=20)

            # Path to Collection (Backward)
            if mission.path_to_collection:
                dx, dy = get_shift(1)
                xs = [p.x + dx for p in mission.path_to_collection]
                ys = [p.y + dy for p in mission.path_to_collection]
                ax.plot(xs, ys, color=color, linestyle=style, linewidth=1.5, alpha=0.8)

                # Plot Wait Positions
                for p_idx, p in enumerate(mission.path_to_collection):
                    if hasattr(p, "wait_duration") and p.wait_duration > 0.001:
                        ax.plot(p.x + dx, p.y + dy, "o", color="purple", markersize=6, zorder=20)

    # Add legend regarding line styles (showing pattern repetition)
    legend_elements.append(Line2D([0], [0], color="black", lw=1, linestyle="-", label="Mission 1, 5, ..."))
    legend_elements.append(Line2D([0], [0], color="black", lw=1, linestyle="--", label="Mission 2, 6, ..."))
    legend_elements.append(Line2D([0], [0], color="black", lw=1, linestyle="-.", label="Mission 3, 7, ..."))
    legend_elements.append(Line2D([0], [0], color="black", lw=1, linestyle=":", label="Mission 4, 8, ..."))

    ax.legend(handles=legend_elements, loc="upper right")
    ax.autoscale()
    plt.title("Planned Paths Verification")
    plt.title("Planned Paths Verification")
    os.makedirs(config.plots_dir, exist_ok=True)
    full_path = os.path.join(config.plots_dir, os.path.basename(output_file))
    plt.savefig(full_path)
    print(f"Plans verification plot saved to {full_path}")
    plt.close(fig)


def plot_mpc_debug(
    agent_name: str,
    current_state,
    static_obstacles,
    dynamic_obstacles: list,
    ref_traj: list,
    x_cvx,
    x_ref,
    obs_nx,
    obs_ny,
    obs_b,
    output_dir: str = None,
    slacks=None,
    goals=None,
    collection_points=None,
    is_parking: bool = False,
):
    """
    Plots comprehensive MPC debug visualization for a single agent.
    Shows environment, obstacles, reference trajectory, and MPC solution.
    """
    import os
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Circle
    from shapely import plotting as shapely_plotting

    if x_cvx is None:
        return

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 14))
    ax.set_aspect("equal")

    # --- 1. ENVIRONMENT ---
    # Static Obstacles (gray polygons)
    if static_obstacles:
        for obs in static_obstacles:
            poly = get_poly(obs)
            if hasattr(poly, "geom_type"):
                if poly.geom_type == "LinearRing":
                    shapely_plotting.plot_line(poly, ax=ax, add_points=False, color="purple", alpha=0.6)
                else:
                    shapely_plotting.plot_polygon(poly, ax=ax, add_points=False, color="purple", alpha=0.6)

    # --- 1b. GOALS (green polygons) ---
    if goals:
        for goal in goals:
            poly = get_poly(goal)
            if hasattr(poly, "geom_type"):
                if poly.geom_type == "LinearRing":
                    shapely_plotting.plot_line(poly, ax=ax, add_points=False, color="green", alpha=0.4)
                else:
                    shapely_plotting.plot_polygon(
                        poly, ax=ax, add_points=False, facecolor="green", edgecolor="darkgreen", alpha=0.4
                    )

    # --- 1c. COLLECTION POINTS (red polygons) ---
    if collection_points:
        for cp in collection_points:
            poly = get_poly(cp)
            if hasattr(poly, "geom_type"):
                if poly.geom_type == "LinearRing":
                    shapely_plotting.plot_line(poly, ax=ax, add_points=False, color="red", alpha=0.4)
                else:
                    shapely_plotting.plot_polygon(
                        poly, ax=ax, add_points=False, facecolor="red", edgecolor="darkred", alpha=0.4
                    )

    # --- 2. DYNAMIC OBSTACLES (other agents) ---
    if dynamic_obstacles:
        for obs_data in dynamic_obstacles:
            obs_name, ox, oy, orad = obs_data
            circle = Circle(
                (ox, oy),
                orad,
                fill=True,
                facecolor="orange",
                edgecolor="red",
                alpha=0.5,
                linewidth=2,
                label=f"Agent: {obs_name}",
            )
            ax.add_patch(circle)
            ax.annotate(obs_name, (ox, oy), ha="center", va="center", fontsize=8, fontweight="bold")

    # --- 3. REFERENCE TRAJECTORY ---
    if ref_traj:
        ref_x = [r["x"] for r in ref_traj]
        ref_y = [r["y"] for r in ref_traj]
        ax.plot(ref_x, ref_y, "g--", linewidth=2, alpha=0.7, label="Reference Trajectory", zorder=5)
        ax.scatter(ref_x, ref_y, c="green", s=20, alpha=0.5, zorder=5)

    # --- 4. MPC SOLUTION ---
    ax.plot(x_cvx[0, :], x_cvx[1, :], "b-", linewidth=3, label="MPC Trajectory", zorder=10)
    ax.scatter(x_cvx[0, :], x_cvx[1, :], c="blue", s=40, zorder=10)

    # --- 5. CURRENT STATE (with heading arrow) ---
    # Use blue color when parking, lime otherwise
    agent_color = "blue" if is_parking else "lime"
    agent_label = f"{agent_name} Position (Parking)" if is_parking else f"{agent_name} Position"
    ax.scatter(
        current_state.x,
        current_state.y,
        c=agent_color,
        s=200,
        edgecolors="black",
        linewidth=2,
        zorder=15,
        label=agent_label,
    )

    # Heading arrow
    arrow_len = 0.8
    dx = arrow_len * np.cos(current_state.psi)
    dy = arrow_len * np.sin(current_state.psi)
    ax.arrow(
        current_state.x, current_state.y, dx, dy, head_width=0.2, head_length=0.1, fc="black", ec="black", zorder=16
    )

    # --- 6. OBSTACLE CONSTRAINT PLANES ---
    N = x_cvx.shape[1] - 1
    num_obs = obs_nx.shape[0]

    # Fixed axis limits for consistent visualization
    x_min, x_max = -14, 14
    y_min, y_max = -14, 14

    x_range = np.linspace(x_min, x_max, 100)

    # Plot active constraints for every 3rd step
    for k in range(0, N, 3):
        for i in range(num_obs):
            nx = obs_nx[i, k]
            ny = obs_ny[i, k]
            b = obs_b[i, k]

            if b < -9000:  # Dummy value check
                continue

            # Plot constraint line: nx*x + ny*y = b
            if abs(ny) > 1e-3:
                y_line = (b - nx * x_range) / ny
                mask = (y_line >= y_min) & (y_line <= y_max)
                if np.any(mask):
                    ax.plot(x_range[mask], y_line[mask], "r-", alpha=0.3, linewidth=1)

    # --- 7. SLACK VISUALIZATION ---
    if slacks is not None:
        max_slack = np.max(slacks)
        if max_slack > 1e-3:
            ax.text(
                0.02,
                0.98,
                f"⚠ Max Slack: {max_slack:.3f}",
                transform=ax.transAxes,
                color="red",
                fontsize=12,
                fontweight="bold",
                va="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

    # --- 8. FORMATTING ---
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_title(f"MPC Debug: {agent_name}", fontsize=14, fontweight="bold")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")

    # Save
    # Save
    save_dir = output_dir if output_dir else config.plots_dir
    os.makedirs(save_dir, exist_ok=True)
    # create a filename based on agent name and time or just agent name
    # Since this is debug, maybe we need a unique name?
    # But for now, let's just use a fixed name or derived from agent name.
    # The user didn't specify filename logic, but saving to a directory requires a filename.
    # I'll create a filename: f"mpc_debug_{agent_name}.png"
    filename = f"mpc_debug_{agent_name}.png"
    full_path = os.path.join(save_dir, filename)

    plt.savefig(full_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def plot_spacetime_reservations(
    graph: Graph,
    reservations,
    output_file: str = "spacetime_reservations.png",
    max_time: float = 20.0,
):
    """
    Plots the graph nodes and their reservations in a 3D Space-Time plot (X, Y, Time).
    """

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    # 1. Plot Graph Structure on the "floor" (t=0)
    # Nodes
    xs, ys = [], []
    for node in graph.nodes.values():
        xs.append(node.x)
        ys.append(node.y)
    ax.scatter(xs, ys, zs=0, c="black", s=10, alpha=0.5, label="Nodes (t=0)")

    # Edges
    for u_id, neighbors in graph.edges.items():
        u = graph.get_node(u_id)
        for v_id in neighbors:
            if u_id < v_id:
                v = graph.get_node(v_id)
                ax.plot([u.x, v.x], [u.y, v.y], [0, 0], color="gray", alpha=0.2, linewidth=0.5)

    # 2. Plot Reservations
    agent_colors = {}
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]

    def get_color(agent_id):
        if agent_id not in agent_colors:
            agent_colors[agent_id] = colors[len(agent_colors) % len(colors)]
        return agent_colors[agent_id]

    # 3. Plot Agent Trajectories from geometric reservations
    # Group geometric reservations by agent to draw continuous paths
    agent_trajectories = {}
    for p1, p2, interval, agent_id in reservations.geometric_reservations:
        if agent_id not in agent_trajectories:
            agent_trajectories[agent_id] = []
        agent_trajectories[agent_id].append((p1, p2, interval))

    # Draw trajectories as 3D lines
    for agent_id, traj_segments in agent_trajectories.items():
        c = get_color(agent_id)
        for p1, p2, interval in traj_segments:
            start_time = max(0, interval.start)
            end_time = min(max_time, interval.end)

            if start_time >= max_time or end_time <= 0:
                continue

            # Draw line segment in 3D space-time
            ax.plot([p1.x, p2.x], [p1.y, p2.y], [start_time, end_time], color=c, linewidth=2, alpha=0.7)

    # Node Reservations (vertical lines showing stationary periods)
    for node_id, res_list in reservations.node_reservations.items():
        if node_id not in graph.nodes:
            continue
        node = graph.nodes[node_id]
        x, y = node.x, node.y

        for interval, agent_id in res_list:
            start = max(0, interval.start)
            end = min(max_time, interval.end)

            if start >= max_time:
                continue
            if end <= 0:
                continue

            c = get_color(agent_id)
            ax.plot([x, x], [y, y], [start, end], color=c, linewidth=3, alpha=0.8)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Time (s)")
    ax.set_title("Space-Time Reservations")

    from matplotlib.lines import Line2D

    legend_elements = [Line2D([0], [0], color=c, lw=2, label=a) for a, c in agent_colors.items()]
    ax.legend(handles=legend_elements)

    # Save Default View
    # Save Default View
    os.makedirs(config.plots_dir, exist_ok=True)
    full_path = os.path.join(config.plots_dir, os.path.basename(output_file))
    plt.savefig(full_path)
    print(f"Space-time reservation plot saved to {full_path}")

    base, ext = full_path.rsplit(".", 1)

    # Save View 2 (Top-Down)
    ax.view_init(elev=90, azim=0)
    out2 = f"{base}_top.{ext}"
    plt.savefig(out2)
    print(f"Space-time reservation plot (top view) saved to {out2}")

    # Save View 3 (Side View)
    ax.view_init(elev=10, azim=45)
    out3 = f"{base}_side.{ext}"
    plt.savefig(out3)
    print(f"Space-time reservation plot (side view) saved to {out3}")

    # Save Rotated Views
    angles = [45, 135, 225, 315]
    for angle in angles:
        ax.view_init(elev=30, azim=angle)
        out_rot = f"{base}_rot_{angle}.{ext}"
        plt.savefig(out_rot)
        print(f"Space-time reservation plot (rot {angle}) saved to {out_rot}")

    plt.close(fig)


def animate_spacetime_reservations(
    graph: Graph,
    reservations,
    output_file: str = "spacetime_reservations_animated.gif",
    max_time: float = 20.0,
    fps: int = 10,
    duration_seconds: int = 6,
):
    """
    Creates an animated 3D rotating visualization of space-time reservations.

    Args:
        graph: The navigation graph
        reservations: ReservationTable object containing node reservations
        output_file: Output filename (supports .gif, .mp4)
        max_time: Maximum time to display on Z-axis
        fps: Frames per second for the animation
        duration_seconds: Total duration of the animation in seconds
    """
    print(f"Generating animated 3D space-time reservations...")
    print(f"  Creating {fps * duration_seconds} frames at {fps} fps...")

    fig = plt.figure(figsize=(8, 6), dpi=80)  # Smaller, lower DPI for speed
    ax = fig.add_subplot(111, projection="3d")

    # 1. Skip plotting the full graph floor - it's too slow and clutters the view
    # Only plot a sparse subset for context
    node_sample = list(graph.nodes.values())[::10]  # Every 10th node
    xs, ys = [], []
    for node in node_sample:
        xs.append(node.x)
        ys.append(node.y)
    ax.scatter(xs, ys, zs=0, c="gray", s=5, alpha=0.3)

    # Skip edges entirely - they clutter the 3D view

    # 2. Plot Reservations
    agent_colors = {}
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]

    def get_color(agent_id):
        if agent_id not in agent_colors:
            agent_colors[agent_id] = colors[len(agent_colors) % len(colors)]
        return agent_colors[agent_id]

    # 3. Plot Agent Trajectories from geometric reservations
    # Group geometric reservations by agent to draw continuous paths
    agent_trajectories = {}
    for p1, p2, interval, agent_id in reservations.geometric_reservations:
        if agent_id not in agent_trajectories:
            agent_trajectories[agent_id] = []
        agent_trajectories[agent_id].append((p1, p2, interval))

    # Draw trajectories as 3D lines
    for agent_id, traj_segments in agent_trajectories.items():
        c = get_color(agent_id)
        for p1, p2, interval in traj_segments:
            start_time = max(0, interval.start)
            end_time = min(max_time, interval.end)

            if start_time >= max_time or end_time <= 0:
                continue

            # Draw line segment in 3D space-time
            ax.plot([p1.x, p2.x], [p1.y, p2.y], [start_time, end_time], color=c, linewidth=2, alpha=0.7)

    # Node Reservations (vertical lines showing stationary periods)
    for node_id, res_list in reservations.node_reservations.items():
        if node_id not in graph.nodes:
            continue
        node = graph.nodes[node_id]
        x, y = node.x, node.y

        for interval, agent_id in res_list:
            start = max(0, interval.start)
            end = min(max_time, interval.end)

            if start >= max_time:
                continue
            if end <= 0:
                continue

            c = get_color(agent_id)
            ax.plot([x, x], [y, y], [start, end], color=c, linewidth=3, alpha=0.8)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Time (s)")
    ax.set_title("Space-Time Reservations (Rotating View)")

    from matplotlib.lines import Line2D

    legend_elements = [Line2D([0], [0], color=c, lw=2, label=a) for a, c in agent_colors.items()]
    ax.legend(handles=legend_elements, loc="upper right")

    # Animation function
    total_frames = fps * duration_seconds

    def update(frame):
        # Smooth rotation: full 360° rotation over the duration
        azim = (frame / total_frames) * 360
        # Gentle elevation oscillation for better 3D perception
        elev = 30 + 10 * np.sin(2 * np.pi * frame / total_frames)
        ax.view_init(elev=elev, azim=azim)
        return (ax,)

    print(f"  Rendering animation...")
    # Create animation
    anim = FuncAnimation(fig, update, frames=total_frames, interval=1000 / fps, blit=False)

    # Save animation using imageio (much faster than PillowWriter for GIFs)
    os.makedirs(config.plots_dir, exist_ok=True)
    full_path = os.path.join(config.plots_dir, os.path.basename(output_file))

    print(f"  Capturing frames...")
    try:
        import imageio
        import time

        start_time = time.time()
        frames = []
        for i in range(total_frames):
            frame_start = time.time()
            update(i)
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8")
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            frames.append(image)

            # Progress updates every 10 frames or at key milestones
            if (i + 1) % 10 == 0 or i == 0:
                elapsed = time.time() - start_time
                frame_time = time.time() - frame_start
                frames_remaining = total_frames - (i + 1)
                est_remaining = frames_remaining * frame_time
                print(
                    f"    Frame {i + 1}/{total_frames} ({100*(i+1)/total_frames:.1f}%) - "
                    f"Elapsed: {elapsed:.1f}s, Est. remaining: {est_remaining:.1f}s"
                )

        print(f"  Writing GIF to {full_path}...")
        write_start = time.time()
        imageio.mimsave(full_path, frames, fps=fps, loop=0)
        write_time = time.time() - write_start
        total_time = time.time() - start_time
        print(f"✓ Animated 3D space-time reservations saved to {full_path}")
        print(f"  Total time: {total_time:.1f}s (capture: {total_time-write_time:.1f}s, write: {write_time:.1f}s)")
    except Exception as e:
        print(f"✗ Failed to create animation: {e}")
        return None

    plt.close(fig)
    return full_path


def plot_live_reservations(
    current_time: float,
    graph: Graph,
    reservations,
    agents_state: dict,
    static_obstacles: list,
    output_dir: str = None,
    agent_name: str = "agent",
    agent_plan: AgentPlan = None,
):
    """
    Plots the live positions of agents and blocked nodes at the current time.

    Args:
        current_time: Current simulation time.
        graph: The navigation graph.
        reservations: The ReservationTable object.
        agents_state: Dictionary of agent names to their current state (must have x, y).
        static_obstacles: List of static obstacles.
        output_dir: Directory to save the plot.
        agent_name: Name of the agent (used for filename).
        agent_plan: Optional plan for the current agent to visualize wait times.
    """
    import matplotlib.pyplot as plt
    from shapely import plotting as shapely_plotting
    import numpy as np
    import os

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect("equal")

    # 1. Plot Static Obstacles
    if static_obstacles:
        for obs in static_obstacles:
            poly = get_poly(obs)
            if hasattr(poly, "geom_type"):
                if poly.geom_type == "LinearRing":
                    shapely_plotting.plot_line(poly, ax=ax, add_points=False, color="gray", alpha=0.5)
                else:
                    shapely_plotting.plot_polygon(poly, ax=ax, add_points=False, color="gray", alpha=0.5)

    # 2. Plot Nodes
    blocked_xs, blocked_ys = [], []
    free_xs, free_ys = [], []

    # Efficiently check all nodes
    for node in graph.nodes.values():
        is_safe = reservations.is_node_safe(node.id, current_time)
        if not is_safe:
            blocked_xs.append(node.x)
            blocked_ys.append(node.y)
        else:
            free_xs.append(node.x)
            free_ys.append(node.y)

    # Plot free nodes faintly
    if free_xs:
        ax.scatter(free_xs, free_ys, c="lightgray", s=10, zorder=1)

    # Plot blocked nodes prominently
    if blocked_xs:
        ax.scatter(blocked_xs, blocked_ys, c="black", s=40, label="Blocked Node", zorder=5)

    # 3. Plot Plan Waits (Purple Dots)
    if agent_plan:
        wait_xs, wait_ys = [], []
        for mission in agent_plan.missions:
            # Check path to goal
            if mission.path_to_goal:
                for p in mission.path_to_goal:
                    if hasattr(p, "wait_duration") and p.wait_duration > 0.001:
                        wait_xs.append(p.x)
                        wait_ys.append(p.y)

            # Check path to collection
            if mission.path_to_collection:
                for p in mission.path_to_collection:
                    if hasattr(p, "wait_duration") and p.wait_duration > 0.001:
                        wait_xs.append(p.x)
                        wait_ys.append(p.y)

        if wait_xs:
            ax.scatter(wait_xs, wait_ys, c="purple", s=60, label="Wait Location", zorder=8, edgecolors="white")

    # 4. Plot Agents
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]

    for i, (name, state) in enumerate(agents_state.items()):
        color = colors[i % len(colors)]
        ax.scatter(state.x, state.y, color=color, s=120, label=name, edgecolors="black", zorder=10)

        # Heading arrow
        if hasattr(state, "psi"):
            arrow_len = 0.6
            dx = arrow_len * np.cos(state.psi)
            dy = arrow_len * np.sin(state.psi)
            ax.arrow(state.x, state.y, dx, dy, head_width=0.25, color="black", length_includes_head=True, zorder=11)

    # Formatting
    ax.legend(loc="upper right")
    ax.set_title(f"Live Reservations @ t={current_time:.2f}s")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")

    # Auto-scale roughly to contents but keep square
    ax.autoscale(enable=True)

    # Save - use config.plots_dir by default, overwrite same file per agent for animation viewing
    save_dir = output_dir if output_dir else config.plots_dir
    os.makedirs(save_dir, exist_ok=True)
    # Fixed filename per agent (no timestamp) so the file gets overwritten each update
    filename = f"live_reservations_{agent_name}.png"
    full_path = os.path.join(save_dir, filename)

    plt.savefig(full_path)
    plt.close(fig)


def plot_speed_factor_analysis(
    final_plans: dict,
    safe_speed_factor: float,
    max_v: float,
    output_file: str = "speed_factor_analysis.png",
):
    """
    Creates a box plot showing the difference between expected duration
    (using Euclidean distance) and planned duration for each mission.

    Shows:
    - Expected duration = euclidean_distance / (max_v * safe_speed_factor)
    - Planned duration = actual path travel time (from Mission data)
    - Marginal difference (faster/slower)
    - Optimal speed factor per mission

    Args:
        final_plans: Dict of agent_name -> AgentPlan with timing data
        safe_speed_factor: The current safe_speed_factor from config
        max_v: Maximum velocity of the robot
        output_file: Output filename for the plot
    """
    import numpy as np

    # Collect all mission timing data
    mission_data = []  # List of (agent, goal_id, expected_duration, planned_duration, optimal_factor)

    safe_speed = max_v  # * safe_speed_factor

    for agent_name, plan in final_plans.items():
        for mission in plan.missions:
            # Skip missions without timing data
            if mission.euclidean_distance is None or mission.planned_duration is None:
                continue
            if mission.planned_duration <= 0:
                continue

            euclidean_dist = mission.euclidean_distance
            planned_dur = mission.planned_duration

            # Expected duration based on Euclidean distance and safe_speed
            expected_dur = euclidean_dist / safe_speed

            # Compute optimal factor: what factor would make expected = planned?
            # expected = euclidean / (max_v * factor) = planned
            # factor = euclidean / (max_v * planned)
            optimal_factor = euclidean_dist / (max_v * planned_dur)

            mission_data.append(
                {
                    "agent": agent_name,
                    "goal_id": mission.goal_id,
                    "euclidean_distance": euclidean_dist,
                    "expected_duration": expected_dur,
                    "planned_duration": planned_dur,
                    "diff": planned_dur - expected_dur,  # Positive = slower than expected
                    "diff_percent": ((planned_dur - expected_dur) / expected_dur) * 100 if expected_dur > 0 else 0,
                    "optimal_factor": optimal_factor,
                }
            )

    if not mission_data:
        print("[SPEED_FACTOR] No mission timing data available for analysis.")
        return

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # --- Subplot 1: Box plot of time differences ---
    # Group by agent
    agents = sorted(set(d["agent"] for d in mission_data))
    agent_diffs = {a: [] for a in agents}

    for d in mission_data:
        agent_diffs[d["agent"]].append(d["diff"])

    # Prepare data for box plot
    box_data = [agent_diffs[a] for a in agents]
    positions = list(range(len(agents)))

    # Colors for agents
    colors = plt.cm.tab10(np.linspace(0, 1, len(agents)))

    bp = ax1.boxplot(box_data, positions=positions, patch_artist=True, widths=0.6)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Add individual points
    for i, (agent, diffs) in enumerate(agent_diffs.items()):
        x = np.random.normal(i, 0.08, size=len(diffs))
        ax1.scatter(x, diffs, alpha=0.6, s=40, c=[colors[i]], edgecolors="black", linewidths=0.5)

    ax1.set_xticks(positions)
    ax1.set_xticklabels(agents, rotation=45, ha="right")
    ax1.axhline(y=0, color="red", linestyle="--", linewidth=1, label="Perfect Prediction")
    ax1.set_xlabel("Agent")
    ax1.set_ylabel("Time Difference (Planned - Expected) [s]")
    ax1.set_title(f"Mission Duration Prediction Error\n(safe_speed_factor = {safe_speed_factor:.3f})")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Add annotations
    all_diffs = [d["diff"] for d in mission_data]
    avg_diff = np.mean(all_diffs)
    std_diff = np.std(all_diffs)
    ax1.text(
        0.02,
        0.98,
        f"Mean: {avg_diff:.2f}s\nStd: {std_diff:.2f}s",
        transform=ax1.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    # --- Subplot 2: Optimal factor distribution ---
    optimal_factors = [d["optimal_factor"] for d in mission_data]

    ax2.hist(optimal_factors, bins=20, color="steelblue", alpha=0.7, edgecolor="black")
    ax2.axvline(
        x=safe_speed_factor, color="red", linestyle="--", linewidth=2, label=f"Current Factor: {safe_speed_factor:.3f}"
    )
    ax2.axvline(
        x=np.mean(optimal_factors),
        color="green",
        linestyle="-",
        linewidth=2,
        label=f"Mean Optimal: {np.mean(optimal_factors):.3f}",
    )
    ax2.axvline(
        x=np.median(optimal_factors),
        color="orange",
        linestyle="-",
        linewidth=2,
        label=f"Median Optimal: {np.median(optimal_factors):.3f}",
    )

    ax2.set_xlabel("Optimal Speed Factor")
    ax2.set_ylabel("Number of Missions")
    ax2.set_title("Optimal Speed Factor Distribution per Mission")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Add statistics
    ax2.text(
        0.98,
        0.98,
        f"Min: {np.min(optimal_factors):.3f}\nMax: {np.max(optimal_factors):.3f}\n"
        f"Mean: {np.mean(optimal_factors):.3f}\nMedian: {np.median(optimal_factors):.3f}",
        transform=ax2.transAxes,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    plt.tight_layout()

    # Save plot
    os.makedirs(config.plots_dir, exist_ok=True)
    full_path = os.path.join(config.plots_dir, os.path.basename(output_file))
    plt.savefig(full_path, dpi=150)
    print(f"[SPEED_FACTOR] Analysis plot saved to {full_path}")
    plt.close(fig)

    # Print summary to console
    print("\n" + "=" * 60)
    print("SPEED FACTOR ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"  Current safe_speed_factor: {safe_speed_factor:.3f}")
    print(f"  Number of missions analyzed: {len(mission_data)}")
    print(f"  Mean optimal factor: {np.mean(optimal_factors):.3f}")
    print(f"  Median optimal factor: {np.median(optimal_factors):.3f}")
    print(
        f"  Recommended factor range: {np.percentile(optimal_factors, 25):.3f} - {np.percentile(optimal_factors, 75):.3f}"
    )
    print(f"  Average prediction error: {avg_diff:.2f}s ({np.mean([d['diff_percent'] for d in mission_data]):.1f}%)")
    print("=" * 60)

    # Print per-mission details
    print("\nPer-Mission Details:")
    for d in mission_data:
        status = "SLOWER" if d["diff"] > 0 else "FASTER"
        print(
            f"  {d['agent']} Goal {d['goal_id']}: "
            f"Expected={d['expected_duration']:.2f}s, Planned={d['planned_duration']:.2f}s, "
            f"Diff={d['diff']:+.2f}s ({status}), Optimal={d['optimal_factor']:.3f}"
        )


def analyze_runtime_speed_factor(
    mission_timing_results: list,
    safe_speed_factor: float,
    max_v: float,
    output_file: str = "runtime_speed_factor_analysis.png",
):
    """
    Analyzes planned vs actual runtime mission completion times.

    Creates a box plot comparing:
    - Planned duration (from Phase 3 conflict resolution)
    - Actual duration (observed in simulator)
    - The optimal speed factor that would have given a perfect prediction

    Args:
        mission_timing_results: List of dicts from planner_data_store.get_mission_timing_results()
        safe_speed_factor: Current safe_speed_factor from config
        max_v: Maximum velocity of the robot
        output_file: Output filename for the plot
    """
    import numpy as np

    if not mission_timing_results:
        print("[RUNTIME_SPEED_FACTOR] No mission timing data collected during simulation.")
        return

    # Process timing data
    mission_data = []
    for d in mission_timing_results:
        planned_dur = d["planned_duration"]
        actual_dur = d["actual_duration"]
        euclidean_dist = d["euclidean_distance"]

        if actual_dur <= 0 or planned_dur <= 0:
            continue

        # Compute optimal factor based on actual runtime
        # optimal_speed = euclidean / actual_dur
        # optimal_factor = optimal_speed / max_v
        optimal_factor = euclidean_dist / (max_v * actual_dur) if actual_dur > 0 and euclidean_dist > 0 else 0

        mission_data.append(
            {
                "agent": d["agent_name"],
                "goal_id": d["goal_id"],
                "euclidean_distance": euclidean_dist,
                "planned_duration": planned_dur,
                "actual_duration": actual_dur,
                "diff": actual_dur - planned_dur,  # Positive = took longer than planned
                "diff_percent": ((actual_dur - planned_dur) / planned_dur) * 100 if planned_dur > 0 else 0,
                "optimal_factor": optimal_factor,
                "planned_eta": d["planned_eta"],
                "actual_completion": d["actual_completion"],
            }
        )

    if not mission_data:
        print("[RUNTIME_SPEED_FACTOR] No valid mission timing data after processing.")
        return

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # --- Subplot 1: Box plot of time differences (actual - planned) ---
    agents = sorted(set(d["agent"] for d in mission_data))
    agent_diffs = {a: [] for a in agents}

    for d in mission_data:
        agent_diffs[d["agent"]].append(d["diff"])

    # Prepare data for box plot
    box_data = [agent_diffs[a] for a in agents]
    positions = list(range(len(agents)))

    # Colors for agents
    import numpy as np

    colors = plt.cm.tab10(np.linspace(0, 1, len(agents)))

    bp = ax1.boxplot(box_data, positions=positions, patch_artist=True, widths=0.6)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Add individual points
    for i, (agent, diffs) in enumerate(agent_diffs.items()):
        x = np.random.normal(i, 0.08, size=len(diffs))
        ax1.scatter(x, diffs, alpha=0.6, s=40, c=[colors[i]], edgecolors="black", linewidths=0.5)

    ax1.set_xticks(positions)
    ax1.set_xticklabels(agents, rotation=45, ha="right")
    ax1.axhline(y=0, color="red", linestyle="--", linewidth=1, label="Perfect Prediction")
    ax1.set_xlabel("Agent")
    ax1.set_ylabel("Time Difference (Actual - Planned) [s]")
    ax1.set_title(f"Planned vs Actual Runtime\n(safe_speed_factor = {safe_speed_factor:.3f})")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Add annotations
    all_diffs = [d["diff"] for d in mission_data]
    avg_diff = np.mean(all_diffs)
    std_diff = np.std(all_diffs)
    ax1.text(
        0.02,
        0.98,
        f"Mean: {avg_diff:+.2f}s\nStd: {std_diff:.2f}s",
        transform=ax1.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    # --- Subplot 2: Optimal factor distribution ---
    optimal_factors = [d["optimal_factor"] for d in mission_data if d["optimal_factor"] > 0]

    if optimal_factors:
        ax2.hist(optimal_factors, bins=20, color="steelblue", alpha=0.7, edgecolor="black")
        ax2.axvline(
            x=safe_speed_factor,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Current Factor: {safe_speed_factor:.3f}",
        )
        ax2.axvline(
            x=np.mean(optimal_factors),
            color="green",
            linestyle="-",
            linewidth=2,
            label=f"Mean Optimal: {np.mean(optimal_factors):.3f}",
        )
        ax2.axvline(
            x=np.median(optimal_factors),
            color="orange",
            linestyle="-",
            linewidth=2,
            label=f"Median Optimal: {np.median(optimal_factors):.3f}",
        )

        ax2.set_xlabel("Optimal Speed Factor (from actual runtime)")
        ax2.set_ylabel("Number of Missions")
        ax2.set_title("Optimal Speed Factor Distribution (Runtime)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Add statistics
        ax2.text(
            0.98,
            0.98,
            f"Min: {np.min(optimal_factors):.3f}\nMax: {np.max(optimal_factors):.3f}\n"
            f"Mean: {np.mean(optimal_factors):.3f}\nMedian: {np.median(optimal_factors):.3f}",
            transform=ax2.transAxes,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

    plt.tight_layout()

    # Save plot
    os.makedirs(config.plots_dir, exist_ok=True)
    full_path = os.path.join(config.plots_dir, os.path.basename(output_file))
    plt.savefig(full_path, dpi=150)
    print(f"[RUNTIME_SPEED_FACTOR] Analysis plot saved to {full_path}")
    plt.close(fig)

    # Print summary to console
    print("\n" + "=" * 70)
    print("RUNTIME SPEED FACTOR ANALYSIS (Planned vs Actual Simulator Time)")
    print("=" * 70)
    print(f"  Current safe_speed_factor: {safe_speed_factor:.3f}")
    print(f"  Number of missions analyzed: {len(mission_data)}")
    if optimal_factors:
        print(f"  Mean optimal factor: {np.mean(optimal_factors):.3f}")
        print(f"  Median optimal factor: {np.median(optimal_factors):.3f}")
        print(
            f"  Recommended factor range: {np.percentile(optimal_factors, 25):.3f} - {np.percentile(optimal_factors, 75):.3f}"
        )
    print(f"  Average timing error: {avg_diff:+.2f}s ({np.mean([d['diff_percent'] for d in mission_data]):+.1f}%)")
    if avg_diff > 0:
        print(f"  → Actual runs SLOWER than planned. Consider DECREASING safe_speed_factor.")
    else:
        print(f"  → Actual runs FASTER than planned. Consider INCREASING safe_speed_factor.")
    print("=" * 70)

    # Print per-mission details
    print("\nPer-Mission Details (Planned vs Actual):")
    for d in mission_data:
        status = "SLOWER" if d["diff"] > 0 else "FASTER"
        print(
            f"  {d['agent']} Goal {d['goal_id']}: "
            f"Planned={d['planned_duration']:.2f}s, Actual={d['actual_duration']:.2f}s, "
            f"Diff={d['diff']:+.2f}s ({status}), Optimal={d['optimal_factor']:.3f}"
        )


def analyze_transition_timing(
    transition_data: list,
    safe_speed_factor: float,
    max_v: float,
    output_file: str = "transition_timing_analysis.png",
):
    """
    Analyzes node-to-node transition timing with 8 box plots.

    Angle-based categories:
    1. first_move - First move on path (no previous direction)
    2. straight - Continue straight (0° turn)
    3. slight_turn - 45° turn
    4. turn - 90° turn
    5. hard_turn - 135° turn
    6. u_turn - 180° turn
    7. goal_time - Time spent at goal pickup
    8. cp_time - Time spent at checkpoint dropoff
    """
    import numpy as np

    if not transition_data:
        print("[TRANSITION_TIMING] No transition timing data collected.")
        return

    # Define the 8 categories in order
    categories = [
        "first_move",
        "straight",
        "slight_turn",
        "turn",
        "hard_turn",
        "u_turn",
        "goal_time",
        "cp_time",
    ]

    category_labels = [
        "First Move",
        "Straight\n(0°)",
        "Slight Turn\n(45°)",
        "Turn\n(90°)",
        "Hard Turn\n(135°)",
        "U-Turn\n(180°)",
        "Goal\nPickup Time",
        "CP\nDropoff Time",
    ]

    # Map categories to config parameter names (new time penalty parameters)
    config_param_names = {
        "first_move": "time_first_move",
        "straight": "time_straight",
        "slight_turn": "time_slight_turn",
        "turn": "time_turn",
        "hard_turn": "time_hard_turn",
        "u_turn": "time_u_turn",
        "goal_time": "pickup_time",
        "cp_time": "dropoff_time",
    }

    # Group data by category
    cat_data = {cat: [] for cat in categories}
    for d in transition_data:
        cat = d["category"]
        if cat in cat_data:
            # Convert to float in case values are Decimal
            expected = float(d["expected_duration"])
            actual = float(d["actual_duration"])

            # For time categories (goal_time, cp_time), we just track actual times
            if cat in ["goal_time", "cp_time"]:
                cat_data[cat].append(
                    {
                        "actual": actual,
                        "expected": expected,
                        "distance": 0.0,
                    }
                )
            elif expected > 0:
                # For transition categories, calculate time difference
                # time_diff = actual - expected (positive = took longer, this is the penalty needed)
                time_diff = actual - expected
                cat_data[cat].append(
                    {
                        "time_diff": time_diff,
                        "actual": actual,
                        "expected": expected,
                        "distance": float(d.get("distance", 0)),
                    }
                )

    # Create 2x4 subplot figure
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    # Summary data for terminal output
    summary_data = {}

    for idx, (cat, label) in enumerate(zip(categories, category_labels)):
        ax = axes[idx]
        data = cat_data[cat]

        if not data:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(label, fontsize=10)
            summary_data[cat] = None
            continue

        # Special handling for goal_time and cp_time: plot absolute times
        if cat in ["goal_time", "cp_time"]:
            actual_times = [d["actual"] for d in data]

            # Create box plot of actual times
            bp = ax.boxplot([actual_times], patch_artist=True, widths=0.6)
            bp["boxes"][0].set_facecolor("coral" if cat == "goal_time" else "mediumseagreen")
            bp["boxes"][0].set_alpha(0.7)

            # Add individual points
            x = np.random.normal(1, 0.04, size=len(actual_times))
            ax.scatter(
                x,
                actual_times,
                alpha=0.5,
                s=25,
                c="darkred" if cat == "goal_time" else "darkgreen",
                edgecolors="white",
                linewidths=0.3,
            )

            # Set labels
            ax.set_ylabel("Actual Time [s]", fontsize=8)
            ax.set_title(label, fontsize=10)
            ax.set_xticks([])
            ax.grid(True, alpha=0.3)

            # Statistics
            mean_time = np.mean(actual_times)
            std_time = np.std(actual_times)
            median_time = np.median(actual_times)

            stats_text = f"N={len(data)}\nMean: {mean_time:.3f}s\nMedian: {median_time:.3f}s\nStd: {std_time:.3f}s"
            ax.text(
                0.02,
                0.98,
                stats_text,
                transform=ax.transAxes,
                fontsize=8,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
            )

            # Store summary - for time categories, optimal_value is the mean actual time
            summary_data[cat] = {
                "count": len(data),
                "mean_time": mean_time,
                "median_time": median_time,
                "std_time": std_time,
                "optimal_value": mean_time,  # This will be used as the config value
            }
        else:
            # Transition categories: plot time differences (actual - expected)
            time_diffs = [d["time_diff"] for d in data]

            # Create box plot
            bp = ax.boxplot([time_diffs], patch_artist=True, widths=0.6)
            bp["boxes"][0].set_facecolor("steelblue")
            bp["boxes"][0].set_alpha(0.7)

            # Add individual points
            x = np.random.normal(1, 0.04, size=len(time_diffs))
            ax.scatter(x, time_diffs, alpha=0.5, s=25, c="navy", edgecolors="white", linewidths=0.3)

            # Add reference line at 0 (perfect prediction)
            ax.axhline(y=0, color="red", linestyle="--", linewidth=1.5, alpha=0.7)

            # Set labels
            ax.set_ylabel("Time Penalty (Actual - Expected) [s]", fontsize=8)
            ax.set_title(label, fontsize=10)
            ax.set_xticks([])
            ax.grid(True, alpha=0.3)

            # Add statistics annotation
            mean_diff = np.mean(time_diffs)
            std_diff = np.std(time_diffs)
            median_diff = np.median(time_diffs)

            stats_text = f"N={len(data)}\nMean: {mean_diff:+.3f}s\nMedian: {median_diff:+.3f}s\nStd: {std_diff:.3f}s"
            ax.text(
                0.02,
                0.98,
                stats_text,
                transform=ax.transAxes,
                fontsize=8,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
            )

            # Store summary - optimal_value is the mean time penalty
            summary_data[cat] = {
                "count": len(data),
                "mean_diff": mean_diff,
                "median_diff": median_diff,
                "std_diff": std_diff,
                "optimal_value": max(0.0, mean_diff),  # Time penalty (clamp to >= 0)
            }

    plt.suptitle(f"Transition Timing Analysis (safe_speed_factor = {safe_speed_factor:.3f})", fontsize=14)
    plt.tight_layout()

    # Save plot
    os.makedirs(config.plots_dir, exist_ok=True)
    full_path = os.path.join(config.plots_dir, os.path.basename(output_file))
    plt.savefig(full_path, dpi=150)
    print(f"[TRANSITION_TIMING] Analysis plot saved to {full_path}")
    plt.close(fig)

    # Print comprehensive terminal report
    print("\n" + "=" * 80)
    print("TRANSITION TIMING ANALYSIS - OPTIMAL TIME PENALTIES")
    print("=" * 80)
    print(f"Current safe_speed_factor: {safe_speed_factor:.3f} (DEPRECATED)")
    print(f"Total transitions recorded: {len(transition_data)}")
    print("-" * 80)
    print(f"{'Category':<20} {'Count':>8} {'Mean Penalty':>15} {'Config Param':<25}")
    print("-" * 80)

    for cat, label in zip(categories, category_labels):
        data = summary_data.get(cat)
        label_clean = label.replace("\n", " ")
        config_param = config_param_names.get(cat, cat)

        if data:
            optimal_val = data["optimal_value"]
            print(f"{label_clean:<20} {data['count']:>8} {optimal_val:>+15.3f}s {config_param:<25}")
        else:
            print(f"{label_clean:<20} {'N/A':>8}")

    print("-" * 80)

    # Print config-compatible output that can be directly copied
    print("\n" + "=" * 80)
    print("COPY-PASTE CONFIG VALUES (for PlannerParams in config.py):")
    print("=" * 80)
    print("")
    print("    # Angle-based turn time penalties (added to base travel time)")

    for cat in categories:
        data = summary_data.get(cat)
        config_param = config_param_names.get(cat, cat)

        if data:
            optimal_val = data["optimal_value"]
            # Format with appropriate comment
            if cat == "first_move":
                comment = "  # First move (no previous direction)"
            elif cat == "straight":
                comment = "       # Continue straight (0° turn)"
            elif cat == "slight_turn":
                comment = "    # Slight turn (45° turn)"
            elif cat == "turn":
                comment = "           # Turn (90° turn)"
            elif cat == "hard_turn":
                comment = "      # Hard turn (135° turn)"
            elif cat == "u_turn":
                comment = "        # U-turn (180° turn)"
            elif cat == "goal_time":
                comment = "  # Time to pick up a goal"
            elif cat == "cp_time":
                comment = " # Time to drop off at CP"
            else:
                comment = ""

            print(f"    {config_param}: float = {optimal_val:.2f}{comment}")

    print("")
    print("=" * 80)


def plot_distance_error_over_time(
    sim_context,
    planned_paths_data: dict,
    output_file: str = "distance_error_over_time.png",
):
    """
    Plots the Euclidean distance error (planned - actual position) over time for all agents.

    Args:
        sim_context: The SimContext object containing actual agent position logs
        planned_paths_data: Dict of agent_name -> [(time, x, y, is_poi, poi_type), ...]
                           from planner_data_store.get_planned_paths_data()
        output_file: Output filename for the plot
    """
    import numpy as np
    from scipy.interpolate import interp1d

    if not planned_paths_data:
        print("[DISTANCE_ERROR] No planned paths data available.")
        return

    fig, ax = plt.subplots(figsize=(14, 8))

    # Define colors for agents
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]

    # Track POI markers for legend
    poi_markers_added = {"goal": False, "cp": False}

    agent_idx = 0
    for agent_name, planned_points in planned_paths_data.items():
        if not planned_points:
            continue

        # Get actual position log from sim_context
        if agent_name not in sim_context.log:
            print(f"[DISTANCE_ERROR] No log for agent {agent_name}")
            continue

        agent_log = sim_context.log[agent_name]

        # Extract actual positions and times
        actual_times = list(agent_log.states.timestamps)
        actual_states = list(agent_log.states.values)

        if not actual_times or not actual_states:
            print(f"[DISTANCE_ERROR] No state data for agent {agent_name}")
            continue

        actual_x = [s.x for s in actual_states]
        actual_y = [s.y for s in actual_states]

        # Sort planned points by time
        planned_points = sorted(planned_points, key=lambda p: p[0])

        # Extract planned path data - convert to native floats to avoid scipy object array errors
        planned_times = [float(p[0]) for p in planned_points]
        planned_x = [float(p[1]) for p in planned_points]
        planned_y = [float(p[2]) for p in planned_points]
        poi_info = [(float(p[0]), p[3], p[4]) for p in planned_points]  # (time, is_poi, poi_type)

        if len(planned_times) < 2:
            print(f"[DISTANCE_ERROR] Not enough planned points for {agent_name}")
            continue

        # Create interpolators for planned path
        # Use linear interpolation; extrapolate with boundary values
        try:
            interp_x = interp1d(
                planned_times, planned_x, kind="linear", bounds_error=False, fill_value=(planned_x[0], planned_x[-1])
            )
            interp_y = interp1d(
                planned_times, planned_y, kind="linear", bounds_error=False, fill_value=(planned_y[0], planned_y[-1])
            )
        except Exception as e:
            print(f"[DISTANCE_ERROR] Interpolation failed for {agent_name}: {e}")
            continue

        # Calculate distance error for each actual time step
        error_times = []
        distance_errors = []

        for i, t in enumerate(actual_times):
            # Convert t to native float for scipy compatibility
            t_float = float(t)

            # Only compute within the planning horizon
            if t_float < planned_times[0] or t_float > planned_times[-1]:
                continue

            # Get planned position at this time
            planned_x_t = float(interp_x(t_float))
            planned_y_t = float(interp_y(t_float))

            # Calculate Euclidean distance error
            error = np.hypot(actual_x[i] - planned_x_t, actual_y[i] - planned_y_t)

            error_times.append(t_float)
            distance_errors.append(error)

        if not error_times:
            print(f"[DISTANCE_ERROR] No overlapping time range for {agent_name}")
            continue

        # Plot distance error
        color = colors[agent_idx % len(colors)]
        ax.plot(error_times, distance_errors, color=color, linewidth=1.5, label=agent_name, alpha=0.8)

        # Add X markers at POI times
        for t, is_poi, poi_type in poi_info:
            if is_poi:
                # Find the error at this POI time
                if t >= min(error_times) and t <= max(error_times):
                    # Interpolate error at POI time
                    idx = np.searchsorted(error_times, t)
                    if idx > 0 and idx < len(error_times):
                        error_at_poi = distance_errors[idx]
                    elif idx == 0:
                        error_at_poi = distance_errors[0]
                    else:
                        error_at_poi = distance_errors[-1]
                else:
                    error_at_poi = 0

                if poi_type == "goal":
                    marker_color = "red"
                    label = "Goal Arrival" if not poi_markers_added["goal"] else None
                    poi_markers_added["goal"] = True
                else:  # cp
                    marker_color = "blue"
                    label = "CP Arrival" if not poi_markers_added["cp"] else None
                    poi_markers_added["cp"] = True

                ax.scatter([t], [error_at_poi], marker="x", s=100, c=marker_color, zorder=10, linewidths=2, label=label)

        agent_idx += 1

    # Formatting
    ax.set_xlabel("Time [s]", fontsize=12)
    ax.set_ylabel("Distance Error (Actual - Planned) [m]", fontsize=12)
    ax.set_title(
        "Position Tracking Error Over Time\n(Euclidean distance between planned and actual position)", fontsize=14
    )
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)

    # Add statistics annotation
    if agent_idx > 0:
        # Flatten all errors for summary
        all_errors = []
        for agent_name, planned_points in planned_paths_data.items():
            if agent_name in sim_context.log:
                agent_log = sim_context.log[agent_name]
                actual_times = list(agent_log.states.timestamps)
                actual_states = list(agent_log.states.values)

                if not planned_points or not actual_times:
                    continue

                planned_points = sorted(planned_points, key=lambda p: p[0])
                planned_times = [float(p[0]) for p in planned_points]
                planned_x = [float(p[1]) for p in planned_points]
                planned_y = [float(p[2]) for p in planned_points]

                if len(planned_times) < 2:
                    continue

                try:
                    interp_x = interp1d(
                        planned_times,
                        planned_x,
                        kind="linear",
                        bounds_error=False,
                        fill_value=(planned_x[0], planned_x[-1]),
                    )
                    interp_y = interp1d(
                        planned_times,
                        planned_y,
                        kind="linear",
                        bounds_error=False,
                        fill_value=(planned_y[0], planned_y[-1]),
                    )

                    for i, t in enumerate(actual_times):
                        t_float = float(t)
                        if planned_times[0] <= t_float <= planned_times[-1]:
                            planned_x_t = float(interp_x(t_float))
                            planned_y_t = float(interp_y(t_float))
                            actual_x_t = actual_states[i].x
                            actual_y_t = actual_states[i].y
                            error = np.hypot(actual_x_t - planned_x_t, actual_y_t - planned_y_t)
                            all_errors.append(error)
                except:
                    pass

        if all_errors:
            stats_text = (
                f"Mean Error: {np.mean(all_errors):.3f}m\n"
                f"Max Error: {np.max(all_errors):.3f}m\n"
                f"Std: {np.std(all_errors):.3f}m"
            )
            ax.text(
                0.02,
                0.98,
                stats_text,
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
            )

    plt.tight_layout()

    # Save plot
    os.makedirs(config.plots_dir, exist_ok=True)
    full_path = os.path.join(config.plots_dir, os.path.basename(output_file))
    plt.savefig(full_path, dpi=150)
    print(f"[DISTANCE_ERROR] Plot saved to {full_path}")
    plt.close(fig)

    # Print summary
    print("\n" + "=" * 60)
    print("DISTANCE ERROR ANALYSIS SUMMARY")
    print("=" * 60)
    if all_errors:
        print(f"  Mean position error: {np.mean(all_errors):.3f}m")
        print(f"  Max position error: {np.max(all_errors):.3f}m")
        print(f"  Std deviation: {np.std(all_errors):.3f}m")
    print("=" * 60)

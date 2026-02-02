#!/usr/bin/env python3
"""Generate an animated 3D visualization of space-time reservations."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from goal_collect.planner.planner import FleetPlanner
from goal_collect.utils import animate_spacetime_reservations
from goal_collect.structures import Point
from evaluation.utils_config import load_config, sim_context_from_config


def main():
    print("=" * 70)
    print("GENERATING ANIMATED 3D SPACE-TIME RESERVATIONS")
    print("=" * 70)
    print()

    # Load scenario
    scenario_path = "src/scenarios/config_3.yaml"
    print(f"Loading scenario: {scenario_path}")

    config = load_config(scenario_path)
    sim_context = sim_context_from_config(config)

    # Extract agent starts from models
    agent_starts = {}
    for name, model in sim_context.models.items():
        # Get initial state from model
        state = model.get_state()
        agent_starts[name] = Point(x=state.x, y=state.y, theta=state.psi)

    # Extract goals and collection points
    goals = []
    pois = []
    if sim_context.shared_goals_manager:
        goals = sim_context.shared_goals_manager.shared_goals
        pois = sim_context.shared_goals_manager.collection_points

    print(f"  Agents: {len(agent_starts)}")
    print(f"  Goals: {len(goals)}")
    print(f"  Collection Points: {len(pois)}")
    print()

    # Run planner
    print("Running fleet planner...")
    planner = FleetPlanner()
    plans = planner.plan(
        agent_starts=agent_starts,
        goals=goals,
        collection_points=pois,
        static_obstacles=sim_context.dg_scenario.static_obstacles,
        max_v=2.0,
        max_omega=1.0,
        num_agents=len(agent_starts),
    )
    print()

    # Generate animated visualization
    if planner.reservations:
        print("Creating animated 3D visualization...")
        output_file = "visualizations/reservations_3d_animated.gif"

        animate_spacetime_reservations(
            graph=planner.graph,
            reservations=planner.reservations,
            output_file=output_file,
            max_time=35.0,
            fps=15,  # Lower FPS for faster generation
            duration_seconds=8,
        )

        file_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
        print()
        print("=" * 70)
        print(f"SUCCESS! Animation saved to: {output_file}")
        print(f"File size: {file_size:.1f} MB")
        print("=" * 70)
    else:
        print("⚠ No reservations available - planner did not generate reservation table")


if __name__ == "__main__":
    main()

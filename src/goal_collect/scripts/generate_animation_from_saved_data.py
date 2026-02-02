#!/usr/bin/env python3
"""Generate animated 3D visualization from saved planner data."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from goal_collect import planner_data_store
from goal_collect.utils import animate_spacetime_reservations

# Load saved data
planner_data_store.load_data()

if planner_data_store.reservations and planner_data_store.graph:
    print("=" * 70)
    print("GENERATING ANIMATED 3D VISUALIZATION FROM SAVED DATA")
    print("=" * 70)
    print()
    print(f"Graph nodes: {len(planner_data_store.graph.nodes)}")
    print(f"Geometric reservations: {len(planner_data_store.reservations.geometric_reservations)}")
    print(f"Node reservations: {len(planner_data_store.reservations.node_reservations)}")
    print()

    output_file = "visualizations/reservations_3d_rotating.gif"

    print("Creating animated 3D visualization...")
    animate_spacetime_reservations(
        graph=planner_data_store.graph,
        reservations=planner_data_store.reservations,
        output_file=output_file,
        max_time=35.0,
        fps=15,
        duration_seconds=8,
    )

    if os.path.exists(output_file):
        file_size = os.path.getsize(output_file) / (1024 * 1024)
        print()
        print("=" * 70)
        print(f"SUCCESS! Animation saved to: {output_file}")
        print(f"File size: {file_size:.1f} MB")
        print("=" * 70)
    else:
        print()
        print("⚠ Warning: Animation file was not created")
else:
    print("⚠ Error: No saved planner data found")
    print("Run a scenario first with plotting enabled to generate data")

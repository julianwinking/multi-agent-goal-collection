#!/usr/bin/env python3
"""Create animated 3D visualization from existing planner data."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from goal_collect import planner_data_store
from goal_collect.utils import animate_spacetime_reservations


def main():
    print("Loading saved planner data...")
    planner_data_store.load_data()

    if planner_data_store.graph is None or planner_data_store.reservations is None:
        print("ERROR: No planner data found. Please run a scenario first with advanced_plotting enabled.")
        return

    print(f"Found graph with {len(planner_data_store.graph.nodes)} nodes")
    print(f"Found {len(planner_data_store.reservations.geometric_reservations)} geometric reservations")
    print(f"Found {len(planner_data_store.reservations.node_reservations)} node reservations")
    print()

    print("Creating animated visualization (should take ~30 seconds)...")
    output_file = "visualizations/reservations_3d_rotating.gif"

    animate_spacetime_reservations(
        graph=planner_data_store.graph,
        reservations=planner_data_store.reservations,
        output_file=output_file,
        max_time=35.0,
        fps=25,  # Smoother animation with more fps
        duration_seconds=12,  # Slower rotation (12 second duration = 300 frames)
    )

    if os.path.exists(output_file):
        size_mb = os.path.getsize(output_file) / (1024 * 1024)
        print(f"\n✓ Animation created: {output_file} ({size_mb:.1f} MB)")

        # Create optimized version
        print("\nOptimizing for web display...")
        os.system(
            f'ffmpeg -i {output_file} -vf "fps=12,scale=600:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=64[p];[s1][p]paletteuse=dither=bayer:bayer_scale=2" visualizations/reservations_3d_rotating_opt.gif -y 2>&1 | grep -E "(frame=|size=)" | tail -1'
        )

        if os.path.exists("visualizations/reservations_3d_rotating_opt.gif"):
            opt_size = os.path.getsize("visualizations/reservations_3d_rotating_opt.gif") / (1024 * 1024)
            print(f"✓ Optimized version: reservations_3d_rotating_opt.gif ({opt_size:.1f} MB)")
    else:
        print("ERROR: Animation file was not created")


if __name__ == "__main__":
    main()

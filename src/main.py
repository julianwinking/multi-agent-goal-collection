#!/usr/bin/env python3
"""
Multi-Agent Goal Collection Planner
Main entry point for running simulations and evaluations.
"""
import argparse
import logging
import os
import pprint
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple
from collections import defaultdict

import numpy as np
from reprep import MIME_MP4, Report
from dg_commons.sim.simulator import SimContext, Simulator
from dg_commons.sim.simulator_animation import create_animation

# Ensure src is in the path
_src_dir = Path(__file__).parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from evaluation.perf_metrics import ex14_metrics
from evaluation.utils_config import load_config, sim_context_from_config
from goal_collect import planner_data_store, config


def run_scenario(scenario_config: Mapping[str, Any], output_dir: str = "out") -> Tuple[str, float]:
    """Run a single scenario and return results.

    Args:
        scenario_config: Configuration dictionary for the scenario
        output_dir: Directory for output files

    Returns:
        Tuple of (scenario_name, score)
    """
    sim_context: SimContext = sim_context_from_config(scenario_config)

    print(f"\n{'='*80}")
    print(f"Running Scenario: {sim_context.description}")
    print(f"{'='*80}\n")

    # Clear previous data
    planner_data_store.clear_transition_timing()
    planner_data_store.clear_planned_paths()

    # Run simulation
    sim = Simulator()
    sim.run(sim_context)

    # Create visualization report
    report = create_visualization(sim_context)

    # Compute metrics
    avg_player_metrics, players_metrics = ex14_metrics(sim_context)

    # Calculate score
    score: float = avg_player_metrics.reduce_to_score()
    score_details = avg_player_metrics.get_score_details()

    # Print results
    print(f"\n{'='*80}")
    print("RESULTS:")
    print(f"{'='*80}")
    print(f"Overall Score: {score:.2f}")
    print("\nScore Breakdown:")
    for k, v in score_details.items():
        print(f"  {k}: {v:.2f}")
    print(f"\nMetrics: {avg_player_metrics}")
    print(f"{'='*80}\n")

    # Save results
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    report_file = out_path / "index.html"
    report.to_html(str(report_file))
    print(f"Report saved to: {report_file}")

    # Save timing information
    timing_file = out_path / "timing_report.txt"
    with open(timing_file, "a") as f:
        f.write(f"\n--- Simulation: {sim_context.description} ---\n")
        f.write(f"Global Planner Time: {avg_player_metrics.global_planning_time:.4f} s\n")
        f.write(f"Average Agent Computation Time: {avg_player_metrics.avg_computation_time:.4f} s\n")
        f.write("Score Breakdown:\n")
        for k, v in score_details.items():
            f.write(f"  {k}: {v:.2f}\n")
        f.write(f"Total Score: {score:.2f}\n")

        for pm in players_metrics:
            f.write(f"Agent {pm.player_name}:\n")
            f.write(f"  Avg Computation Time: {pm.avg_computation_time:.4f} s\n")
            if pm.avg_computation_time > 0.1:
                f.write(f"  WARNING: Exceeded 0.1s computation limit!\n")

    return sim_context.description, score


def create_visualization(sim_context: SimContext) -> Report:
    """Create visualization report for a simulation.

    Args:
        sim_context: The simulation context

    Returns:
        Report object with visualization
    """
    r = Report(f"MultiAgentPlanner-{sim_context.description}")
    gif_viz = r.figure(cols=1)

    with gif_viz.data_file("Animation", MIME_MP4) as fn:
        create_animation(
            file_path=fn,
            sim_context=sim_context,
            figsize=(16, 16),
            dt=50,
            dpi=50,
            plot_limits=[[-12, 12], [-12, 12]],
        )

    return r


def load_scenarios(scenario_dir: str = "src/scenarios") -> List[Mapping[str, Any]]:
    """Load scenario configurations from directory.

    Args:
        scenario_dir: Directory containing YAML scenario files

    Returns:
        List of scenario configurations
    """
    scenario_path = Path(scenario_dir)

    # Default scenarios to run
    default_configs = [
        "config_1.yaml",
        "config_2.yaml",
        "config_3.yaml",
    ]

    scenarios = []
    for config_file in default_configs:
        config_path = scenario_path / config_file
        if config_path.exists():
            scenarios.append(load_config(str(config_path)))
            print(f"Loaded scenario: {config_file}")
        else:
            print(f"Warning: Scenario file not found: {config_path}")

    return scenarios


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Multi-Agent Goal Collection Planner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/main.py                    # Run default scenarios
  python src/main.py --scenario config_1.yaml
  python src/main.py --all-scenarios    # Run all scenarios in scenarios/
        """,
    )
    parser.add_argument("-s", "--scenario", help="Specific scenario YAML file to run", type=str)
    parser.add_argument("-a", "--all-scenarios", help="Run all scenarios in scenarios directory", action="store_true")
    parser.add_argument("-o", "--output", help="Output directory for results", type=str, default="out")
    parser.add_argument("-v", "--verbose", help="Enable verbose output", action="store_true")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s: %(message)s")

    # Suppress annoying reprep messages
    logging.getLogger("reprep").setLevel(logging.WARNING)

    # Load scenarios
    if args.scenario:
        # Run specific scenario
        scenario_path = Path("src/scenarios") / args.scenario
        if not scenario_path.exists():
            print(f"Error: Scenario file not found: {scenario_path}")
            sys.exit(1)
        scenarios = [load_config(str(scenario_path))]
    elif args.all_scenarios:
        # Run all scenarios
        scenario_dir = Path("src/scenarios")
        yaml_files = sorted(scenario_dir.glob("*.yaml"))
        scenarios = [load_config(str(f)) for f in yaml_files]
        print(f"Found {len(scenarios)} scenarios to run")
    else:
        # Run default scenarios
        scenarios = load_scenarios()

    if not scenarios:
        print("Error: No scenarios to run!")
        sys.exit(1)

    # Run scenarios
    results = []
    for scenario in scenarios:
        try:
            result = run_scenario(scenario, output_dir=args.output)
            results.append(result)
        except Exception as e:
            print(f"\nError running scenario: {e}")
            import traceback

            traceback.print_exc()
            continue

    # Print summary
    if len(results) > 1:
        print(f"\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}")
        for name, score in results:
            print(f"  {name}: {score:.2f}")

        avg_score = np.mean([score for _, score in results])
        print(f"\nAverage Score: {avg_score:.2f}")
        print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

"""
Iterative Greedy Planner for Multi-Agent Goal Collection.

This module implements a simple heuristic-based planner that assigns goals to agents
by minimizing distance to goals and then to collection points.

Algorithm:
1. In each round, assign one mission (goal + closest CP) to each agent that minimizes
   the maximum finish time (makespan minimization).
2. Repeat until all goals are assigned.
"""

import math
from typing import Dict, List, Set, Tuple

from ..config import config
from ..structures import GlobalSolution


def solve_iterative_greedy(
    agents: List[str],
    goal_ids: List[str],
    cp_ids: List[str],
    agent_to_goal: Dict[str, Dict[str, float]],
    goal_to_cp: Dict[str, Dict[str, float]],
    cp_to_goal: Dict[str, Dict[str, float]],
    safe_speed: float,
) -> GlobalSolution:
    """
    Solves the Multi-Agent VRP using an Iterative Greedy approach.
    
    In each iteration:
    1. For each unassigned goal, find the best (agent, goal, CP) assignment
       that minimizes the finish time (current agent accumulated time + travel to goal + travel to CP).
    2. Pick the assignment with the smallest finish time.
    3. Update the agent's virtual state and repeat until all goals are assigned.
    
    Args:
        agents: List of agent names
        goal_ids: List of goal IDs to assign
        cp_ids: List of collection point IDs
        agent_to_goal: Pre-computed distances from each agent start to each goal
        goal_to_cp: Pre-computed distances from each goal to each CP
        cp_to_goal: Pre-computed distances from each CP to each goal
        safe_speed: Safe speed for time calculations
        
    Returns:
        GlobalSolution with sequences, cp_assignments, estimated_times, etc.
    """
    print(f"[Iterative Greedy] Starting solver for {len(agents)} agents, {len(goal_ids)} goals.")
    
    if not goal_ids:
        return GlobalSolution(
            sequences={agent: [] for agent in agents},
            cp_assignments={},
            estimated_times={agent: {} for agent in agents},
            makespan=0.0,
            total_distance=0.0,
            success=True,
            status="No goals",
        )
    
    # Initialize result structures
    routes: Dict[str, List[str]] = {agent: [] for agent in agents}
    cp_assignments: Dict[str, str] = {}
    estimated_times: Dict[str, Dict[str, float]] = {agent: {} for agent in agents}
    
    # Virtual agent state: tracks accumulated time and current "position" 
    # (the CP where the agent ends up after its last delivery)
    agent_accum_time: Dict[str, float] = {agent: 0.0 for agent in agents}
    agent_last_cp: Dict[str, str] = {}  # Empty initially (agents start from their spawn)
    
    # Pending goals
    pending_goals: Set[str] = set(goal_ids)
    
    # Helper: Get best CP for a goal (minimizes goal -> CP distance)
    def get_best_cp(goal_id: str) -> Tuple[str, float]:
        """Returns (best_cp_id, distance_to_cp) for the given goal."""
        best_cpid = None
        best_dist = math.inf
        for cpid in cp_ids:
            d = goal_to_cp[goal_id].get(cpid, math.inf)
            if d < best_dist:
                best_dist = d
                best_cpid = cpid
        return best_cpid, best_dist
    
    # Main assignment loop
    iteration = 0
    while pending_goals:
        iteration += 1
        if iteration % 1 == 0:
            print(f"[Iterative Greedy] Round {iteration}: {len(pending_goals)} goals remaining...")
        
        best_assignment = None  # (agent, goal_id, cp_id, finish_time, dist_to_goal, dist_to_cp)
        min_finish_time = math.inf
        
        # Evaluate all (agent, goal) pairs
        for goal_id in list(pending_goals):
            for agent in agents:
                current_time = agent_accum_time[agent]
                
                # Calculate distance to goal
                if agent not in agent_last_cp:
                    # Agent hasn't delivered yet -> use direct agent-to-goal distance
                    dist_to_goal = agent_to_goal[agent].get(goal_id, math.inf)
                else:
                    # Agent is at a CP -> use CP-to-goal distance
                    last_cp = agent_last_cp[agent]
                    dist_to_goal = cp_to_goal.get(last_cp, {}).get(goal_id, math.inf)
                
                if math.isinf(dist_to_goal):
                    continue
                
                # Get best CP for this goal
                best_cpid, dist_to_cp = get_best_cp(goal_id)
                if best_cpid is None or math.isinf(dist_to_cp):
                    continue
                
                # Calculate finish time for this task
                travel_to_goal_time = dist_to_goal / safe_speed # + config.planner.pickup_time
                travel_to_cp_time = dist_to_cp / safe_speed # + config.planner.dropoff_time
                finish_time = current_time + travel_to_goal_time + travel_to_cp_time
                
                if finish_time < min_finish_time:
                    min_finish_time = finish_time
                    best_assignment = (agent, goal_id, best_cpid, finish_time, dist_to_goal, dist_to_cp)
        
        if best_assignment is None:
            # No valid assignments possible
            print(f"[Iterative Greedy] WARNING: No valid assignments found for {len(pending_goals)} remaining goals!")
            break
        
        # Commit the best assignment
        agent, goal_id, cp_id, finish_time, dist_to_goal, dist_to_cp = best_assignment
        
        routes[agent].append(goal_id)
        cp_assignments[goal_id] = cp_id
        
        # Update estimated times for goal arrival
        travel_time_to_goal = dist_to_goal / safe_speed # + config.planner.pickup_time
        arrival_time = agent_accum_time[agent] + travel_time_to_goal
        estimated_times[agent][goal_id] = arrival_time
        
        # Update agent's virtual state
        agent_accum_time[agent] = finish_time
        agent_last_cp[agent] = cp_id
        
        pending_goals.remove(goal_id)
    
    # Calculate final metrics
    final_makespan = max(agent_accum_time.values()) if agent_accum_time else 0.0
    
    # Calculate total distance
    total_distance = 0.0
    for agent, route in routes.items():
        if not route:
            continue
        
        # Distance from start to first goal
        first_goal = route[0]
        total_distance += agent_to_goal[agent].get(first_goal, 0.0)
        
        # Distance for each goal -> CP (-> next goal)
        for i, goal_id in enumerate(route):
            cpid = cp_assignments.get(goal_id)
            if cpid:
                total_distance += goal_to_cp[goal_id].get(cpid, 0.0)
                
                # If there's a next goal, add CP -> next goal distance
                if i < len(route) - 1:
                    next_goal = route[i + 1]
                    total_distance += cp_to_goal.get(cpid, {}).get(next_goal, 0.0)
    
    success = len(pending_goals) == 0
    
    print(f"[Iterative Greedy] Completed: makespan={final_makespan:.2f}s, distance={total_distance:.2f}")
    for agent, route in routes.items():
        if route:
            cp_info = " -> ".join([f"{g}(CP:{cp_assignments.get(g, '?')})" for g in route])
            print(f"    {agent}: {cp_info}")
    
    return GlobalSolution(
        sequences=routes,
        cp_assignments=cp_assignments,
        estimated_times=estimated_times,
        makespan=final_makespan,
        total_distance=total_distance,
        success=success,
        status="Success" if success else f"Partial ({len(pending_goals)} unassigned)",
    )

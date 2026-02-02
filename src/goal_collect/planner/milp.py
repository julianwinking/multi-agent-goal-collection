"""
MILP-based VRP Solver.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import pulp

from ..config import config
from ..structures import GlobalSolution


def solve_milp_vrp(
    agents: List[str],
    goal_ids: List[str],
    cp_ids: List[str],
    agent_to_goal: Dict[str, Dict[str, float]],
    goal_to_cp: Dict[str, Dict[str, float]],
    cp_to_goal: Dict[str, Dict[str, float]],
    safe_speed: float,
) -> GlobalSolution:
    """
    Solve the VRP using MILP with:
    - Makespan + Distance minimization
    - CP-aware chain variables: x_chain_cp[agent, g1, cp, g2]
      This properly optimizes which CP to use based on the next goal!
    - Time-based scheduling
    - Per-agent MTZ subtour elimination
    """
    cfg = config.milp

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

    prob = pulp.LpProblem("fleet_vrp_milp", sense=pulp.LpMinimize)

    # Get CP options per goal ranked by best goal->CP->next-goal cost.
    goal_cp_options: Dict[str, List[str]] = {}
    for gid in goal_ids:
        cp_scores: List[Tuple[float, str]] = []
        for cpid in cp_ids:
            direct = goal_to_cp[gid].get(cpid, math.inf)
            if math.isinf(direct):
                continue

            best_chain = math.inf
            for next_gid in goal_ids:
                if next_gid == gid:
                    continue
                next_dist = cp_to_goal.get(cpid, {}).get(next_gid, math.inf)
                if math.isinf(next_dist):
                    continue
                best_chain = min(best_chain, direct + next_dist)

            score = best_chain if not math.isinf(best_chain) else direct
            if math.isinf(score):
                continue
            cp_scores.append((score, cpid))

        cp_scores.sort(key=lambda item: item[0])
        goal_cp_options[gid] = [cpid for _, cpid in cp_scores[: cfg.max_cp_options]]

    # Filter out goals with no reachable CPs
    valid_goals = [gid for gid in goal_ids if goal_cp_options.get(gid)]
    if not valid_goals:
        return GlobalSolution(
            sequences={agent: [] for agent in agents},
            cp_assignments={},
            estimated_times={agent: {} for agent in agents},
            makespan=0.0,
            total_distance=0.0,
            success=False,
            status="No valid goals with reachable CPs",
        )

    n_goals = len(valid_goals)
    n_agents = len(agents)

    print(f"  > MILP: {n_agents} agents, {n_goals} goals, {len(cp_ids)} CPs")

    # =========================================================================
    # DECISION VARIABLES
    # =========================================================================

    # x_start[agent, goal]: agent starts by going to this goal
    x_start: Dict[Tuple[str, str], pulp.LpVariable] = {}
    for agent in agents:
        for gid in valid_goals:
            if not math.isinf(agent_to_goal[agent].get(gid, math.inf)):
                x_start[(agent, gid)] = pulp.LpVariable(f"x_start_{agent}_{gid}", cat="Binary")

    # Precompute feasible CP-aware chains independent of agents.
    chain_distances: Dict[Tuple[str, str, str], float] = {}
    min_chain_dist: Dict[Tuple[str, str], float] = {}
    for g1 in valid_goals:
        for cpid in goal_cp_options.get(g1, []):
            d1 = goal_to_cp[g1].get(cpid, math.inf)
            if math.isinf(d1):
                continue
            for g2 in valid_goals:
                if g1 == g2:
                    continue
                d2 = cp_to_goal.get(cpid, {}).get(g2, math.inf)
                if math.isinf(d2):
                    continue
                chain_dist = d1 + d2
                chain_distances[(g1, cpid, g2)] = chain_dist
                pair_key = (g1, g2)
                prev = min_chain_dist.get(pair_key, math.inf)
                if chain_dist < prev:
                    min_chain_dist[pair_key] = chain_dist

    # Apply pruning based on best chain distance
    pruned_chain_distances: Dict[Tuple[str, str, str], float] = {}
    for (g1, cpid, g2), chain_dist in chain_distances.items():
        best = min_chain_dist.get((g1, g2), math.inf)
        if math.isinf(best):
            continue
        if chain_dist <= best * cfg.edge_prune_factor:
            pruned_chain_distances[(g1, cpid, g2)] = chain_dist

    # x_chain_cp[agent, goal1, cp, goal2]: agent delivers goal1 to cp, then goes to goal2
    # Instantiate only if the agent can reach g1 from its start.
    x_chain_cp: Dict[Tuple[str, str, str, str], pulp.LpVariable] = {}
    chain_dist_cache: Dict[Tuple[str, str, str], float] = dict(pruned_chain_distances)
    for (g1, cpid, g2), chain_dist in pruned_chain_distances.items():
        for agent in agents:
            if math.isinf(agent_to_goal[agent].get(g1, math.inf)):
                continue
            x_chain_cp[(agent, g1, cpid, g2)] = pulp.LpVariable(
                f"x_chain_{agent}_{g1}_{cpid}_{g2}", cat="Binary"
            )

    print(f"  > Chain variables: {len(x_chain_cp)}")

    # x_end[agent, goal, cp]: agent ends at goal and delivers to cp
    x_end_cp: Dict[Tuple[str, str, str], pulp.LpVariable] = {}
    for agent in agents:
        for gid in valid_goals:
            for cpid in goal_cp_options.get(gid, []):
                d = goal_to_cp[gid].get(cpid, math.inf)
                if not math.isinf(d):
                    x_end_cp[(agent, gid, cpid)] = pulp.LpVariable(f"x_end_{agent}_{gid}_{cpid}", cat="Binary")

    # Time variables
    big_m_time = 1000.0  # Large constant for time
    t_arrival: Dict[Tuple[str, str], pulp.LpVariable] = {}
    for agent in agents:
        for gid in valid_goals:
            t_arrival[(agent, gid)] = pulp.LpVariable(f"t_arr_{agent}_{gid}", lowBound=0, upBound=big_m_time)

    t_finish: Dict[str, pulp.LpVariable] = {}
    for agent in agents:
        t_finish[agent] = pulp.LpVariable(f"t_finish_{agent}", lowBound=0, upBound=big_m_time)

    makespan = pulp.LpVariable("makespan", lowBound=0, upBound=big_m_time)

    # MTZ subtour elimination variables (per agent)
    u_vars: Dict[Tuple[str, str], pulp.LpVariable] = {}
    for agent in agents:
        for gid in valid_goals:
            u_vars[(agent, gid)] = pulp.LpVariable(f"u_{agent}_{gid}", lowBound=0, upBound=n_goals)

    # =========================================================================
    # OBJECTIVE FUNCTION
    # =========================================================================

    distance_terms = []

    # Distance from start to first goal
    for (agent, gid), var in x_start.items():
        dist = agent_to_goal[agent].get(gid, 0)
        if not math.isinf(dist):
            distance_terms.append(dist * var)

    # Distance for CP-aware chains (g1 -> CP -> g2)
    # Now we use the EXACT chain distance for each CP choice!
    for (agent, g1, cpid, g2), var in x_chain_cp.items():
        chain_dist = chain_dist_cache.get((g1, cpid, g2), 0)
        if chain_dist > 0:
            distance_terms.append(chain_dist * var)

    # Distance from last goal to CP (for final delivery)
    for (agent, gid, cpid), var in x_end_cp.items():
        dist = goal_to_cp[gid].get(cpid, 0)
        if not math.isinf(dist):
            distance_terms.append(dist * var)

    total_distance = pulp.lpSum(distance_terms) if distance_terms else 0

    # Objective: minimize weighted sum of makespan and distance
    prob += cfg.makespan_weight * makespan + cfg.distance_weight * total_distance

    # =========================================================================
    # CONSTRAINTS
    # =========================================================================

    # 1. Each goal is visited exactly once
    for gid in valid_goals:
        incoming_terms = []
        for agent in agents:
            # From start
            var = x_start.get((agent, gid))
            if var is not None:
                incoming_terms.append(var)
            # From chain (any CP)
            for prev_gid in valid_goals:
                if prev_gid == gid:
                    continue
                for cpid in goal_cp_options.get(prev_gid, []):
                    var = x_chain_cp.get((agent, prev_gid, cpid, gid))
                    if var is not None:
                        incoming_terms.append(var)
        prob += pulp.lpSum(incoming_terms) == 1, f"visit_once_{gid}"

    # 2. Flow conservation per agent
    for agent in agents:
        start_terms = [var for (ag, _), var in x_start.items() if ag == agent]
        end_terms = [var for (ag, _, _), var in x_end_cp.items() if ag == agent]
        prob += pulp.lpSum(start_terms) <= 1, f"start_limit_{agent}"
        prob += pulp.lpSum(end_terms) == pulp.lpSum(start_terms), f"end_balance_{agent}"

        # Flow at each goal: incoming = outgoing
        for gid in valid_goals:
            incoming = []
            outgoing = []

            # Incoming from start
            var = x_start.get((agent, gid))
            if var is not None:
                incoming.append(var)

            # Incoming from chain (any CP, from any prev goal)
            for prev_gid in valid_goals:
                if prev_gid == gid:
                    continue
                for cpid in goal_cp_options.get(prev_gid, []):
                    var = x_chain_cp.get((agent, prev_gid, cpid, gid))
                    if var is not None:
                        incoming.append(var)

            # Outgoing via chain (any CP, to any next goal)
            for next_gid in valid_goals:
                if next_gid == gid:
                    continue
                for cpid in goal_cp_options.get(gid, []):
                    var = x_chain_cp.get((agent, gid, cpid, next_gid))
                    if var is not None:
                        outgoing.append(var)

            # Outgoing via end (any CP)
            for cpid in goal_cp_options.get(gid, []):
                var = x_end_cp.get((agent, gid, cpid))
                if var is not None:
                    outgoing.append(var)

            prob += pulp.lpSum(outgoing) - pulp.lpSum(incoming) == 0, f"flow_{agent}_{gid}"

    # 3. Time constraints
    for agent in agents:
        # Time to reach first goal
        for gid in valid_goals:
            var = x_start.get((agent, gid))
            if var is None:
                continue
            dist = agent_to_goal[agent].get(gid, math.inf)
            if math.isinf(dist):
                continue
            travel_time = dist / safe_speed
            prob += t_arrival[(agent, gid)] >= travel_time * var, f"t_start_{agent}_{gid}"

        # Time for CP-aware chains
        for (ag, g1, cpid, g2), var in x_chain_cp.items():
            if ag != agent:
                continue
            chain_dist = chain_dist_cache.get((g1, cpid, g2), math.inf)
            if math.isinf(chain_dist):
                continue
            chain_time = chain_dist / safe_speed

            # Big-M constraint: t_arrival[g2] >= t_arrival[g1] + chain_time if var = 1
            prob += (
                t_arrival[(agent, g2)] >= t_arrival[(agent, g1)] + chain_time - big_m_time * (1 - var)
            ), f"t_chain_{agent}_{g1}_{cpid}_{g2}"

        # Finish time
        for (ag, gid, cpid), var in x_end_cp.items():
            if ag != agent:
                continue
            d = goal_to_cp[gid].get(cpid, math.inf)
            if math.isinf(d):
                continue
            delivery_time = d / safe_speed

            prob += (
                t_finish[agent] >= t_arrival[(agent, gid)] + delivery_time - big_m_time * (1 - var)
            ), f"t_finish_{agent}_{gid}_{cpid}"

        # Ensure finish >= 0 (for agents with no tasks)
        prob += t_finish[agent] >= 0, f"t_finish_min_{agent}"

    # 4. Makespan constraint
    for agent in agents:
        prob += makespan >= t_finish[agent], f"makespan_{agent}"

    # 5. MTZ subtour elimination (per agent)
    # We need to aggregate chain_cp variables to determine if agent goes g1 -> g2
    for agent in agents:
        for g1 in valid_goals:
            for g2 in valid_goals:
                if g1 == g2:
                    continue
                # Sum of all CPs for this (agent, g1, g2) transition
                chain_vars = []
                for cpid in goal_cp_options.get(g1, []):
                    var = x_chain_cp.get((agent, g1, cpid, g2))
                    if var is not None:
                        chain_vars.append(var)

                if chain_vars:
                    # MTZ: u[g1] - u[g2] + n * sum(chains) <= n - 1
                    prob += (
                        u_vars[(agent, g1)] - u_vars[(agent, g2)] + n_goals * pulp.lpSum(chain_vars) <= n_goals - 1
                    ), f"mtz_{agent}_{g1}_{g2}"

    # =========================================================================
    # SOLVE
    # =========================================================================

    solver = pulp.PULP_CBC_CMD(
        msg=False,
        timeLimit=cfg.time_limit,
        gapRel=cfg.gap_tolerance,
        threads=cfg.threads,
    )

    print(f"  > Solving MILP (time_limit={cfg.time_limit}s, gap={cfg.gap_tolerance})...")

    try:
        result = prob.solve(solver)
    except Exception as exc:
        print(f"  > ERROR: MILP solver failed with exception: {exc}")
        return GlobalSolution(
            sequences={agent: [] for agent in agents},
            cp_assignments={},
            estimated_times={agent: {} for agent in agents},
            makespan=0.0,
            total_distance=0.0,
            success=False,
            status=f"Solver exception: {exc}",
        )

    status = pulp.LpStatus.get(result, pulp.LpStatus[prob.status])
    if status not in ("Optimal", "Feasible"):
        print(f"  > WARN: MILP solver status {pulp.LpStatus[prob.status]}")
        return GlobalSolution(
            sequences={agent: [] for agent in agents},
            cp_assignments={},
            estimated_times={agent: {} for agent in agents},
            makespan=0.0,
            total_distance=0.0,
            success=False,
            status=f"Solver status: {status}",
        )

    # =========================================================================
    # EXTRACT SOLUTION
    # =========================================================================

    sequences: Dict[str, List[str]] = {agent: [] for agent in agents}
    start_successor: Dict[str, str] = {}
    next_goal_map: Dict[Tuple[str, str], Tuple[str, str]] = {}  # (agent, g1) -> (cp, g2)

    for (agent, gid), var in x_start.items():
        val = pulp.value(var)
        if val and val > 0.5:
            start_successor[agent] = gid

    for (agent, g_from, cpid, g_to), var in x_chain_cp.items():
        val = pulp.value(var)
        if val and val > 0.5:
            next_goal_map[(agent, g_from)] = (cpid, g_to)

    # Extract CP assignments from chains and ends
    cp_assignments: Dict[str, str] = {}

    for agent in agents:
        route: List[str] = []
        current_goal = start_successor.get(agent)
        while current_goal:
            route.append(current_goal)
            chain_info = next_goal_map.get((agent, current_goal))
            if chain_info:
                cp_used, next_goal = chain_info
                cp_assignments[current_goal] = cp_used
                current_goal = next_goal
            else:
                # This is the last goal - find which CP was used
                for (ag, gid, cpid), var in x_end_cp.items():
                    if ag == agent and gid == current_goal:
                        val = pulp.value(var)
                        if val and val > 0.5:
                            cp_assignments[current_goal] = cpid
                            break
                current_goal = None
        sequences[agent] = route

    # Extract times
    estimated_times: Dict[str, Dict[str, float]] = {agent: {} for agent in agents}
    for (agent, gid), var in t_arrival.items():
        val = pulp.value(var)
        if val is not None:
            estimated_times[agent][gid] = val

    final_makespan = pulp.value(makespan) or 0.0
    final_distance = pulp.value(total_distance) if isinstance(total_distance, pulp.LpAffineExpression) else 0.0

    # Verify solution
    assigned = sum(len(route) for route in sequences.values())
    if assigned != len(valid_goals):
        print(f"  > WARN: MILP produced inconsistent assignment ({assigned} vs {len(valid_goals)}).")
        return GlobalSolution(
            sequences=sequences,
            cp_assignments=cp_assignments,
            estimated_times=estimated_times,
            makespan=final_makespan,
            total_distance=final_distance,
            success=False,
            status=f"Inconsistent assignment: {assigned} vs {len(valid_goals)}",
        )

    print(f"  > MILP solved: makespan={final_makespan:.2f}s, distance={final_distance:.2f}")
    for agent, route in sequences.items():
        if route:
            cp_info = " -> ".join([f"{g}(CP:{cp_assignments.get(g, '?')})" for g in route])
            print(f"    {agent}: {cp_info}")

    return GlobalSolution(
        sequences=sequences,
        cp_assignments=cp_assignments,
        estimated_times=estimated_times,
        makespan=final_makespan,
        total_distance=final_distance,
        success=True,
        status="Success",
    )


import math
import copy
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass

from ..config import config
from ..structures import GlobalSolution


@dataclass
class InsertionMove:
    agent: str
    goal_id: str
    index: int  # Insertion index (0 <= index <= len(current_route))
    cost_increase: float


def solve_regret_insertion(
    agents: List[str],
    goal_ids: List[str],
    cp_ids: List[str],
    agent_to_goal: Dict[str, Dict[str, float]],
    goal_to_cp: Dict[str, Dict[str, float]],
    cp_to_goal: Dict[str, Dict[str, float]],
    safe_speed: float,
    goal_coords: Dict[str, Tuple[float, float]],  # Added for spatial conflict check
) -> GlobalSolution:
    """
    Solves the Multi-Agent VRP using a Regret-Based Insertion Heuristic.
    Includes spatial conflict checking and iterative refinement.
    """
    cfg = config.regret
    print(f"[Regret-Based Insertion] Starting solver for {len(agents)} agents, {len(goal_ids)} goals.")
    
    # 1. State Initialization
    routes: Dict[str, List[str]] = {agent: [] for agent in agents}
    unassigned_goals = set(goal_ids)
    
    # Precompute valid CP choices for every Goal->Goal transition
    best_link_cache: Dict[Tuple[str, str], Tuple[str, float]] = {}
    
    def get_best_link(g1: str, g2: str) -> Tuple[str, float]:
        if (g1, g2) in best_link_cache:
            return best_link_cache[(g1, g2)]
        best_cpid = None
        best_dist = math.inf
        for cpid in cp_ids:
            d1 = goal_to_cp[g1].get(cpid, math.inf)
            if math.isinf(d1): continue
            d2 = cp_to_goal[cpid].get(g2, math.inf)
            if math.isinf(d2): continue
            if d1 + d2 < best_dist:
                best_dist = d1 + d2
                best_cpid = cpid
        best_link_cache[(g1, g2)] = (best_cpid, best_dist)
        return best_cpid, best_dist

    def get_best_end(g1: str) -> Tuple[str, float]:
        best_dist = math.inf
        best_cpid = None
        for cpid in cp_ids:
            d = goal_to_cp[g1].get(cpid, math.inf)
            if d < best_dist:
                best_dist = d
                best_cpid = cpid
        return best_cpid, best_dist

    # 2. Heuristic Loop
    while unassigned_goals:
        print(f"[Regret-Based Insertion] {len(unassigned_goals)} goals remaining...")
        
        goal_insertions: Dict[str, List[InsertionMove]] = {}
        
        # Precompute current metrics for all agents to calculate marginal costs
        current_agent_times = {}
        current_agent_dists = {}
        for agent in agents:
            current_agent_times[agent] = _agent_time(
                agent, routes[agent], agent_to_goal, goal_to_cp, cp_to_goal, 
                best_link_cache, safe_speed, fixed_cps=None, get_best_link_fn=get_best_link
            )
            current_agent_dists[agent] = _agent_dist(
                agent, routes[agent], agent_to_goal, goal_to_cp, cp_to_goal, 
                best_link_cache, fixed_cps=None, get_best_link_fn=get_best_link
            )
        
        for gid in unassigned_goals:
            moves: List[InsertionMove] = []
            for agent in agents:
                current_route = routes[agent]
                for idx in range(len(current_route) + 1):
                    cost_delta, new_cpid = _evaluate_insertion(
                        agent, gid, idx, current_route,
                        routes,  # Pass all routes for conflict check
                        agent_to_goal, goal_to_cp, cp_to_goal,
                        get_best_link, get_best_end, safe_speed,

                        current_agent_times[agent],
                        current_agent_dists[agent],
                        goal_coords, cfg
                    )
                    
                    if math.isinf(cost_delta):
                        continue
                    moves.append(InsertionMove(agent, gid, idx, cost_delta))
            
            moves.sort(key=lambda x: x.cost_increase)
            goal_insertions[gid] = moves

        # Calculate Regret
        best_goal = None
        max_regret = -1.0
        best_move = None
        
        for gid, moves in goal_insertions.items():
            if not moves: continue
            if len(moves) == 1:
                regret = float('inf')
                move = moves[0]
            else:
                best = moves[0]
                second = moves[1]
                regret = (second.cost_increase - best.cost_increase)
                if cfg.p > 1:
                    pass
                move = best
            
            if regret > max_regret:
                max_regret = regret
                best_goal = gid
                best_move = move
            
        if best_goal is None:
            print(f"[Regret-Based Insertion] No valid insertions found for {len(unassigned_goals)} goals!")
            break
            
        agent = best_move.agent
        routes[agent].insert(best_move.index, best_goal)
        unassigned_goals.remove(best_goal)
        print(f"[Regret-Based Insertion] Assigned Goal {best_goal} to {agent}. Cost Increase: {best_move.cost_increase:.2f}")

    # 3. Refinement Phase
    print("[Regret-Based Insertion] Starting Refinement Phase...")
    cp_assignments: Dict[str, str] = {}
    for _ in range(cfg.refinement_iterations):
        changed = False
        for agent, route in routes.items():
            if not route: continue
            for i, gid in enumerate(route):
                if i < len(route) - 1:
                    next_gid = route[i+1]
                    best_cpid, _ = get_best_link(gid, next_gid)
                else:
                    best_cpid, _ = get_best_end(gid)
                
                if cp_assignments.get(gid) != best_cpid:
                    cp_assignments[gid] = best_cpid
                    changed = True
        if not changed:
            break

    # 4. Construct Solution
    final_makespan = _calculate_makespan(routes, agents, agent_to_goal, goal_to_cp, cp_to_goal, best_link_cache, safe_speed, fixed_cps=cp_assignments, get_best_link_fn=get_best_link)
    final_distance = _calculate_total_distance(routes, agents, agent_to_goal, goal_to_cp, cp_to_goal, best_link_cache, fixed_cps=cp_assignments, get_best_link_fn=get_best_link)
    
    estimated_times = {agent: {} for agent in agents}
    for agent, route in routes.items():
        if not route: continue
        curr_time = 0.0
        # Start -> First
        d = agent_to_goal[agent][route[0]]
        curr_time += d / safe_speed
        estimated_times[agent][route[0]] = curr_time
        
        for i in range(len(route)):
            gid = route[i]
            cpid = cp_assignments[gid]
            # Goal -> CP
            d_cp = goal_to_cp[gid][cpid]
            curr_time += d_cp / safe_speed
            
            if i < len(route) - 1:
                next_gid = route[i+1]
                # CP -> Next Goal
                d_next = cp_to_goal[cpid][next_gid]
                curr_time += d_next / safe_speed
                estimated_times[agent][next_gid] = curr_time

    success = len(unassigned_goals) == 0
    return GlobalSolution(
        sequences={a: [str(g) for g in r] for a, r in routes.items()},
        cp_assignments=cp_assignments,
        estimated_times=estimated_times,
        makespan=final_makespan,
        total_distance=final_distance,
        success=success,
        status="Success" if success else "Partial"
    )

def _evaluate_insertion(
    agent, gid, idx, current_route,
    all_routes, # Added
    agent_to_goal, goal_to_cp, cp_to_goal,
    get_best_link_fn, get_best_end_fn, safe_speed,
    current_agent_time,
    current_agent_dist,
    goal_coords, cfg
) -> Tuple[float, Optional[str]]:
    
    temp_route = current_route[:idx] + [gid] + current_route[idx:]
    
    dist = 0.0
    time = 0.0
    
    if not temp_route:
        return 0.0, None

    # Calculate agent metrics
    first = temp_route[0]
    d = agent_to_goal[agent].get(first, math.inf)
    if math.isinf(d): return math.inf, None
    dist += d
    time += d / safe_speed
    
    for i in range(len(temp_route) - 1):
        curr = temp_route[i]
        nex = temp_route[i+1]
        cpid, link_dist = get_best_link_fn(curr, nex)
        if math.isinf(link_dist): return math.inf, None
        dist += link_dist
        time += link_dist / safe_speed
        
    last = temp_route[-1]
    last_cpid, last_dist = get_best_end_fn(last)
    if math.isinf(last_dist): return math.inf, None
    dist += last_dist
    time += last_dist / safe_speed
    
    # Spatial Conflict Check
    conflict_cost = 0.0
    if cfg.conflict_radius > 0:
        c_new = goal_coords.get(gid)
        if c_new:
            # Check against all assigned goals of OTHER agents
            for other_agent, route in all_routes.items():
                if other_agent == agent: continue
                for other_gid in route:
                    c_other = goal_coords.get(other_gid)
                    if c_other:
                        d_sq = (c_new[0] - c_other[0])**2 + (c_new[1] - c_other[1])**2
                        if d_sq < cfg.conflict_radius**2:
                            conflict_cost += cfg.conflict_penalty

    # NEW: Calculate NON-LINEAR Marginal Cost
    # We penalize extending a long schedule much more than starting a new one
    
    # 1. Squared Time Penalty (Forces Load Balancing)
    # The cost is the difference in the SQUARES of the time
    time_cost_delta = (time ** 2) - (current_agent_time ** 2)
    
    # 2. Distance is still linear (Fuel/Energy is linear)
    dist_cost_delta = dist - current_agent_dist
    
    # 3. Combine
    total_cost = (cfg.makespan_weight * time_cost_delta) + \
                 (cfg.distance_weight * dist_cost_delta) + \
                 conflict_cost
    
    return total_cost, None


def _calculate_makespan(routes, agents, agent_to_goal, goal_to_cp, cp_to_goal, link_cache, safe_speed, fixed_cps=None, get_best_link_fn=None) -> float:
    max_time = 0.0
    for agent, route in routes.items():
        if not route: continue
        t = _agent_time(agent, route, agent_to_goal, goal_to_cp, cp_to_goal, link_cache, safe_speed, fixed_cps, get_best_link_fn=get_best_link_fn)
        if t > max_time: max_time = t
    return max_time

def _calculate_total_distance(routes, agents, agent_to_goal, goal_to_cp, cp_to_goal, link_cache, fixed_cps=None, get_best_link_fn=None) -> float:
    total = 0.0
    for agent, route in routes.items():
        if not route: continue
        total += _agent_dist(agent, route, agent_to_goal, goal_to_cp, cp_to_goal, link_cache, fixed_cps, get_best_link_fn=get_best_link_fn)
    return total

def _agent_time(agent, route, a2g, g2c, c2g, cache, speed, fixed_cps=None, get_best_link_fn=None):
    if not route: return 0.0
    t = 0.0
    # Start -> G1
    t += a2g[agent].get(route[0], 0) / speed
    # G1 -> ... -> Gn
    for i in range(len(route)-1):
        curr, nex = route[i], route[i+1]
        if fixed_cps:
            cpid = fixed_cps[curr]
            d = g2c[curr][cpid] + c2g[cpid][nex]
        else:
            if cache is not None and (curr, nex) in cache:
                _, d = cache[(curr, nex)]
            elif get_best_link_fn:
                 _, d = get_best_link_fn(curr, nex)
            else:
                 d = 0 # Fallback should imply error/warning, but keeping 0 for now if no fn provided
        t += d / speed
    # Gn -> End
    last = route[-1]
    if fixed_cps:
        cpid = fixed_cps[last]
        d = g2c[last][cpid]
    else:
        # We need a proper helper or cache for End too?
        # Re-calc minimal end
        d = math.inf

        # Quick hack: calculate minimal end dist
        best = math.inf
        for cpid, dist in g2c[last].items():
            if dist < best: best = dist
        d = best
    t += d / speed
    return t

def _agent_dist(agent, route, a2g, g2c, c2g, cache, fixed_cps=None, get_best_link_fn=None):
    if not route: return 0.0
    d_total = 0.0
    d_total += a2g[agent].get(route[0], 0)
    for i in range(len(route)-1):
        curr, nex = route[i], route[i+1]
        if fixed_cps:
            cpid = fixed_cps[curr]
            d = g2c[curr][cpid] + c2g[cpid][nex]
        else:
            if cache is not None and (curr, nex) in cache:
                _, d = cache[(curr, nex)]
            elif get_best_link_fn:
                _, d = get_best_link_fn(curr, nex)
            else:
                d = 0
        d_total += d
    
    last = route[-1]
    # End dist
    best = math.inf
    if fixed_cps:
         best = g2c[last][fixed_cps[last]]
    else:
        for val in g2c[last].values():
            if val < best: best = val
    d_total += best
    return d_total

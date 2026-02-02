import math
from dataclasses import dataclass
from typing import Optional, Sequence, Dict, List, Tuple

import numpy as np
from dg_commons import PlayerName
from dg_commons.sim import InitSimGlobalObservations, InitSimObservations, SimObservations
from dg_commons.sim.agents import Agent, GlobalPlanner
from dg_commons.sim.goals import PlanningGoal
from dg_commons.sim.models.diff_drive import DiffDriveCommands
from dg_commons.sim.models.diff_drive_structures import DiffDriveGeometry, DiffDriveParameters
from dg_commons.sim.models.obstacles import StaticObstacle

from .structures import Point, GlobalPlanMessage, Mission
from .planner.planner import FleetPlanner
from .planner.graph import EnvironmentMap
from .config import config
from .simple_controller import SimpleController




@dataclass(frozen=True)
class Pdm4arAgentParams:
    param1: float = 10



class Pdm4arAgent(Agent):
    """This is the PDM4AR agent.
    Do *NOT* modify the naming of the existing methods and the input/output types.
    Feel free to add additional methods, objects and functions that help you to solve the task"""

    name: PlayerName
    goal: PlanningGoal
    static_obstacles: Sequence[StaticObstacle]
    sg: DiffDriveGeometry
    sp: DiffDriveParameters

    controller: Optional[SimpleController] = None
    env_map: Optional[EnvironmentMap] = None

    def __init__(self):
        self.missions: List[Mission] = []
        self.current_mission_idx = 0
        self.controller = None
        self.env_map = None

    def on_episode_init(self, init_obs: InitSimObservations):
        self.name = init_obs.my_name
        self.sg = init_obs.model_geometry
        self.sp = init_obs.model_params
        
        # Determine max_omega for planner
        # max_v = max_w * wheel_radius
        # max_v = max_w * wheel_radius
        # 1. Read limits directly
        max_wheel_omega = self.sp.omega_limits[1]
        
        # 2. Calculate max_v and max_omega (Robot Body)
        # v = w_wheel * r
        # Kinematic model - instant velocity response, no need for safety margin
        SAFETY_FACTOR = 1.0
        self.max_v = (max_wheel_omega * self.sg.wheelradius) * SAFETY_FACTOR
        
        # w_robot = w_wheel * r / (L/2)
        L = self.sg.wheelbase
        R = self.sg.wheelradius
        base_max_omega = max_wheel_omega * R / (L / 2.0) if L > 0 else 0.0
        self.max_omega = base_max_omega * SAFETY_FACTOR

        print(f"--- AGENT PARAMETERS ---")
        print(f"Omega Limits (Wheel): {self.sp.omega_limits}")
        print(f"Geometry: R={R}, L={L}")
        print(f"Calculated: max_v={self.max_v:.4f}, max_omega (body)={self.max_omega:.4f}")
        print(f"------------------------")

        self.static_obstacles = init_obs.dg_scenario.static_obstacles
        

        self.controller = SimpleController(
            self.sg.wheelradius, 
            self.sg.wheelbase, 
            self.sp,
            max_v=self.max_v,
            max_omega=self.max_omega
        )

        # Environment Map (only for collision avoidance if needed locally)
        self.env_map = EnvironmentMap(static_obstacles=self.static_obstacles, goals=[], collection_points=[])

    def on_receive_global_plan(self, serialized_msg: str):
        msg = GlobalPlanMessage.model_validate_json(serialized_msg)
        if self.name in msg.agent_plans:
            self.missions = msg.agent_plans[self.name].missions
            if config.verbose_agent_logging:
                print(f"[AGENT] Received Plan with {len(self.missions)} missions.")
                for i, m in enumerate(self.missions):
                    print(f"  Mission {i}: {len(m.commands)} cmds")
                    for j, c in enumerate(m.commands):
                        print(f"    Cmd {j}: {c.type.name} Val={c.value:.2f} Dur={c.duration:.2f} Range=[{c.start_time:.2f}, {c.end_time:.2f}]")
        
        # Reset state
        self.current_mission_idx = 0
        self.mode = "IDLE"
        self.completed_commands_indices = set()

    def get_commands(self, sim_obs: SimObservations) -> DiffDriveCommands:
        my_obs = sim_obs.players[self.name]
        
        # 1. Check Mission Status
        if self.current_mission_idx >= len(self.missions):
            return DiffDriveCommands(0.0, 0.0)
            
        current_mission = self.missions[self.current_mission_idx]
        
        # Check if we are done with this mission (all commands executed)
        # Note: mission.commands includes goal pick up wait and drop off wait
        if current_mission.commands:
            last_cmd = current_mission.commands[-1]
            if sim_obs.time >= last_cmd.end_time - 1e-4: # Finished
                self.current_mission_idx += 1
                if self.current_mission_idx >= len(self.missions):
                     return DiffDriveCommands(0.0, 0.0)
                current_mission = self.missions[self.current_mission_idx]
                
                # Critical Fix: Reset completion tracking for the new mission
                self.completed_commands_indices.clear()

        # 2. Get commands from SimpleController (returns wheel velocities directly now)
        wl, wr = self.controller.get_commands(
            current_time=sim_obs.time,
            current_pose=(my_obs.state.x, my_obs.state.y, my_obs.state.psi),
            commands_queue=current_mission.commands
        )

        # DEBUG: Drift Check (Event-Based)
        # Check if any commands have finished since last check
        if not hasattr(self, "completed_commands_indices"):
             self.completed_commands_indices = set()
             
        for idx, cmd in enumerate(current_mission.commands):
            if idx in self.completed_commands_indices:
                continue
                
            # If command finished (current_time >= end_time - epsilon)
            if sim_obs.time >= cmd.end_time - 1e-4:
                self.completed_commands_indices.add(idx)
                
                # Check drift against end_pose
                if cmd.end_pose:
                    expected_x, expected_y, _ = cmd.end_pose
                    dx = my_obs.state.x - expected_x
                    dy = my_obs.state.y - expected_y
                    dist_err = math.sqrt(dx*dx + dy*dy)
                    
                    if config.verbose_agent_logging:
                        print(f"[AGENT] Cmd {idx} ({cmd.type.name}) Finished at t={sim_obs.time:.2f}. "
                              f"Val={cmd.value:.2f} Dur={cmd.duration:.2f} "
                              f"Range=[{cmd.start_time:.2f}, {cmd.end_time:.2f}] "
                              f"Pos=({my_obs.state.x:.2f}, {my_obs.state.y:.2f}) Psi={my_obs.state.psi:.2f} "
                              f"Exp=({expected_x:.2f}, {expected_y:.2f}) Err={dist_err:.2f}")

        return DiffDriveCommands(omega_l=wl, omega_r=wr)


class Pdm4arGlobalPlanner(GlobalPlanner):
    """
    This is the Global Planner for PDM4AR
    Do *NOT* modify the naming of the existing methods and the input/output types.
    Feel free to add additional methods, objects and functions that help you to solve the task
    """

    def __init__(self):
        self.planner = FleetPlanner()

    def send_plan(self, init_sim_obs: InitSimGlobalObservations) -> str:
        static_obstacles = init_sim_obs.dg_scenario.static_obstacles
        goals = (
            init_sim_obs.shared_goals.values()
            if isinstance(init_sim_obs.shared_goals, dict)
            else init_sim_obs.shared_goals
        )
        cps = (
            init_sim_obs.collection_points.values()
            if isinstance(init_sim_obs.collection_points, dict)
            else init_sim_obs.collection_points
        )

        first_player_obs = next(iter(init_sim_obs.players_obs.values()))

        # Calculate max_v from model params
        geom = first_player_obs.model_geometry
        params = first_player_obs.model_params
        
        max_wheel_omega = params.omega_limits[1]
        
        # v = w_wheel * r
        max_v = max_wheel_omega * geom.wheelradius
        
        # w_robot = w_wheel * r / (L/2)
        L = geom.wheelbase
        R = geom.wheelradius
        base_max_omega = max_wheel_omega * R / (L / 2.0) if L > 0 else 1.0
        
        # Kinematic model - instant velocity response, no need for safety margin
        SAFETY_FACTOR = 1.0
        max_v = (max_wheel_omega * geom.wheelradius) * SAFETY_FACTOR
        max_omega = base_max_omega * SAFETY_FACTOR

        print(f"--- PLANNER PARAMETERS ---")
        print(f"Omega Limits (Wheel): {params.omega_limits}")
        print(f"Geometry: R={R}, L={L}")
        print(f"Calculated (Scaled): max_v={max_v:.4f}, max_omega (body)={max_omega:.4f}")
        print(f"--------------------------")

        # Count the number of agents in this configuration
        num_agents = len(init_sim_obs.players_obs)
        print(f"Number of agents: {num_agents}")

        # Prepare start positions
        agent_starts = {}
        for name, state in init_sim_obs.initial_states.items():
            # Extract psi (heading) from state
            agent_starts[name] = Point(x=state.x, y=state.y, theta=state.psi)
            print(f"[DEBUG] Agent {name} Start: ({state.x:.2f}, {state.y:.2f}, theta={state.psi:.2f})")

        # Call the FleetPlanner
        plans = self.planner.plan(
            agent_starts=agent_starts,
            goals=goals,
            collection_points=cps,
            static_obstacles=static_obstacles,
            max_v=max_v,
            max_omega=max_omega,
            num_agents=num_agents,
        )

        # Merging individual player plans to global message
        global_plan_message = GlobalPlanMessage(agent_plans=plans)

        self.controllers = {}
        if init_sim_obs.players_obs:
            for name, player_obs in init_sim_obs.players_obs.items():
                self.controllers[name] = SimpleController(
                    player_obs.model_geometry.wheelradius, 
                    player_obs.model_geometry.wheelbase, 
                    player_obs.model_params
                )

        return global_plan_message.model_dump_json(round_trip=True)

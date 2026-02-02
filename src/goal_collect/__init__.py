"""
Multi-Agent Goal Collection Planner

A sophisticated multi-robot coordination system for efficient goal collection tasks.
"""

__version__ = "1.0.0"

from .agent import Pdm4arAgent, Pdm4arAgentParams, Pdm4arGlobalPlanner
from .structures import (
    Point,
    Mission,
    GlobalPlanMessage,
    AgentPlan,
    AgentState,
    Command,
    CommandType,
    ReservationTable,
    GlobalSolution,
)
from .config import config

__all__ = [
    "Pdm4arAgent",
    "Pdm4arAgentParams",
    "Pdm4arGlobalPlanner",
    "Point",
    "Mission",
    "GlobalPlanMessage",
    "AgentPlan",
    "AgentState",
    "Command",
    "CommandType",
    "ReservationTable",
    "GlobalSolution",
    "config",
]

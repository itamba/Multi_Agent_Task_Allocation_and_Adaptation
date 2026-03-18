"""observation_utils.py

Utilities for updating existing MATCH-AOU `Agent` objects from a fresh Blade
observation (Scenario state).

Why this exists
- During simulation (or RL), the world state changes.
- We want to keep the MATCH-AOU objects (Agents) in sync with the simulator
  without rebuilding everything from scratch.

This module is intentionally Blade-facing: it reads fields from Blade unit
objects (aircraft/ships) and updates the corresponding MATCH-AOU Agent.
"""

from __future__ import annotations

from typing import Any, Dict, List

from ...models import Agent, Location


def update_agents_from_observation(agents_by_side: Dict[str, List[Agent]], scenario: Any) -> None:
    """Update existing `Agent` objects (location, budget, weapon quantities).

    Args:
        agents_by_side: Mapping side_color -> list of Agents previously created.
        scenario: Blade `Scenario` observation.

    Notes
    - Updates are performed in-place.
    - Capability update logic assumes the demo convention:
        Capability(name="attack", properties={weapon_class_name: quantity, ...}).
    """

    agents_by_id: Dict[str, Agent] = {agent.id: agent for agents in agents_by_side.values() for agent in agents}

    # --- Aircraft ---
    for ac in getattr(scenario, "aircraft", []) or []:
        agent = agents_by_id.get(getattr(ac, "id", None))
        if not agent:
            continue

        agent.location = Location(ac.latitude, ac.longitude, getattr(ac, "altitude", 0))
        agent.budget = getattr(ac, "current_fuel", 0) or 0

        # Update weapon quantities inside attack capability properties
        for cap in agent.capabilities:
            if cap.name != "attack":
                continue
            for w in getattr(ac, "weapons", []) or []:
                weapon_name = str(getattr(w, "class_name", "weapon"))
                if weapon_name in (cap.properties or {}):
                    cap.properties[weapon_name] = getattr(w, "current_quantity", 0)

    # --- Ships ---
    for ship in getattr(scenario, "ships", []) or []:
        agent = agents_by_id.get(getattr(ship, "id", None))
        if not agent:
            continue

        agent.location = Location(ship.latitude, ship.longitude, getattr(ship, "altitude", 0))
        agent.budget = getattr(ship, "current_fuel", 0) or 0

        for cap in agent.capabilities:
            if cap.name != "attack":
                continue
            for w in getattr(ship, "weapons", []) or []:
                weapon_name = str(getattr(w, "class_name", "weapon"))
                if weapon_name in (cap.properties or {}):
                    cap.properties[weapon_name] = getattr(w, "current_quantity", 0)

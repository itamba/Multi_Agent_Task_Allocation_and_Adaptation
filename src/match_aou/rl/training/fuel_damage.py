"""
Fuel Damage Events - Runtime Surprise for RL Training
======================================================

Simulates mid-mission fuel damage (hit, malfunction, leak) to force RL agents
to learn policies that deviate from the oracle's plan.

How it works:
- MATCH-AOU solves with full fuel (doesn't know about future damage)
- During the episode, at a random tick, an agent's *observed* fuel drops
- The RL agent sees low fuel in its observation → should learn FORCE_RTB
- The oracle had no knowledge of this → RL must learn independent decision-making

Implementation approach:
- BLADE continues running with real fuel (simulation integrity preserved)
- Damage is applied at the OBSERVATION level: when building the RL observation,
  we multiply the agent's fuel reading by a damage factor
- This means the RL "perceives" fuel damage even though BLADE's physics are unaffected
- The gap between oracle plan (full fuel) and RL reaction (damaged fuel) is the
  learning signal we want

Usage in train_full.py:
    manager = FuelDamageManager(FuelDamageConfig())
    events = manager.plan_episode(agent_ids, max_ticks)

    # In observation building:
    fuel = get_aircraft_fuel(observation, agent_id)
    fuel = manager.apply_damage(agent_id, tick, fuel)
    # Use adjusted fuel for observation vector
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class FuelDamageConfig:
    """
    Configuration for fuel damage events.

    Attributes:
        enabled: Master toggle for fuel damage events
        probability: Probability of damage occurring in an episode (0.0 - 1.0)
        max_damaged_agents: Maximum number of agents damaged per episode
        damage_factor_range: (min, max) multiplier applied to fuel.
            0.3 means fuel drops to 30% of current value.
            Range is sampled uniformly.
        tick_window: (earliest_fraction, latest_fraction) of episode ticks
            where damage can occur. (0.2, 0.7) means damage happens
            between 20% and 70% of max_ticks.
    """
    enabled: bool = True
    probability: float = 0.5
    max_damaged_agents: int = 1
    damage_factor_range: tuple = (0.2, 0.4)
    tick_window: tuple = (0.2, 0.7)


@dataclass
class FuelDamageEvent:
    """A single planned fuel damage event."""
    agent_id: str
    trigger_tick: int
    damage_factor: float  # fuel multiplied by this (e.g., 0.3 = 30% remaining)
    activated: bool = False


class FuelDamageManager:
    """
    Manages fuel damage events during a training episode.

    Lifecycle:
        1. plan_episode() — called at episode start, rolls dice for damage
        2. check_and_activate() — called each tick, activates events
        3. apply_damage() — called when building observations, adjusts fuel

    Example:
        >>> config = FuelDamageConfig(probability=0.5, damage_factor_range=(0.2, 0.4))
        >>> manager = FuelDamageManager(config)
        >>> events = manager.plan_episode(['agent-1', 'agent-2'], max_ticks=14400)
        >>> # events might be: [FuelDamageEvent(agent_id='agent-2', trigger_tick=4320, ...)]
        >>>
        >>> # During simulation loop:
        >>> newly_activated = manager.check_and_activate(current_tick=4320)
        >>> # newly_activated = ['agent-2']
        >>>
        >>> # When building observation:
        >>> real_fuel = 8000.0
        >>> adjusted_fuel = manager.apply_damage('agent-2', current_tick=5000, fuel=real_fuel)
        >>> # adjusted_fuel = 2400.0  (8000 * 0.3)
    """

    def __init__(self, config: Optional[FuelDamageConfig] = None):
        self.config = config or FuelDamageConfig()
        self.events: List[FuelDamageEvent] = []
        self._active_agents: Dict[str, float] = {}  # agent_id → damage_factor

    def plan_episode(
        self,
        agent_ids: List[str],
        max_ticks: int,
        seed: Optional[int] = None,
    ) -> List[FuelDamageEvent]:
        """
        Plan fuel damage events for an episode.

        Called once at the start of each episode. Rolls dice to decide
        if/when/who/how-much damage occurs.

        Args:
            agent_ids: List of agent IDs in this episode
            max_ticks: Maximum ticks in the episode
            seed: Optional seed for reproducibility

        Returns:
            List of planned FuelDamageEvent objects (may be empty)
        """
        self.events = []
        self._active_agents = {}

        if not self.config.enabled:
            return []

        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random.Random()

        # Roll dice: does damage happen this episode?
        if rng.random() > self.config.probability:
            logger.debug("Fuel damage: no damage this episode (dice roll)")
            return []

        # How many agents get damaged?
        n_damaged = rng.randint(1, min(self.config.max_damaged_agents, len(agent_ids)))

        # Which agents?
        damaged_agents = rng.sample(agent_ids, n_damaged)

        # When and how much?
        earliest = int(max_ticks * self.config.tick_window[0])
        latest = int(max_ticks * self.config.tick_window[1])

        for agent_id in damaged_agents:
            trigger_tick = rng.randint(earliest, max(earliest, latest))
            damage_factor = rng.uniform(*self.config.damage_factor_range)

            event = FuelDamageEvent(
                agent_id=agent_id,
                trigger_tick=trigger_tick,
                damage_factor=round(damage_factor, 3),
            )
            self.events.append(event)

            logger.info(
                f"  Fuel damage planned: agent={agent_id[:8]}.. "
                f"tick={trigger_tick} factor={damage_factor:.2f}"
            )

        return self.events

    def check_and_activate(self, current_tick: int) -> List[str]:
        """
        Check if any damage events should activate at this tick.

        Args:
            current_tick: Current simulation tick

        Returns:
            List of agent IDs that were just damaged (newly activated)
        """
        newly_activated = []

        for event in self.events:
            if event.activated:
                continue
            if current_tick >= event.trigger_tick:
                event.activated = True
                self._active_agents[event.agent_id] = event.damage_factor
                newly_activated.append(event.agent_id)

                logger.info(
                    f"  *** FUEL DAMAGE at tick {current_tick}: "
                    f"agent={event.agent_id[:8]}.. "
                    f"fuel reduced to {event.damage_factor:.0%} ***"
                )

        return newly_activated

    def apply_damage(self, agent_id: str, fuel: float) -> float:
        """
        Apply fuel damage to an agent's fuel reading.

        Call this when building observations. If the agent has active
        damage, the fuel is multiplied by the damage factor.

        Args:
            agent_id: Agent ID
            fuel: Real fuel value from BLADE

        Returns:
            Adjusted fuel (reduced if agent is damaged, unchanged otherwise)
        """
        if agent_id not in self._active_agents:
            return fuel

        return fuel * self._active_agents[agent_id]

    def is_damaged(self, agent_id: str) -> bool:
        """Check if an agent currently has fuel damage."""
        return agent_id in self._active_agents

    def get_damaged_agents(self) -> Set[str]:
        """Get set of currently damaged agent IDs."""
        return set(self._active_agents.keys())

    def get_event_summary(self) -> str:
        """Get human-readable summary of damage events."""
        if not self.events:
            return "No fuel damage events"

        lines = []
        for ev in self.events:
            status = "ACTIVE" if ev.activated else f"pending (tick {ev.trigger_tick})"
            lines.append(
                f"  {ev.agent_id[:8]}.. → {ev.damage_factor:.0%} fuel [{status}]"
            )
        return "\n".join(lines)

    def reset(self):
        """Reset for next episode."""
        self.events = []
        self._active_agents = {}

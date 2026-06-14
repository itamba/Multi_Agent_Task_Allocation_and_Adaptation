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
- One-time real-fuel hit at the activation tick: aircraft.current_fuel *= damage_factor
  is applied directly to BLADE state. From then on, BLADE drains fuel normally and
  the observation reads the (already reduced) real fuel naturally.
- Asymmetry preserved: the oracle plan was solved at episode start with full fuel,
  so the damaged agent must learn to deviate when its (real) fuel can no longer
  cover the oracle's plan. If it ignores the damage, BLADE physics will drain
  the tank to zero and remove_aircraft() fires → crash.

Behavior change vs. earlier version: damage was originally observation-only
(BLADE physics ran on full fuel). Under sparse utility-based reward that created
a perverse incentive — ignoring damage gave higher utility. Real-fuel mutation
makes the RTB signal direct.

Usage in train_full.py:
    manager = FuelDamageManager(FuelDamageConfig())
    events = manager.plan_episode(agent_ids, max_ticks)

    # Each tick:
    newly_damaged = manager.check_and_activate(tick)
    for aid in newly_damaged:
        manager.apply_real_damage(scenario, aid)  # mutates BLADE state once
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
        3. apply_real_damage() — called once per newly-activated agent,
           mutates aircraft.current_fuel in BLADE state

    Example:
        >>> config = FuelDamageConfig(probability=0.5, damage_factor_range=(0.2, 0.4))
        >>> manager = FuelDamageManager(config)
        >>> events = manager.plan_episode(['agent-1', 'agent-2'], max_ticks=14400)
        >>>
        >>> # During simulation loop:
        >>> newly_activated = manager.check_and_activate(current_tick=4320)
        >>> for aid in newly_activated:
        ...     manager.apply_real_damage(scenario, aid)
        >>> # The agent's BLADE current_fuel is now factor × old_value (one-time hit).
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

            logger.debug(
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

                logger.debug(
                    f"  *** FUEL DAMAGE at tick {current_tick}: "
                    f"agent={event.agent_id[:8]}.. "
                    f"fuel reduced to {event.damage_factor:.0%} ***"
                )

        return newly_activated

    def apply_real_damage(self, scenario: Any, agent_id: str) -> Optional[float]:
        """
        Apply a one-time real-fuel reduction to BLADE state.

        Called once per agent right after check_and_activate() reports the
        activation. Mutates aircraft.current_fuel *= damage_factor on the
        BLADE Aircraft object so the agent physically has less fuel from
        this tick onward. From then on, BLADE drains fuel normally; the
        observation reads the reduced real fuel directly (no obs-side
        multiplier — that would compound).

        Args:
            scenario: BLADE Scenario observation (has .aircraft list)
            agent_id: Agent ID whose damage just activated

        Returns:
            New current_fuel after mutation, or None if the aircraft was
            not found (e.g., already landed/crashed before activation).
        """
        if agent_id not in self._active_agents:
            return None
        factor = self._active_agents[agent_id]

        for ac in getattr(scenario, "aircraft", []) or []:
            if str(getattr(ac, "id", "")) == str(agent_id):
                old_fuel = float(getattr(ac, "current_fuel", 0.0))
                new_fuel = old_fuel * factor
                ac.current_fuel = new_fuel
                logger.debug(
                    f"  Real fuel hit: agent={agent_id[:8]}.. "
                    f"{old_fuel:.0f} → {new_fuel:.0f} (factor {factor:.2f})"
                )
                return new_fuel

        return None

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

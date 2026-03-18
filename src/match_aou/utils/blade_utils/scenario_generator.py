"""scenario_generator.py

Generates varied BLADE scenario JSONs from a base template for RL training.

Capabilities:
- Randomize facility (target) positions within reachable range
- Add/remove facilities (targets)
- Add/remove aircraft (agents)
- Full fuel-based reachability validation
- Traceability: each generated scenario is tagged with its episode number

Usage:
    generator = ScenarioGenerator(base_scenario_path="strike_training_2v3.json")
    scenario_json = generator.generate(episode=5, config=VariationConfig(...))
    # Saves to: generated_scenarios/episode_005_scenario.json
"""

from __future__ import annotations

import copy
import json
import math
import random
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class VariationConfig:
    """Controls what varies between episodes.

    Set a field to None to keep the base-template value unchanged.
    """

    # --- Facility (target) count ---
    # None  = keep base count
    # int   = exact number of facilities to have
    # (min, max) = sample uniformly from this range (inclusive)
    num_facilities: Optional[int | Tuple[int, int]] = None

    # --- Aircraft (agent) count ---
    # Same semantics as num_facilities
    num_aircraft: Optional[int | Tuple[int, int]] = None

    # --- Position randomization ---
    randomize_facility_positions: bool = True

    # --- Max target distance (km) ---
    # Caps how far facilities can be placed from the blue base.
    # The effective sampling radius is min(max_target_distance_km, fuel-based range).
    # None = no cap (fuel-based range only)
    max_target_distance_km: Optional[float] = 500.0

    # --- Min target distance (km) ---
    # Facilities won't be placed closer than this to the blue base.
    min_target_distance_km: float = 50.0

    # --- Blue base randomization ---
    # If True, the blue base is moved to a random position before
    # placing targets. The shift radius is controlled by base_shift_radius_km.
    randomize_base_position: bool = False
    base_shift_radius_km: float = 200.0

    # --- Fuel safety margin (0.0 - 1.0) ---
    # 0.3 means we only use 70% of the theoretical max one-way range
    fuel_safety_margin: float = 0.3

    # --- Random seed (None = random each time) ---
    seed: Optional[int] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two lat/lon points."""
    R = 6371.0  # Earth radius in km
    rlat1, rlat2 = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(rlat1) * math.cos(rlat2) * math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _new_uuid() -> str:
    return str(uuid.uuid4())


def _resolve_range(
    value: Optional[int | Tuple[int, int]], rng: random.Random
) -> Optional[int]:
    """Resolve a config value that can be None, int, or (min, max) tuple."""
    if value is None:
        return None
    if isinstance(value, int):
        return value
    lo, hi = value
    return rng.randint(lo, hi)


# ---------------------------------------------------------------------------
# Reachability calculator
# ---------------------------------------------------------------------------

class ReachabilityCalculator:
    """Computes how far an aircraft can fly (one-way, round-trip) based on fuel."""

    def __init__(self, safety_margin: float = 0.3):
        self.safety_margin = safety_margin

    def max_one_way_km(self, aircraft: Dict[str, Any]) -> float:
        """Max one-way distance (km) an aircraft can travel and still return.

        Formula:
            total_range_km = (currentFuel / fuelRate) * speed_knots * 1.852
            one_way = total_range_km / 2
            usable  = one_way * (1 - safety_margin)
        """
        fuel = float(aircraft.get("currentFuel", 0))
        fuel_rate = float(aircraft.get("fuelRate", 1))  # per hour
        speed_knots = float(aircraft.get("speed", 0))

        if fuel_rate <= 0 or speed_knots <= 0:
            return 0.0

        flight_hours = fuel / fuel_rate
        total_range_km = flight_hours * speed_knots * 1.852
        one_way = total_range_km / 2.0
        return one_way * (1.0 - self.safety_margin)

    def is_reachable(
        self,
        aircraft: Dict[str, Any],
        base_lat: float, base_lon: float,
        target_lat: float, target_lon: float,
    ) -> bool:
        """Can this aircraft fly from base to target and back?"""
        dist = _haversine_km(base_lat, base_lon, target_lat, target_lon)
        return dist <= self.max_one_way_km(aircraft)

    def is_reachable_by_any(
        self,
        aircraft_list: List[Dict[str, Any]],
        base_lat: float, base_lon: float,
        target_lat: float, target_lon: float,
    ) -> bool:
        """Can at least one aircraft reach this target?"""
        return any(
            self.is_reachable(ac, base_lat, base_lon, target_lat, target_lon)
            for ac in aircraft_list
        )


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

class ScenarioGenerator:
    """Generates BLADE scenario JSON variations from a base template."""

    def __init__(
        self,
        base_scenario_path: str,
        output_dir: str = "generated_scenarios",
    ):
        self.base_path = Path(base_scenario_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        with open(self.base_path, "r", encoding="utf-8") as f:
            self._base_data = json.load(f)

        # Cache side IDs for quick lookups
        self._blue_side_id, self._red_side_id = self._identify_sides()

    # ---- Side identification ----

    def _identify_sides(self) -> Tuple[str, str]:
        """Find BLUE and RED side IDs from the scenario."""
        scenario = self._base_data["currentScenario"]
        blue_id = red_id = ""
        for side in scenario["sides"]:
            if side["color"].lower() == "blue":
                blue_id = side["id"]
            elif side["color"].lower() == "red":
                red_id = side["id"]
        return blue_id, red_id

    # ---- Blue base location ----

    def _get_blue_base(self, scenario: Dict) -> Tuple[float, float, Dict]:
        """Returns (lat, lon, airbase_dict) for the blue airbase."""
        for ab in scenario.get("airbases", []):
            if ab.get("sideId") == self._blue_side_id:
                return ab["latitude"], ab["longitude"], ab
        raise ValueError("No BLUE airbase found in scenario")

    # ---- Aircraft from blue base ----

    def _get_blue_aircraft(self, scenario: Dict) -> List[Dict]:
        """Get list of aircraft dicts from the blue airbase."""
        _, _, blue_base = self._get_blue_base(scenario)
        return blue_base.get("aircraft", [])

    # ---- Red facilities ----

    def _get_red_facilities(self, scenario: Dict) -> List[Dict]:
        """Get all RED-side facilities."""
        return [
            f for f in scenario.get("facilities", [])
            if f.get("sideId") == self._red_side_id
        ]

    # ==================================================================
    # CORE: Generate a varied scenario
    # ==================================================================

    def generate(
        self, episode: int, config: Optional[VariationConfig] = None
    ) -> Path:
        """Generate a scenario variation and save it to disk.

        Args:
            episode: Episode number (used in filename and metadata).
            config: Variation settings. Uses defaults if None.

        Returns:
            Path to the saved JSON file.
        """
        config = config or VariationConfig()
        rng = random.Random(config.seed if config.seed is not None else None)
        reachability = ReachabilityCalculator(
            safety_margin=config.fuel_safety_margin
        )

        # Deep copy the base scenario
        data = copy.deepcopy(self._base_data)
        scenario = data["currentScenario"]

        # Step 1: Adjust aircraft count (before reachability checks)
        desired_aircraft = _resolve_range(config.num_aircraft, rng)
        if desired_aircraft is not None:
            self._adjust_aircraft_count(scenario, desired_aircraft, rng)

        # Step 2: Adjust facility count
        desired_facilities = _resolve_range(config.num_facilities, rng)
        if desired_facilities is not None:
            self._adjust_facility_count(scenario, desired_facilities, rng)

        # Step 3: Randomize blue base position (before target placement)
        if config.randomize_base_position:
            self._randomize_base_position(
                scenario, config.base_shift_radius_km, rng
            )

        # Step 4: Randomize facility positions (with reachability validation)
        if config.randomize_facility_positions:
            self._randomize_facility_positions(
                scenario, reachability, config, rng
            )

        # Step 5: Tag with episode metadata
        scenario["name"] = f"episode_{episode:04d}"
        scenario["id"] = _new_uuid()

        # Step 6: Save
        filename = f"episode_{episode:04d}_scenario.json"
        out_path = self.output_dir / filename
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        return out_path

    # ==================================================================
    # Position randomization
    # ==================================================================

    def _randomize_facility_positions(
        self,
        scenario: Dict,
        reachability: ReachabilityCalculator,
        config: VariationConfig,
        rng: random.Random,
        max_attempts: int = 100,
    ) -> None:
        """Randomize RED facility positions within aircraft reachable range.

        The sampling radius is capped by both:
        - The weakest aircraft's fuel-based range
        - config.max_target_distance_km (if set)

        Minimum distance is config.min_target_distance_km.
        """
        base_lat, base_lon, _ = self._get_blue_base(scenario)
        aircraft_list = self._get_blue_aircraft(scenario)

        if not aircraft_list:
            return

        # Fuel-based max radius (weakest aircraft)
        fuel_range_km = min(
            reachability.max_one_way_km(ac) for ac in aircraft_list
        )

        # Apply the config cap
        max_radius = fuel_range_km
        if config.max_target_distance_km is not None:
            max_radius = min(max_radius, config.max_target_distance_km)

        min_radius = config.min_target_distance_km

        # Safety: if min >= max, force a small valid ring
        if min_radius >= max_radius:
            min_radius = max_radius * 0.1

        red_facilities = self._get_red_facilities(scenario)

        for facility in red_facilities:
            placed = False
            for _ in range(max_attempts):
                new_lat, new_lon = self._random_point_in_ring(
                    base_lat, base_lon,
                    min_radius, max_radius,
                    rng,
                )
                if reachability.is_reachable_by_any(
                    aircraft_list, base_lat, base_lon, new_lat, new_lon
                ):
                    facility["latitude"] = new_lat
                    facility["longitude"] = new_lon
                    placed = True
                    break

            if not placed:
                # Fallback: keep original position (from the base template)
                pass

    @staticmethod
    def _random_point_in_ring(
        center_lat: float, center_lon: float,
        min_km: float, max_km: float,
        rng: random.Random,
    ) -> Tuple[float, float]:
        """Sample a random lat/lon within a ring [min_km, max_km] from center.

        Uses uniform random bearing + uniform distance in the ring.
        Approximation: treats lat/lon offsets linearly (accurate enough
        for distances < ~500 km at these latitudes).
        """
        bearing_rad = rng.uniform(0, 2 * math.pi)
        distance_km = rng.uniform(min_km, max_km)

        # 1 degree lat ~ 111 km, 1 degree lon ~ 111 * cos(lat) km
        dlat = (distance_km * math.cos(bearing_rad)) / 111.0
        dlon = (distance_km * math.sin(bearing_rad)) / (
            111.0 * math.cos(math.radians(center_lat))
        )

        return center_lat + dlat, center_lon + dlon

    # ==================================================================
    # Blue base randomization
    # ==================================================================

    def _randomize_base_position(
        self,
        scenario: Dict,
        shift_radius_km: float,
        rng: random.Random,
    ) -> None:
        """Move the blue airbase to a random nearby position.

        Also updates all aircraft positions to match the new base location,
        since aircraft inherit their starting coordinates from the base.

        Args:
            scenario: The scenario dict to modify in-place.
            shift_radius_km: Max distance (km) to shift from original position.
            rng: Random number generator.
        """
        old_lat, old_lon, blue_base = self._get_blue_base(scenario)

        # Sample a new base position within [0, shift_radius_km] of the original
        new_lat, new_lon = self._random_point_in_ring(
            old_lat, old_lon,
            min_km=0.0,
            max_km=shift_radius_km,
            rng=rng,
        )

        # Update the airbase itself
        blue_base["latitude"] = new_lat
        blue_base["longitude"] = new_lon

        # Update all aircraft in this base to the new position.
        # In the base JSON, aircraft lat/lon represent their starting
        # position. We shift them by the same delta as the base.
        dlat = new_lat - old_lat
        dlon = new_lon - old_lon

        for ac in blue_base.get("aircraft", []):
            ac["latitude"] += dlat
            ac["longitude"] += dlon

    # ==================================================================
    # Facility count adjustment
    # ==================================================================

    def _adjust_facility_count(
        self, scenario: Dict, desired: int, rng: random.Random
    ) -> None:
        """Add or remove RED facilities to reach the desired count."""
        red_facilities = self._get_red_facilities(scenario)
        current = len(red_facilities)

        if desired < 1:
            desired = 1  # Must have at least one target

        if desired == current:
            return

        if desired < current:
            # Remove excess facilities (random selection)
            to_remove = rng.sample(red_facilities, current - desired)
            remove_ids = {f["id"] for f in to_remove}
            scenario["facilities"] = [
                f for f in scenario["facilities"]
                if f["id"] not in remove_ids
            ]
        else:
            # Add facilities by cloning existing ones
            template = rng.choice(red_facilities)
            for i in range(desired - current):
                new_facility = copy.deepcopy(template)
                new_facility["id"] = _new_uuid()
                new_facility["name"] = f"Target Site {chr(65 + current + i)}"

                # Give each weapon a new UUID too
                for weapon in new_facility.get("weapons", []):
                    weapon["id"] = _new_uuid()

                scenario["facilities"].append(new_facility)

    # ==================================================================
    # Aircraft count adjustment
    # ==================================================================

    def _adjust_aircraft_count(
        self, scenario: Dict, desired: int, rng: random.Random
    ) -> None:
        """Add or remove aircraft from the BLUE airbase."""
        _, _, blue_base = self._get_blue_base(scenario)
        aircraft_list = blue_base.get("aircraft", [])
        current = len(aircraft_list)

        if desired < 1:
            desired = 1  # Must have at least one agent

        if desired == current:
            return

        if desired < current:
            # Keep first N, remove the rest
            blue_base["aircraft"] = aircraft_list[:desired]
        else:
            # Clone existing aircraft with new IDs
            template = aircraft_list[0] if aircraft_list else None
            if template is None:
                return

            for i in range(desired - current):
                new_ac = copy.deepcopy(template)
                new_ac["id"] = _new_uuid()

                # Generate a new tail number
                tail_num = rng.randint(100, 999)
                new_ac["name"] = f"F-16 Fighting Falcon #{tail_num}"

                # New UUIDs for all weapons
                for weapon in new_ac.get("weapons", []):
                    weapon["id"] = _new_uuid()

                aircraft_list.append(new_ac)

    # ==================================================================
    # Batch generation
    # ==================================================================

    def generate_batch(
        self,
        num_episodes: int,
        config: Optional[VariationConfig] = None,
        start_episode: int = 0,
    ) -> List[Path]:
        """Generate multiple scenario variations.

        Args:
            num_episodes: How many scenarios to generate.
            config: Shared variation config. If seed is None,
                    each episode gets its own seed (= episode number)
                    for reproducibility.
            start_episode: Starting episode number.

        Returns:
            List of paths to generated files.
        """
        config = config or VariationConfig()
        paths = []

        for i in range(num_episodes):
            ep = start_episode + i
            ep_config = copy.deepcopy(config)

            # Deterministic per-episode seed for reproducibility
            if ep_config.seed is None:
                ep_config.seed = ep

            paths.append(self.generate(episode=ep, config=ep_config))

        return paths


# ---------------------------------------------------------------------------
# Quick test / demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    base_path = (
        sys.argv[1] if len(sys.argv) > 1 else "strike_training_2v3.json"
    )

    gen = ScenarioGenerator(base_scenario_path=base_path)

    configs = [
        # Episode 0: Just randomize positions, capped at 500km (default)
        VariationConfig(randomize_facility_positions=True, seed=42),
        # Episode 1: 2v4, tighter range (300km max)
        VariationConfig(num_facilities=4, max_target_distance_km=300.0, seed=43),
        # Episode 2: 3v2 with base shift
        VariationConfig(
            num_aircraft=3, num_facilities=2,
            randomize_base_position=True, base_shift_radius_km=150.0,
            seed=44,
        ),
        # Episode 3: Random targets 2-5, base shift, tight range
        VariationConfig(
            num_facilities=(2, 5),
            randomize_base_position=True, base_shift_radius_km=100.0,
            max_target_distance_km=400.0,
            seed=45,
        ),
        # Episode 4: Random everything
        VariationConfig(
            num_aircraft=(2, 4), num_facilities=(2, 5),
            randomize_base_position=True, base_shift_radius_km=200.0,
            max_target_distance_km=500.0,
            seed=46,
        ),
    ]

    for ep, cfg in enumerate(configs):
        path = gen.generate(episode=ep, config=cfg)
        print(f"Episode {ep}: {path}")

        with open(path) as f:
            data = json.load(f)
        sc = data["currentScenario"]
        blue_ab = next(
            ab for ab in sc["airbases"] if ab["sideColor"] == "blue"
        )
        n_ac = len(blue_ab.get("aircraft", []))
        n_fac = len(sc["facilities"])
        print(
            f"  -> {n_ac} aircraft, {n_fac} facilities, "
            f"base=({blue_ab['latitude']:.2f}, {blue_ab['longitude']:.2f})"
        )

"""scenario_generator.py

Generates varied BLADE scenario JSONs from a base template for RL training.

Capabilities:
- Aircraft pool: extract templates from scenario JSONs, build diverse fleets
- Facility pool: extract facility templates (SAM types), build diverse targets
- Randomize facility (target) positions within reachable range
- Add/remove/randomize RED airbases as targets
- Add/remove facilities (SAM sites) from diverse pool
- Full fuel-based reachability validation
- Traceability: each generated scenario is tagged with its episode number

Usage:
    generator = ScenarioGenerator(
        base_scenario_path="strike_training_4v5.json",
    )
    scenario_json = generator.generate(episode=5, config=VariationConfig(...))
"""

from __future__ import annotations

import copy
import json
import logging
import math
import random
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class VariationConfig:
    """Controls what varies between episodes.

    Set a field to None to keep the base-template value unchanged.
    """

    # --- SAM inclusion toggle ---
    # When False, all facilities (SAM sites) are removed from the scenario.
    # This ensures targets are only RED airbases (no interception capability),
    # so blue missiles can hit their targets and the RL loop gets reward signal.
    include_sams: bool = True

    # --- Facility (SAM site) count ---
    # None  = keep base count
    # int   = exact number
    # (min, max) = sample uniformly (inclusive)
    # Ignored when include_sams=False (forced to 0).
    num_facilities: Optional[int | Tuple[int, int]] = None

    # --- RED airbase (target) count ---
    # Same semantics. Only counts RED-side airbases (empty enemy bases).
    num_red_airbases: Optional[int | Tuple[int, int]] = None

    # --- Aircraft (agent) count ---
    num_aircraft: Optional[int | Tuple[int, int]] = None

    # --- Aircraft class filter ---
    # When set, only these aircraft types are used. Existing aircraft of
    # other types are removed BEFORE adjusting count. New aircraft are
    # sampled uniformly from this list.
    # Example: ["F-35A Lightning II"] → all agents will be F-35s.
    # None = use any type from the pool (original behavior).
    allowed_aircraft_classes: Optional[List[str]] = None

    # --- Position randomization ---
    randomize_facility_positions: bool = True
    randomize_red_airbase_positions: bool = True

    # --- Max / min target distance (km) ---
    # Applies to both facilities and RED airbases.
    max_target_distance_km: Optional[float] = 500.0
    min_target_distance_km: float = 50.0

    # --- Blue base randomization ---
    randomize_base_position: bool = False
    base_shift_radius_km: float = 200.0

    # --- Fuel safety margin (0.0 - 1.0) ---
    fuel_safety_margin: float = 0.3

    # --- Random seed (None = random each time) ---
    seed: Optional[int] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two lat/lon points."""
    R = 6371.0
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


def _random_point_in_ring(
    center_lat: float, center_lon: float,
    min_km: float, max_km: float,
    rng: random.Random,
) -> Tuple[float, float]:
    """Sample a random lat/lon within a ring [min_km, max_km] from center."""
    bearing_rad = rng.uniform(0, 2 * math.pi)
    distance_km = rng.uniform(min_km, max_km)

    dlat = (distance_km * math.cos(bearing_rad)) / 111.0
    dlon = (distance_km * math.sin(bearing_rad)) / (
        111.0 * math.cos(math.radians(center_lat))
    )
    return center_lat + dlat, center_lon + dlon


# ---------------------------------------------------------------------------
# Reachability calculator
# ---------------------------------------------------------------------------

class ReachabilityCalculator:
    """Computes how far an aircraft can fly (one-way, round-trip) based on fuel."""

    def __init__(self, safety_margin: float = 0.3):
        self.safety_margin = safety_margin

    def max_one_way_km(self, aircraft: Dict[str, Any]) -> float:
        """Max one-way distance (km) keeping enough fuel to return."""
        fuel = float(aircraft.get("currentFuel", 0))
        fuel_rate = float(aircraft.get("fuelRate", 1))
        speed_knots = float(aircraft.get("speed", 0))

        if fuel_rate <= 0 or speed_knots <= 0:
            return 0.0

        flight_hours = fuel / fuel_rate
        total_range_km = flight_hours * speed_knots * 1.852
        one_way = total_range_km / 2.0
        return one_way * (1.0 - self.safety_margin)

    def is_reachable_by_any(
        self,
        aircraft_list: List[Dict[str, Any]],
        base_lat: float, base_lon: float,
        target_lat: float, target_lon: float,
    ) -> bool:
        """Can at least one aircraft reach this target and return?"""
        for ac in aircraft_list:
            dist = _haversine_km(base_lat, base_lon, target_lat, target_lon)
            if dist <= self.max_one_way_km(ac):
                return True
        return False


# ---------------------------------------------------------------------------
# Aircraft template pool
# ---------------------------------------------------------------------------

class AircraftPool:
    """Stores aircraft templates keyed by className.

    Templates are extracted from scenario JSONs. Each template is a full
    aircraft dict (with weapons, fuel, etc.) ready to be cloned into a
    new scenario with fresh UUIDs.
    """

    def __init__(self):
        self._templates: Dict[str, Dict[str, Any]] = {}

    @property
    def class_names(self) -> List[str]:
        return list(self._templates.keys())

    def __len__(self) -> int:
        return len(self._templates)

    def add_from_scenario_file(self, path: str) -> None:
        """Extract aircraft templates from a scenario JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        scenario = data["currentScenario"]
        for airbase in scenario.get("airbases", []):
            for ac in airbase.get("aircraft", []):
                class_name = ac.get("className", "")
                if class_name and class_name not in self._templates:
                    self._templates[class_name] = copy.deepcopy(ac)

    def pick(self, rng: random.Random) -> Dict[str, Any]:
        """Return a deep copy of a random template with fresh UUIDs."""
        if not self._templates:
            raise ValueError("Aircraft pool is empty")

        template = rng.choice(list(self._templates.values()))
        return self._stamp_new_ids(template, rng)

    def pick_by_class(self, class_name: str, rng: random.Random) -> Dict[str, Any]:
        """Return a deep copy of a specific class template with fresh UUIDs."""
        if class_name not in self._templates:
            raise KeyError(
                f"No template for '{class_name}'. "
                f"Available: {self.class_names}"
            )
        return self._stamp_new_ids(self._templates[class_name], rng)

    @staticmethod
    def _stamp_new_ids(
        template: Dict[str, Any], rng: random.Random
    ) -> Dict[str, Any]:
        """Deep copy a template and assign fresh UUIDs + tail number."""
        ac = copy.deepcopy(template)
        ac["id"] = _new_uuid()

        tail_num = rng.randint(100, 999)
        class_name = ac.get("className", "Aircraft")
        ac["name"] = f"{class_name} #{tail_num}"

        for weapon in ac.get("weapons", []):
            weapon["id"] = _new_uuid()

        return ac


# ---------------------------------------------------------------------------
# Facility template pool
# ---------------------------------------------------------------------------

class FacilityPool:
    """Stores facility templates keyed by className.

    Templates are extracted from scenario JSONs — only RED-side facilities.
    Each template is a full facility dict (with weapons) ready to be cloned
    into a new scenario with fresh UUIDs.

    Same pattern as AircraftPool.
    """

    def __init__(self):
        self._templates: Dict[str, Dict[str, Any]] = {}

    @property
    def class_names(self) -> List[str]:
        return list(self._templates.keys())

    def __len__(self) -> int:
        return len(self._templates)

    def add_from_scenario_file(self, path: str) -> None:
        """Extract RED facility templates from a scenario JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        scenario = data["currentScenario"]
        for facility in scenario.get("facilities", []):
            # Only RED-side facilities are targets
            if facility.get("sideColor", "").lower() != "red":
                continue
            class_name = facility.get("className", "")
            if class_name and class_name not in self._templates:
                self._templates[class_name] = copy.deepcopy(facility)

    def pick(self, rng: random.Random) -> Dict[str, Any]:
        """Return a deep copy of a random template with fresh UUIDs."""
        if not self._templates:
            raise ValueError("Facility pool is empty")

        template = rng.choice(list(self._templates.values()))
        return self._stamp_new_ids(template, rng)

    def pick_by_class(self, class_name: str, rng: random.Random) -> Dict[str, Any]:
        """Return a deep copy of a specific class template with fresh UUIDs."""
        if class_name not in self._templates:
            raise KeyError(
                f"No template for '{class_name}'. "
                f"Available: {self.class_names}"
            )
        return self._stamp_new_ids(self._templates[class_name], rng)

    @staticmethod
    def _stamp_new_ids(
        template: Dict[str, Any], rng: random.Random
    ) -> Dict[str, Any]:
        """Deep copy a template and assign fresh UUIDs + name."""
        fac = copy.deepcopy(template)
        fac["id"] = _new_uuid()

        num = rng.randint(1000, 9999)
        class_name = fac.get("className", "Facility")
        fac["name"] = f"{class_name} #{num}"

        for weapon in fac.get("weapons", []):
            weapon["id"] = _new_uuid()

        return fac


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

class ScenarioGenerator:
    """Generates BLADE scenario JSON variations from a base template.

    Args:
        base_scenario_path: Path to the primary template JSON.
        extra_template_paths: Additional JSONs to extract aircraft
                              templates from.
        output_dir: Where generated scenarios are saved.
    """

    def __init__(
        self,
        base_scenario_path: str,
        extra_template_paths: Optional[List[str]] = None,
        output_dir: str = "generated_scenarios",
    ):
        self.base_path = Path(base_scenario_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        with open(self.base_path, "r", encoding="utf-8") as f:
            self._base_data = json.load(f)

        # Cache side IDs
        self._blue_side_id, self._red_side_id = self._identify_sides()

        # Build aircraft pool from base + extras
        self.aircraft_pool = AircraftPool()
        self.aircraft_pool.add_from_scenario_file(str(self.base_path))
        for extra in (extra_template_paths or []):
            self.aircraft_pool.add_from_scenario_file(extra)

        # Build facility pool from base + extras (same sources)
        self.facility_pool = FacilityPool()
        self.facility_pool.add_from_scenario_file(str(self.base_path))
        for extra in (extra_template_paths or []):
            self.facility_pool.add_from_scenario_file(extra)

        logger.info(
            f"ScenarioGenerator ready: base={self.base_path.name}, "
            f"aircraft_pool={self.aircraft_pool.class_names}, "
            f"facility_pool={self.facility_pool.class_names}"
        )

    # ---- Side identification ----

    def _identify_sides(self) -> Tuple[str, str]:
        scenario = self._base_data["currentScenario"]
        blue_id = red_id = ""
        for side in scenario["sides"]:
            if side["color"].lower() == "blue":
                blue_id = side["id"]
            elif side["color"].lower() == "red":
                red_id = side["id"]
        return blue_id, red_id

    # ---- Accessors ----

    def _get_blue_base(self, scenario: Dict) -> Tuple[float, float, Dict]:
        for ab in scenario.get("airbases", []):
            if ab.get("sideId") == self._blue_side_id:
                return ab["latitude"], ab["longitude"], ab
        raise ValueError("No BLUE airbase found in scenario")

    def _get_blue_aircraft(self, scenario: Dict) -> List[Dict]:
        _, _, blue_base = self._get_blue_base(scenario)
        return blue_base.get("aircraft", [])

    def _get_red_facilities(self, scenario: Dict) -> List[Dict]:
        return [
            f for f in scenario.get("facilities", [])
            if f.get("sideId") == self._red_side_id
        ]

    def _get_red_airbases(self, scenario: Dict) -> List[Dict]:
        return [
            ab for ab in scenario.get("airbases", [])
            if ab.get("sideId") == self._red_side_id
        ]

    # ==================================================================
    # CORE: Generate a varied scenario
    # ==================================================================

    def generate(
        self, episode: int, config: Optional[VariationConfig] = None
    ) -> Path:
        """Generate a scenario variation and save it to disk."""
        config = config or VariationConfig()
        rng = random.Random(config.seed if config.seed is not None else None)
        reachability = ReachabilityCalculator(
            safety_margin=config.fuel_safety_margin
        )

        data = copy.deepcopy(self._base_data)
        scenario = data["currentScenario"]

        # Step 1: Adjust aircraft count and/or filter by allowed classes
        desired_aircraft = _resolve_range(config.num_aircraft, rng)
        if desired_aircraft is not None or config.allowed_aircraft_classes:
            # If only filtering (no count change), pass current count as desired
            if desired_aircraft is None:
                _, _, bb = self._get_blue_base(scenario)
                desired_aircraft = len(bb.get("aircraft", []))
            self._adjust_aircraft_count(
                scenario, desired_aircraft, rng,
                allowed_classes=config.allowed_aircraft_classes,
            )

        # Step 2: Adjust facility count (SAM sites)
        if not config.include_sams:
            # Remove ALL facilities — targets will be RED airbases only
            self._adjust_facility_count(scenario, 0, rng)
            logger.info("  include_sams=False → removed all SAM facilities")
        else:
            desired_facilities = _resolve_range(config.num_facilities, rng)
            if desired_facilities is not None:
                self._adjust_facility_count(scenario, desired_facilities, rng)

        # Step 3: Adjust RED airbase count
        desired_red_ab = _resolve_range(config.num_red_airbases, rng)
        if desired_red_ab is not None:
            self._adjust_red_airbase_count(scenario, desired_red_ab, rng)

        # Step 4: Randomize blue base position
        if config.randomize_base_position:
            self._randomize_base_position(
                scenario, config.base_shift_radius_km, rng
            )

        # Step 5: Randomize target positions
        if config.include_sams and config.randomize_facility_positions:
            self._randomize_target_positions(
                self._get_red_facilities(scenario),
                scenario, reachability, config, rng,
            )
        if config.randomize_red_airbase_positions:
            self._randomize_target_positions(
                self._get_red_airbases(scenario),
                scenario, reachability, config, rng,
            )

        # Step 6: Tag with episode metadata
        scenario["name"] = f"episode_{episode:04d}"
        scenario["id"] = _new_uuid()

        # Step 7: Save
        filename = f"episode_{episode:04d}_scenario.json"
        out_path = self.output_dir / filename
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        return out_path

    # ==================================================================
    # Position randomization (shared for facilities and RED airbases)
    # ==================================================================

    def _randomize_target_positions(
        self,
        targets: List[Dict],
        scenario: Dict,
        reachability: ReachabilityCalculator,
        config: VariationConfig,
        rng: random.Random,
        max_attempts: int = 100,
    ) -> None:
        """Randomize positions of a list of target dicts.

        Works for both facilities and RED airbases — any dict with
        latitude/longitude fields.
        """
        base_lat, base_lon, _ = self._get_blue_base(scenario)
        aircraft_list = self._get_blue_aircraft(scenario)

        if not aircraft_list:
            return

        fuel_range_km = min(
            reachability.max_one_way_km(ac) for ac in aircraft_list
        )
        max_radius = fuel_range_km
        if config.max_target_distance_km is not None:
            max_radius = min(max_radius, config.max_target_distance_km)

        min_radius = config.min_target_distance_km
        if min_radius >= max_radius:
            min_radius = max_radius * 0.1

        for target in targets:
            for _ in range(max_attempts):
                new_lat, new_lon = _random_point_in_ring(
                    base_lat, base_lon, min_radius, max_radius, rng,
                )
                if reachability.is_reachable_by_any(
                    aircraft_list, base_lat, base_lon, new_lat, new_lon
                ):
                    target["latitude"] = new_lat
                    target["longitude"] = new_lon
                    break

    # ==================================================================
    # Blue base randomization
    # ==================================================================

    def _randomize_base_position(
        self, scenario: Dict, shift_radius_km: float, rng: random.Random,
    ) -> None:
        """Move the blue airbase (and its aircraft) to a random nearby position."""
        old_lat, old_lon, blue_base = self._get_blue_base(scenario)

        new_lat, new_lon = _random_point_in_ring(
            old_lat, old_lon, 0.0, shift_radius_km, rng,
        )

        blue_base["latitude"] = new_lat
        blue_base["longitude"] = new_lon

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
        """Add or remove RED facilities (pool-based).

        New facilities are picked randomly from the facility pool,
        giving diverse target compositions (Tor-M2, Pantsir-S1, etc.).
        """
        red_facilities = self._get_red_facilities(scenario)
        current = len(red_facilities)
        desired = max(desired, 0)  # 0 is valid (e.g. include_sams=False)

        if desired == current:
            return

        if desired < current:
            to_remove = rng.sample(red_facilities, current - desired)
            remove_ids = {f["id"] for f in to_remove}
            scenario["facilities"] = [
                f for f in scenario["facilities"]
                if f["id"] not in remove_ids
            ]
        else:
            for _ in range(desired - current):
                new_fac = self.facility_pool.pick(rng)

                # Set ownership to RED side
                new_fac["sideId"] = self._red_side_id
                new_fac["sideColor"] = "red"
                for weapon in new_fac.get("weapons", []):
                    weapon["sideId"] = self._red_side_id
                    weapon["sideColor"] = "red"

                scenario["facilities"].append(new_fac)

    # ==================================================================
    # RED airbase count adjustment
    # ==================================================================

    def _adjust_red_airbase_count(
        self, scenario: Dict, desired: int, rng: random.Random
    ) -> None:
        """Add or remove RED airbases (empty enemy bases as targets)."""
        red_airbases = self._get_red_airbases(scenario)
        current = len(red_airbases)
        desired = max(desired, 0)  # Can have zero RED airbases

        if desired == current:
            return

        if desired < current:
            to_remove = rng.sample(red_airbases, current - desired)
            remove_ids = {ab["id"] for ab in to_remove}
            scenario["airbases"] = [
                ab for ab in scenario["airbases"]
                if ab["id"] not in remove_ids
            ]
        else:
            if red_airbases:
                template = rng.choice(red_airbases)
            else:
                # No existing RED airbase — build a minimal one
                template = {
                    "id": "",
                    "name": "",
                    "sideId": self._red_side_id,
                    "className": "Airfield",
                    "latitude": 0.0,
                    "longitude": 0.0,
                    "altitude": 0,
                    "sideColor": "red",
                    "aircraft": [],
                }

            for i in range(desired - current):
                new_ab = copy.deepcopy(template)
                new_ab["id"] = _new_uuid()
                num = rng.randint(1000, 9999)
                new_ab["name"] = f"Enemy Airbase #{num}"
                new_ab["aircraft"] = []  # Always empty
                scenario["airbases"].append(new_ab)

    # ==================================================================
    # Aircraft count adjustment (pool-based)
    # ==================================================================

    def _adjust_aircraft_count(
        self, scenario: Dict, desired: int, rng: random.Random,
        allowed_classes: Optional[List[str]] = None,
    ) -> None:
        """Add or remove aircraft from the BLUE airbase.

        Args:
            scenario: The scenario dict to modify.
            desired: Target number of aircraft.
            rng: Random number generator.
            allowed_classes: If set, only these aircraft types are kept.
                Existing aircraft of other types are removed first.
                New aircraft are sampled uniformly from this list.
                None = use any type from the pool (original behavior).
        """
        base_lat, base_lon, blue_base = self._get_blue_base(scenario)
        aircraft_list = blue_base.get("aircraft", [])

        # Step 1: Filter by allowed classes (before adjusting count)
        if allowed_classes:
            aircraft_list = [
                ac for ac in aircraft_list
                if ac.get("className", "") in allowed_classes
            ]
            blue_base["aircraft"] = aircraft_list

        current = len(aircraft_list)
        desired = max(desired, 1)

        if desired == current:
            return

        if desired < current:
            # Randomly select which aircraft to keep (not just first N)
            blue_base["aircraft"] = rng.sample(aircraft_list, desired)
        else:
            for _ in range(desired - current):
                # Pick a random class from the allowed list, or from full pool
                if allowed_classes:
                    cls = rng.choice(allowed_classes)
                    new_ac = self.aircraft_pool.pick_by_class(cls, rng)
                else:
                    new_ac = self.aircraft_pool.pick(rng)

                # Set ownership and position to match this blue base
                new_ac["homeBaseId"] = blue_base["id"]
                new_ac["sideId"] = self._blue_side_id
                new_ac["sideColor"] = "blue"

                # Match position of existing aircraft, or use base offset
                if aircraft_list:
                    ref = aircraft_list[0]
                    new_ac["latitude"] = ref["latitude"]
                    new_ac["longitude"] = ref["longitude"]
                    new_ac["altitude"] = ref["altitude"]
                else:
                    new_ac["latitude"] = base_lat - 0.5
                    new_ac["longitude"] = base_lon - 0.5
                    new_ac["altitude"] = 10000

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
        config = config or VariationConfig()
        paths = []
        for i in range(num_episodes):
            ep = start_episode + i
            ep_config = copy.deepcopy(config)
            if ep_config.seed is None:
                ep_config.seed = ep
            paths.append(self.generate(episode=ep, config=ep_config))
        return paths


# ---------------------------------------------------------------------------
# Quick test / demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    base_path = sys.argv[1] if len(sys.argv) > 1 else "strike_training_4v5.json"

    gen = ScenarioGenerator(base_scenario_path=base_path)
    print(f"Aircraft pool: {gen.aircraft_pool.class_names}")
    print(f"Facility pool: {gen.facility_pool.class_names}")

    configs = [
        # Episode 0: Base scenario, just randomize positions
        VariationConfig(randomize_facility_positions=True, seed=42),
        # Episode 1: 3 aircraft (from pool), 4 facilities, 2 RED airbases
        VariationConfig(
            num_aircraft=3, num_facilities=4, num_red_airbases=2,
            max_target_distance_km=400.0, seed=43,
        ),
        # Episode 2: 5 aircraft, 3 facilities, 1 RED airbase, base shift
        VariationConfig(
            num_aircraft=5, num_facilities=3, num_red_airbases=1,
            randomize_base_position=True, base_shift_radius_km=150.0,
            seed=44,
        ),
        # Episode 3: Random everything
        VariationConfig(
            num_aircraft=(2, 5), num_facilities=(2, 5),
            num_red_airbases=(0, 3),
            randomize_base_position=True, base_shift_radius_km=200.0,
            max_target_distance_km=500.0, seed=45,
        ),
    ]

    for ep, cfg in enumerate(configs):
        path = gen.generate(episode=ep, config=cfg)
        print(f"\nEpisode {ep}: {path}")

        with open(path) as f:
            data = json.load(f)
        sc = data["currentScenario"]
        blue_ab = next(
            ab for ab in sc["airbases"] if ab["sideColor"] == "blue"
        )
        red_abs = [ab for ab in sc["airbases"] if ab["sideColor"] == "red"]

        ac_types = [ac["className"] for ac in blue_ab.get("aircraft", [])]
        fac_types = [f["className"] for f in sc["facilities"]]
        print(f"  Aircraft ({len(ac_types)}): {ac_types}")
        print(f"  Facilities ({len(fac_types)}): {fac_types}")
        print(f"  RED airbases: {len(red_abs)}")
        print(f"  Base: ({blue_ab['latitude']:.2f}, {blue_ab['longitude']:.2f})")
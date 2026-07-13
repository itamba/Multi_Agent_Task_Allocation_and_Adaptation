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
# Class-based range tiers
# ---------------------------------------------------------------------------
# Target one-way effective range (km) per aircraft class. The generator
# overrides `currentFuel` (and `maxFuel`) on each aircraft so its reachable
# range matches its tier, creating natural allocation constraints: short-
# range fighters must take close targets, long-range platforms take far ones.
#
# Classes not listed here are left untouched (their fuel comes from the
# base template unchanged).
CLASS_RANGE_TIERS: Dict[str, float] = {
    "F-16 Fighting Falcon": 400.0,
    "F-35A Lightning II": 900.0,
    "B-2 Spirit": 1500.0,
    "KC-135R Stratotanker": 2100.0,
}


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
    # None = no upper cap; easy zone extends up to the shortest fleet range.
    max_target_distance_km: Optional[float] = None
    min_target_distance_km: float = 50.0

    # --- Stretch targets ---
    # Fraction of targets placed in the "stretch zone" — beyond the range
    # of the weakest aircraft but within range of the strongest.
    # Creates allocation constraints: the solver must assign long-range
    # agents to far targets and short-range agents to close ones.
    # 0.0 = all targets reachable by everyone (original behavior).
    # Has no effect when all aircraft have the same range (e.g. all F-35s).
    stretch_target_ratio: float = 0.5

    # --- Time-feasibility cap on stretch_max (km, one-way) ---
    # When set, caps stretch_max so the slowest aircraft in the fleet
    # can round-trip a stretch target within MAX_SIM_TICKS at cruise speed
    # with the same safety margin as the fuel reachability calculator.
    # None = no cap (preserves pre-fix behaviour; produces (c) timeouts on
    # slow agents). Auto-computed at ScenarioGenerator init from the pool
    # if not explicitly set on the VariationConfig. Pass a very large
    # number (e.g. 1e9) to explicitly disable for ablation runs.
    time_feasible_max_km: Optional[float] = None

    # --- Blue base randomization ---
    randomize_base_position: bool = False
    base_shift_radius_km: float = 200.0

    # --- Fuel safety margin (0.0 - 1.0) ---
    fuel_safety_margin: float = 0.2

    # --- Discovery-chain connectivity radius (km) ---
    # Radius used by `_ensure_discovery_chain` to decide whether two same-zone
    # targets are radar/sensor neighbours (Layer 1 connectivity).
    #   None = legacy behaviour: derive it from the fleet's min BLADE `aircraft.range`.
    #   float = use this fixed radius instead (e.g. the graph model's unified
    #           DETECTION_KM). Threaded IN so the graph training path can build
    #           connectivity at the SAME radius the split checks and the runtime
    #           senses at — closing the "discoverable at a radius never sensed"
    #           gap. Does NOT change zone semantics; only the connectivity radius.
    detection_km: Optional[float] = None

    # --- Random seed (None = random each time) ---
    seed: Optional[int] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_time_feasible_one_way_km(
    aircraft_pool_speeds_kmh: List[float],
    max_ticks: int,
    safety_margin: float = 0.3,
) -> float:
    """One-way distance the slowest aircraft can round-trip within
    ``max_ticks`` seconds at cruise speed with the given safety margin.

    Each tick is one second of simulation time (BLADE convention).
    The slowest agent is binding because if it can't time-round-trip a
    stretch target, the solver will sometimes assign the trip anyway
    (it only enforces fuel-budget) and the episode hits MAX_SIM_TICKS
    while the agent is still airborne — the !TIMEOUT category (c) signal.

    Args:
        aircraft_pool_speeds_kmh: cruise speeds (km/h) of the eligible
            aircraft pool (post `--allowed-aircraft` filtering).
        max_ticks: simulation tick cap (1 tick = 1 second).
        safety_margin: fraction subtracted from the time budget. Default
            0.3 to match `ReachabilityCalculator.safety_margin=0.3`.

    Returns:
        One-way km the slowest agent can round-trip within max_ticks.
    """
    if not aircraft_pool_speeds_kmh:
        raise ValueError("aircraft_pool_speeds_kmh is empty")
    slowest_kmh = min(aircraft_pool_speeds_kmh)
    slowest_kms = slowest_kmh / 3600.0
    round_trip_max = max_ticks * slowest_kms * (1.0 - safety_margin)
    return round_trip_max / 2.0


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


def _fuel_for_range(
    aircraft: Dict[str, Any], target_one_way_km: float, safety_margin: float,
) -> Optional[float]:
    """Invert ReachabilityCalculator: fuel needed for a given effective range.

    Returns None if speed/fuelRate/margin make the computation ill-defined.
    """
    fuel_rate = float(aircraft.get("fuelRate", 0))
    speed_knots = float(aircraft.get("speed", 0))
    denom = speed_knots * 1.852 * (1.0 - safety_margin)
    if fuel_rate <= 0 or denom <= 0:
        return None
    return 2.0 * target_one_way_km * fuel_rate / denom


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

    def speeds_kmh(self, allowed_classes: Optional[List[str]] = None) -> List[float]:
        """Return cruise speeds (km/h) for eligible templates.

        ``allowed_classes`` mirrors ``VariationConfig.allowed_aircraft_classes``;
        when None, includes the full pool. Skips templates with speed=0.
        """
        out: List[float] = []
        for cls, template in self._templates.items():
            if allowed_classes is not None and cls not in allowed_classes:
                continue
            speed_knots = float(template.get("speed", 0))
            if speed_knots > 0:
                out.append(speed_knots * 1.852)
        return out

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
        max_sim_ticks: simulation tick cap (1 tick = 1 second). Used to
            auto-compute ``time_feasible_max_km``. When None (default),
            no auto-compute happens and ``time_feasible_max_km`` defaults
            to the value on each ``VariationConfig`` (None = no cap,
            preserving the pre-fix behaviour).
        time_feasible_safety_margin: matches `ReachabilityCalculator`'s
            safety margin (default 0.3) so the time cap and the fuel cap
            apply the same reserve.
    """

    def __init__(
        self,
        base_scenario_path: str,
        extra_template_paths: Optional[List[str]] = None,
        output_dir: str = "generated_scenarios",
        max_sim_ticks: Optional[int] = None,
        time_feasible_safety_margin: float = 0.3,
    ):
        self.base_path = Path(base_scenario_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_sim_ticks = max_sim_ticks
        self.time_feasible_safety_margin = time_feasible_safety_margin
        self.time_feasible_max_km_default: Optional[float] = None
        # Slowest-agent diagnostics for the run header / log.
        self.time_feasible_inputs: Dict[str, Any] = {}

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

        # Snapshot of stats from the most recent generate() call. Read by the
        # training loop to build the per-episode summary line. Keys:
        #   n_easy, n_stretch                    (target placement zones)
        #   easy_relocated, easy_total, easy_isolated
        #   stretch_relocated, stretch_total, stretch_isolated
        #   reachable_count, total_targets       (reachability audit)
        #   min_radar_km
        self.last_generation_stats: Dict[str, Any] = {}

        # Auto-compute time-feasibility cap when max_sim_ticks is provided.
        # The cap can be refined later (e.g. after applying
        # --allowed-aircraft) via recompute_time_feasible_cap().
        if self.max_sim_ticks is not None:
            self.recompute_time_feasible_cap(allowed_classes=None)

        logger.info(
            f"ScenarioGenerator ready: base={self.base_path.name}, "
            f"aircraft_pool={self.aircraft_pool.class_names}, "
            f"facility_pool={self.facility_pool.class_names}"
        )

    def recompute_time_feasible_cap(
        self, allowed_classes: Optional[List[str]] = None,
    ) -> Optional[float]:
        """Recompute the auto-default ``time_feasible_max_km`` from the
        slowest aircraft in the eligible pool.

        Call this after applying ``allowed_aircraft_classes`` if you want
        the cap to track the filtered fleet (so removing the slowest agent
        relaxes the cap accordingly). Stores the result on the generator.

        Returns the computed value (or None when max_sim_ticks isn't set).
        """
        if self.max_sim_ticks is None:
            return None
        speeds = self.aircraft_pool.speeds_kmh(allowed_classes=allowed_classes)
        if not speeds:
            return None
        cap = compute_time_feasible_one_way_km(
            speeds, self.max_sim_ticks, self.time_feasible_safety_margin,
        )
        self.time_feasible_max_km_default = cap
        slowest_kmh = min(speeds)
        # Find the class name corresponding to the slowest speed for logging
        slowest_class = "?"
        for cls, template in self.aircraft_pool._templates.items():
            if allowed_classes is not None and cls not in allowed_classes:
                continue
            sk = float(template.get("speed", 0))
            if sk > 0 and abs(sk * 1.852 - slowest_kmh) < 0.5:
                slowest_class = cls
                break
        self.time_feasible_inputs = {
            "slowest_class": slowest_class,
            "slowest_kmh": slowest_kmh,
            "max_ticks": self.max_sim_ticks,
            "safety": self.time_feasible_safety_margin,
            "cap_km": cap,
        }
        return cap

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
        # Inject the auto-computed time-feasibility cap into the config
        # when the caller didn't override it. The cap is derived from the
        # eligible pool (post `allowed_aircraft_classes` filter) so a run
        # that removes the slowest aircraft relaxes the cap accordingly.
        if config.time_feasible_max_km is None and self.max_sim_ticks is not None:
            cap = self.recompute_time_feasible_cap(
                allowed_classes=config.allowed_aircraft_classes,
            )
            if cap is not None:
                config.time_feasible_max_km = cap
        rng = random.Random(config.seed if config.seed is not None else None)
        # Reset stats snapshot for this generation pass
        self.last_generation_stats = {}
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

        # Step 1.5: Override fuel per class tier so fleet ranges are
        # meaningfully differentiated (F-16 short, F-35A medium, etc.).
        self._apply_fuel_tiers(scenario, config.fuel_safety_margin)

        # Step 2: Adjust facility count (SAM sites)
        if not config.include_sams:
            # Remove ALL facilities — targets will be RED airbases only
            self._adjust_facility_count(scenario, 0, rng)
            logger.debug("  include_sams=False → removed all SAM facilities")
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

        # Step 5.25: Discovery chain — ensure every target has a radar-visible
        # neighbor *within its own zone*, so an agent flying to any known
        # target in a given zone can detect nearby masked targets in the
        # same zone (cross-zone discovery is impossible: zones are separated
        # by far more than radar range).
        self._ensure_discovery_chain(scenario, reachability, config, rng)

        # Step 5.5: Reachability audit (read-only)
        base_lat, base_lon, _ = self._get_blue_base(scenario)
        aircraft_list = self._get_blue_aircraft(scenario)
        all_targets = (
            self._get_red_facilities(scenario) + self._get_red_airbases(scenario)
        )
        reachable_count = 0
        total_count = len(all_targets)
        for target in all_targets:
            if reachability.is_reachable_by_any(
                aircraft_list,
                base_lat, base_lon,
                target["latitude"], target["longitude"],
            ):
                reachable_count += 1
            else:
                # Frequent under stretch placement — debug only.
                logger.debug(
                    "Target '%s' is unreachable by all agents - "
                    "expected behavior for stretch targets", target["name"]
                )
        logger.debug(
            "Reachability audit: %d/%d targets reachable by at least one agent",
            reachable_count, total_count
        )
        self.last_generation_stats["reachable_count"] = reachable_count
        self.last_generation_stats["total_targets"] = total_count

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

        Two placement zones:
        - Easy zone: [min_dist, min(min_range, max_target_distance_km)]
          All aircraft can reach these targets.
        - Stretch zone: [min_range, max_range]
          Only longer-range aircraft can reach these. Ignores
          max_target_distance_km since stretch targets are intentionally
          beyond the easy cap to test solver allocation.

        When all aircraft have the same range (homogeneous fleet),
        the stretch zone is empty and all targets land in the easy zone.
        """
        base_lat, base_lon, _ = self._get_blue_base(scenario)
        aircraft_list = self._get_blue_aircraft(scenario)

        if not aircraft_list:
            return

        # Per-aircraft ranges
        ranges = [reachability.max_one_way_km(ac) for ac in aircraft_list]
        min_range = min(ranges)
        max_range = max(ranges)

        # Easy zone: all aircraft can reach
        easy_max = min_range
        if config.max_target_distance_km is not None:
            easy_max = min(easy_max, config.max_target_distance_km)
        easy_min = config.min_target_distance_km
        if easy_min >= easy_max:
            easy_min = easy_max * 0.1

        # Stretch zone: only some aircraft can reach
        # Only meaningful when aircraft have different ranges
        range_gap = max_range - min_range
        has_stretch = (
            config.stretch_target_ratio > 0
            and range_gap > 50  # At least 50km gap to be meaningful
        )

        if has_stretch:
            # Small buffer so stretch targets are clearly beyond min_range
            stretch_min = min_range + range_gap * 0.1
            stretch_max = max_range
            # Apply time-feasibility cap so the slowest aircraft can
            # round-trip a stretch target within MAX_SIM_TICKS at cruise
            # speed. Without this, the MATCH-AOU solver (which only
            # enforces fuel-budget) routinely assigns time-infeasible
            # stretch missions and the episode hits the tick cap with
            # the agent still airborne — !TIMEOUT category (c).
            if config.time_feasible_max_km is not None:
                if stretch_max > config.time_feasible_max_km:
                    stretch_max = config.time_feasible_max_km
                if stretch_max <= stretch_min:
                    logger.debug(
                        f"  Stretch zone collapsed by time-feasibility cap "
                        f"(stretch_max={stretch_max:.0f} ≤ stretch_min={stretch_min:.0f})"
                    )
                    has_stretch = False
        if has_stretch:
            n_stretch = max(1, round(len(targets) * config.stretch_target_ratio))
            n_easy = len(targets) - n_stretch
            logger.debug(
                f"  Target placement: {n_easy} easy (≤{easy_max:.0f}km), "
                f"{n_stretch} stretch ({stretch_min:.0f}–{stretch_max:.0f}km)"
            )
        else:
            n_easy = len(targets)
            n_stretch = 0
            if config.stretch_target_ratio > 0 and range_gap <= 50:
                logger.debug(
                    f"  Stretch targets disabled: fleet range gap "
                    f"({range_gap:.0f}km) too small for differentiation"
                )
        # Track placement counts so the training loop can render the
        # tg=N[Xe+Ys] portion of the per-episode summary line.
        self.last_generation_stats["n_easy"] = (
            self.last_generation_stats.get("n_easy", 0) + n_easy
        )
        self.last_generation_stats["n_stretch"] = (
            self.last_generation_stats.get("n_stretch", 0) + n_stretch
        )

        # Shuffle targets so stretch assignment is random
        shuffled_indices = list(range(len(targets)))
        rng.shuffle(shuffled_indices)

        for i, target_idx in enumerate(shuffled_indices):
            target = targets[target_idx]
            is_stretch = (i >= n_easy)  # First n_easy are easy, rest stretch

            if is_stretch:
                ring_min, ring_max = stretch_min, stretch_max
            else:
                ring_min, ring_max = easy_min, easy_max

            placed = False
            for _ in range(max_attempts):
                new_lat, new_lon = _random_point_in_ring(
                    base_lat, base_lon, ring_min, ring_max, rng,
                )
                if reachability.is_reachable_by_any(
                    aircraft_list, base_lat, base_lon, new_lat, new_lon
                ):
                    target["latitude"] = new_lat
                    target["longitude"] = new_lon
                    placed = True
                    break

            # Fallback: if stretch target couldn't be placed, try easy zone
            if not placed and is_stretch:
                for _ in range(max_attempts):
                    new_lat, new_lon = _random_point_in_ring(
                        base_lat, base_lon, easy_min, easy_max, rng,
                    )
                    if reachability.is_reachable_by_any(
                        aircraft_list, base_lat, base_lon, new_lat, new_lon
                    ):
                        target["latitude"] = new_lat
                        target["longitude"] = new_lon
                        logger.debug(
                            f"  Stretch target fell back to easy zone"
                        )
                        break

    # ==================================================================
    # Discovery chain (radar-neighbor connectivity)
    # ==================================================================

    def _compute_zone_bounds(
        self,
        aircraft_list: List[Dict[str, Any]],
        config: VariationConfig,
        reachability: ReachabilityCalculator,
    ) -> Tuple[float, float, Optional[float], Optional[float]]:
        """Re-derive the easy/stretch zone radii from base.

        Mirrors the boundary logic in `_randomize_target_positions` so the
        discovery-chain step can classify targets by zone and constrain
        relocations to stay within the same zone.

        Returns:
            (easy_min, easy_max, stretch_min, stretch_max). The stretch
            bounds are None when the fleet has no meaningful range
            differentiation (gap ≤ 50 km or stretch_target_ratio == 0).
        """
        ranges = [reachability.max_one_way_km(ac) for ac in aircraft_list]
        min_range = min(ranges)
        max_range = max(ranges)

        easy_max = min_range
        if config.max_target_distance_km is not None:
            easy_max = min(easy_max, config.max_target_distance_km)
        easy_min = config.min_target_distance_km
        if easy_min >= easy_max:
            easy_min = easy_max * 0.1

        range_gap = max_range - min_range
        has_stretch = (
            config.stretch_target_ratio > 0 and range_gap > 50
        )
        stretch_min = (min_range + range_gap * 0.1) if has_stretch else None
        stretch_max = max_range if has_stretch else None
        # Mirror the time-feasibility cap from _randomize_target_positions
        # so the discovery-chain step classifies targets against the same
        # zone boundary the placement loop will use.
        if (
            stretch_max is not None
            and config.time_feasible_max_km is not None
            and stretch_max > config.time_feasible_max_km
        ):
            stretch_max = config.time_feasible_max_km
        if (
            stretch_min is not None and stretch_max is not None
            and stretch_max <= stretch_min
        ):
            stretch_min = stretch_max = None
        return easy_min, easy_max, stretch_min, stretch_max

    def _ensure_discovery_chain(
        self,
        scenario: Dict,
        reachability: ReachabilityCalculator,
        config: VariationConfig,
        rng: random.Random,
        max_attempts: int = 50,
    ) -> None:
        """Per-zone connectivity for radar-discovery during masked episodes.

        Goal:
        Guarantee that within each placement zone (easy, stretch), every
        target has at least one radar-range neighbor in the *same zone*.
        This is the graph-level precondition that makes the downstream
        split-time mask-aware sampler in `train_full.py` solvable: with
        every target connected to at least one same-zone peer, the
        sampler can almost always find a partition where every hidden
        target has at least one known same-zone neighbor.

        Why per-zone, not graph-wide:
        The zone system is the project's primary mechanism for forcing
        heterogeneous fleet allocation — short-range fighters take easy
        targets, long-range platforms take stretch targets. The two zones
        are separated by hundreds of kilometres along the radial axis,
        far beyond any aircraft radar (typically 50–200 km). Cross-zone
        radar discovery is therefore physically impossible. A relocation
        that crossed the zone boundary would silently demote a stretch
        target into easy or vice versa, breaking allocation pressure
        without producing any extra discovery benefit. We constrain
        relocations to stay in the same zone (both the radar ring around
        an in-zone anchor *and* the [zone_min, zone_max] band from base
        must be satisfied) to keep zone semantics intact.

        Why this isn't sufficient on its own:
        Even with per-zone connectivity, masking decided downstream may
        hide both members of a connected pair, leaving them invisible.
        The split-time sampler in `train_full.py` is responsible for
        avoiding such partitions. This step's job is only to make a
        valid partition exist.

        Single-target zones, or zones where geometry makes connection
        impossible, are left as-is and logged. The split-time sampler
        will pin such isolated targets into the known set so they are
        at least seen by the solver.
        """
        aircraft_list = self._get_blue_aircraft(scenario)
        if not aircraft_list:
            return

        # Connectivity radius source. When the caller supplies `detection_km` (the
        # graph model's unified sensing/attack/arrival radius) use it verbatim, so
        # generator connectivity is built at the SAME radius the split checks and the
        # runtime senses at. Otherwise fall back to the legacy min-fleet-radar radius
        # derived from BLADE `aircraft.range`. Only the *source* of this radius
        # changes — zone semantics below are untouched. (`min_radar_km` keeps its
        # name to localise the diff; it is just "the connectivity radius".)
        if config.detection_km is not None:
            min_radar_km = float(config.detection_km)
        else:
            radar_ranges_nm = [
                float(ac.get("range", 0)) for ac in aircraft_list
                if float(ac.get("range", 0)) > 0
            ]
            if not radar_ranges_nm:
                return
            min_radar_km = min(radar_ranges_nm) * 1.852

        all_targets = (
            self._get_red_facilities(scenario)
            + self._get_red_airbases(scenario)
        )
        if len(all_targets) < 2:
            return

        base_lat, base_lon, _ = self._get_blue_base(scenario)
        easy_min, easy_max, stretch_min, stretch_max = self._compute_zone_bounds(
            aircraft_list, config, reachability,
        )

        # Classify each target by its current distance from base.
        # Targets that fall in the inter-zone gap (between easy_max and
        # stretch_min) or beyond stretch_max are absorbed into the closest
        # zone for connectivity purposes.
        easy_targets: List[Dict[str, Any]] = []
        stretch_targets: List[Dict[str, Any]] = []
        for t in all_targets:
            d_from_base = _haversine_km(
                base_lat, base_lon, t["latitude"], t["longitude"],
            )
            if stretch_min is not None and d_from_base >= stretch_min:
                stretch_targets.append(t)
            else:
                easy_targets.append(t)

        ring_min = min_radar_km * 0.2
        ring_max = min_radar_km * 0.8

        easy_relocated, easy_isolated = self._connect_zone_targets(
            easy_targets, base_lat, base_lon,
            zone_min=easy_min, zone_max=easy_max,
            min_radar_km=min_radar_km,
            ring_min=ring_min, ring_max=ring_max,
            aircraft_list=aircraft_list, reachability=reachability,
            rng=rng, max_attempts=max_attempts,
        )
        stretch_relocated = stretch_isolated = 0
        if stretch_min is not None:
            stretch_relocated, stretch_isolated = self._connect_zone_targets(
                stretch_targets, base_lat, base_lon,
                zone_min=stretch_min, zone_max=stretch_max,
                min_radar_km=min_radar_km,
                ring_min=ring_min, ring_max=ring_max,
                aircraft_list=aircraft_list, reachability=reachability,
                rng=rng, max_attempts=max_attempts,
            )

        logger.debug(
            "Discovery chain: easy relocated=%d/%d isolated=%d, "
            "stretch relocated=%d/%d isolated=%d (min fleet radar=%.0f km)",
            easy_relocated, len(easy_targets), easy_isolated,
            stretch_relocated, len(stretch_targets), stretch_isolated,
            min_radar_km,
        )
        # Stash L1 (gen-time discovery chain) stats for the summary line.
        self.last_generation_stats["easy_relocated"] = easy_relocated
        self.last_generation_stats["easy_total"] = len(easy_targets)
        self.last_generation_stats["easy_isolated"] = easy_isolated
        self.last_generation_stats["stretch_relocated"] = stretch_relocated
        self.last_generation_stats["stretch_total"] = len(stretch_targets)
        self.last_generation_stats["stretch_isolated"] = stretch_isolated
        self.last_generation_stats["min_radar_km"] = min_radar_km

    def _connect_zone_targets(
        self,
        targets: List[Dict[str, Any]],
        base_lat: float,
        base_lon: float,
        *,
        zone_min: float,
        zone_max: float,
        min_radar_km: float,
        ring_min: float,
        ring_max: float,
        aircraft_list: List[Dict[str, Any]],
        reachability: ReachabilityCalculator,
        rng: random.Random,
        max_attempts: int,
    ) -> Tuple[int, int]:
        """Relocate isolated targets within a single zone.

        For each target with no radar-range peer in the same zone, try to
        place it within `[ring_min, ring_max]` km of an in-zone anchor,
        constrained to the band `[zone_min, zone_max]` km from base, and
        still reachable by some aircraft. If no valid placement is found
        for any anchor after `max_attempts` per-anchor tries, the target
        is left where it was and counted as `isolated` (the split-time
        sampler will pin it to the known set).

        Returns:
            (relocated_count, isolated_count) for this zone.
        """
        if len(targets) < 2:
            # Single-target zones can't have in-zone neighbors by
            # definition. They are isolated by structure, not by bad
            # placement. Layer 2 (split-time sampler) handles them.
            return 0, len(targets)

        relocated = 0
        isolated = 0
        for target in targets:
            has_neighbor = False
            for other in targets:
                if other is target:
                    continue
                d = _haversine_km(
                    target["latitude"], target["longitude"],
                    other["latitude"], other["longitude"],
                )
                if d <= min_radar_km:
                    has_neighbor = True
                    break
            if has_neighbor:
                continue

            anchors = [t for t in targets if t is not target]
            rng.shuffle(anchors)
            placed = False
            for anchor in anchors:
                for _ in range(max_attempts):
                    new_lat, new_lon = _random_point_in_ring(
                        anchor["latitude"], anchor["longitude"],
                        ring_min, ring_max, rng,
                    )
                    d_base = _haversine_km(
                        base_lat, base_lon, new_lat, new_lon,
                    )
                    if not (zone_min <= d_base <= zone_max):
                        continue
                    if not reachability.is_reachable_by_any(
                        aircraft_list, base_lat, base_lon, new_lat, new_lon,
                    ):
                        continue
                    target["latitude"] = new_lat
                    target["longitude"] = new_lon
                    placed = True
                    relocated += 1
                    break
                if placed:
                    break

            if not placed:
                isolated += 1
                logger.warning(
                    "Discovery chain: could not connect target '%s' within "
                    "zone bounds [%.0f-%.0f km]; leaving isolated",
                    target.get("name", "?"), zone_min, zone_max,
                )

        return relocated, isolated

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
    # Class-based fuel tiers
    # ==================================================================

    def _apply_fuel_tiers(
        self, scenario: Dict, safety_margin: float,
    ) -> None:
        """Override currentFuel/maxFuel on blue aircraft per CLASS_RANGE_TIERS.

        For each aircraft whose className is in the tier map, compute the
        fuel needed so its effective one-way range equals the tier value
        (given the current safety_margin, speed, and fuelRate). Aircraft
        with classes not in the map are left unchanged.
        """
        _, _, blue_base = self._get_blue_base(scenario)
        aircraft_list = blue_base.get("aircraft", [])
        for ac in aircraft_list:
            cls = ac.get("className", "")
            target_km = CLASS_RANGE_TIERS.get(cls)
            if target_km is None:
                logger.debug(
                    f"  No fuel tier for class '{cls}'; keeping template fuel"
                )
                continue
            fuel = _fuel_for_range(ac, target_km, safety_margin)
            if fuel is None:
                logger.debug(
                    f"  Cannot compute fuel for '{cls}' (speed/fuelRate "
                    f"invalid); keeping template fuel"
                )
                continue
            ac["currentFuel"] = fuel
            ac["maxFuel"] = fuel

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
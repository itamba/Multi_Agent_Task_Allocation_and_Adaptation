from enum import Enum
from typing import List, Optional
from match_aou.models.capability import Capability
from match_aou.models.location import Location


class StepKind(Enum):
    """Semantic kind of a Step.

    This is a DOMAIN concept describing what a step *means* (e.g. "this is an
    attack"), NOT a simulator instruction. Translating a StepKind into a
    concrete command (a BLADE action string) lives entirely in the executor
    (the translation layer), so domain objects stay free of simulator strings.

    Only ATTACK exists today; add new kinds here as the domain grows.
    """
    ATTACK = "attack"


class Step:
    """
    A single step in a task: pure semantic data, no simulator-specific commands.

    A Step describes WHAT must happen (act on this target, at this location, with
    these capabilities) but never HOW to express it to a particular simulator.
    The executor (GraphPlanExecutor) is the sole translation layer: it turns a
    Step plus its agent assignment into concrete BLADE commands. The Step itself
    carries no agent identity, no action template, and no placeholders.
    """

    def __init__(
        self,
        location: Optional[Location],
        target_id: Optional[str],
        capabilities: List[Capability],
        probability: float,
        effort: int,
        step_kind: StepKind,
    ):
        """
        Initialize a step.
        :param location: Location of the step (can be None for non-location-dependent steps).
        :param target_id: Semantic identifier of the target this step acts on.
        :param capabilities: List of Capability objects required for the step (solver matching).
        :param probability: Probability of successful completion (the p in (1 - p)^m).
        :param effort: Effort required to complete the step (currently inert; kept for
                       future interfaces).
        :param step_kind: Semantic kind of the step (StepKind enum, e.g. ATTACK).
        """
        self.location = location
        self.capabilities = capabilities
        self.target_id = target_id
        self.probability = probability
        self.effort = effort
        self.step_kind = step_kind

    def __repr__(self):
        return (
            f"Step(\n"
            f"  Location: {self.location},\n"
            f"  Capabilities: {self.capabilities},\n"
            f"  Target ID: {self.target_id},\n"
            f"  Probability: {self.probability},\n"
            f"  Effort: {self.effort},\n"
            f"  Step Kind: {self.step_kind}\n"
            f")"
        )

from typing import List, Optional, Dict
from match_aou.models.capability import Capability
from match_aou.models.step_type import StepType
from match_aou.models.location import Location

class Step:
    """
    Represents a single step in a task.
    Supports multi-agent execution, with each agent having a specific action and execution time.
    """

    def __init__(self, location: Optional[Location], capabilities: List[Capability], step_type: StepType,
                 effort: int, probability: float, action: Optional[str] = None):
        """
        Initialize a step.
        :param location: Location object representing the step's location (can be None for non-location-dependent tasks).
        :param capabilities: List of Capability objects required for the step.
        :param step_type: StepType object representing the type of the step (e.g., 'surveillance', 'photography').
        :param effort: Effort required to complete the step (e.g., number of photos to take).
        :param probability: Probability of successful completion by an agent.
        :param action: (Optional) Action to be executed for the step in BLADE (e.g., move_aircraft('{agent_id}', ...)).
                        Should contain a placeholder like '{agent_id}' to be filled during simulation.
        """
        self.location = location
        self.capabilities = capabilities
        self.step_type = step_type
        self.effort = effort
        self.probability = probability
        self.agent_id_placeholder='AGENT_ID'

        self.action = action  # Template string to be filled at execution time using the agent ID
        self.execution_times: Dict[str, int] = {}   # agent_id → execution time. Will be filled later based on timing logic

    def compute_step_cost(self) -> float:
        """Calculate cost of completing this step."""
        return self.step_type.compute_cost(self.effort)

    def get_action(self, agent_id: str) -> Optional[str]:
        """
        Get the action string with the agent ID inserted.
        :param agent_id: The ID of the agent performing this step.
        :return: Action string with placeholders filled.
        """
        if not self.action:
            return None
        return self.action.replace(self.agent_id_placeholder, agent_id)

    def get_execution_time(self, agent_id: Optional[str] = None):
        """
        Gets the execution time for a specific agent.
        If no agent ID is provided, returns the default execution_time .

        :param agent_id: Optional agent ID to retrieve specific time.
        :return: Execution time in time steps, or None.
        """
        return self.execution_times.get(agent_id)

    def __repr__(self):
        return (
            f"Step(\n"
            f"  Location: {self.location},\n"
            f"  Capabilities: {self.capabilities},\n"
            f"  Step Type: {self.step_type},\n"
            f"  Effort: {self.effort},\n"
            f"  Probability: {self.probability},\n"
            f"  Action Template: {self.action},\n"
            f"  Execution Time(s): {self.execution_times}\n"
            f")"
        )

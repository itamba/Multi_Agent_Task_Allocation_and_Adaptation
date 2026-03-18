from match_aou.models.location import Location


class Agent:
    """
    Represents an agent capable of completing tasks.
    """
    def __init__(
        self,
        location,
        capabilities,
        budget,
        move_cost_function,
        speed=None,
        return_location=None,
        agent_id=None,
        side_color=None,
        weapon_id=None,
        home_base_id=None,
        target_id=None
    ):
        """
        Initialize an agent.

        :param location: Location object representing the agent's starting location.
        :param capabilities: List of Capability objects representing the agent's capabilities.
        :param budget: Constraint budget of the agent (e.g., fuel).
        :param move_cost_function: Function to compute the cost of moving between locations.
        :param return_location: Location object representing where the agent must return (optional).
        :param agent_id: Unique identifier for the agent.
        :param side_color: Color representing the agent's side.
        :param home_base_id: Identifier of the agent's home base.
        :param target_id: Identifier of the agent's current target.
        """
        self.location = location
        self.capabilities = capabilities
        self.budget = budget
        self.move_cost_function = move_cost_function
        self.speed = speed
        self.return_location = return_location

        # Additional properties
        self.id = agent_id
        self.side_color = side_color
        self.weapon_id = weapon_id
        self.home_base_id = home_base_id
        self.target_id = target_id

    def move_cost(self, destination, source=None):
        """
        Calculate the cost of moving to a new location.

        :param destination: Location object representing the target location.
        :return: Movement cost.
        """
        if source is None:
            source = self.location
        if destination is None:
            return 0
        if not isinstance(destination, Location) or not isinstance(source, Location):
            raise ValueError("Source and Destination must be a Location object.")
        return self.move_cost_function(source, destination)

    def step_cost(self, step):
        """
        Calculate the cost of performing a step based on its type and effort.

        :param step: A Step object.
        :return: Cost of performing the step.
        """
        return step.step_type.compute_cost(step.effort)

    def has_capabilities(self, required_capabilities):
        """
        Check if the agent has the required capabilities for a task step.

        :param required_capabilities: List of Capability objects.
        :return: True if all required capabilities are present, False otherwise.
        """
        for req_capability in required_capabilities:
            found_capability = False
            for agent_capability in self.capabilities:
                if req_capability.name == agent_capability.name:
                    found_capability = True
                    break
            if not found_capability:
                return False
        return True

    def __repr__(self):
        return (
            f"Agent(\n"
            f"  Id: {self.id},\n"
            f"  Side Color: {self.side_color},\n"
            f"  Location: {self.location},\n"
            f"  Capabilities: {self.capabilities},\n"
            f"  Budget: {self.budget},\n"
            f"  Return Location: {self.return_location},\n"
            f"  Weapon Id: {self.weapon_id},\n"
            f"  Home Base Id: {self.home_base_id},\n"
            f")"
        )


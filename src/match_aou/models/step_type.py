class StepType:
    """
    Represents the type of step and its associated attributes.
    """
    def __init__(self, name, base_cost, custom_cost_function=None):
        """
        Initialize a step type.
        :param name: Name of the step type (e.g., "surveillance").
        :param base_cost: Base cost per unit of effort for this type.
        :param custom_cost_function: Optional function for custom cost calculation based on the step.
        """
        self.name = name
        self.base_cost = base_cost
        self.custom_cost_function = custom_cost_function  # Allows for custom cost calculation

    def compute_cost(self, effort):
        """
        Calculate the cost based on effort required.
        :param effort: Effort required to complete the step.
        :return: Total cost for the step.
        """
        if self.custom_cost_function:
            return self.custom_cost_function(effort)  # Apply custom cost function if provided
        else:
            return self.base_cost * effort  # Default cost calculation

    def __repr__(self):
        return f"StepType(name={self.name}, base_cost={self.base_cost}, custom_cost_function={self.custom_cost_function})"

class Task:
    """
    Represents a task consisting of multiple steps.
    """
    def __init__(self, steps, utility, precedence_relations=None):
        """
        Initialize a task.
        :param steps: List of Step objects representing the task's steps.
        :param utility: Utility gained upon completing the task.
        :param precedence_relations: List of tuples (k1, k2) where k1 must precede k2.
        """
        self.steps = steps
        self.utility = utility
        self.precedence_relations = precedence_relations if precedence_relations else []

    def __repr__(self):
        return (
                f"Task(\n"
                f"  Utility: {self.utility},\n"
                f"  Steps:\n" +
                "".join([f"    {repr(step)}\n" for step in self.steps]) +
                f")"
        )

    # def execute_task(self):
    #     """
    #     Executes the task by performing each step, ensuring precedence relations if present.
    #     If no precedence relations exist, execute steps in the order they appear.
    #     :return: List of actions that were executed in order.
    #     """
    #     executed_steps = []
    #
    #     if not self.precedence_relations:
    #         # If no precedence relations exist, execute steps in their current order
    #         for step in self.steps:
    #             executed_steps.append(step.get_action())
    #     else:
    #         # If precedence relations exist, execute them in the correct order
    #         for (k1, k2) in self.precedence_relations:
    #             if k1 < k2:
    #                 executed_steps.append(self.steps[k1].get_action())
    #                 executed_steps.append(self.steps[k2].get_action())
    #             else:
    #                 executed_steps.append(self.steps[k2].get_action())
    #                 executed_steps.append(self.steps[k1].get_action())
    #     return executed_steps

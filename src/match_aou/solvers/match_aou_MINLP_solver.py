from __future__ import annotations

from pyomo.environ import (
    Binary,
    ConcreteModel,
    Constraint,
    ConstraintList,
    NonNegativeIntegers,
    Objective,
    RangeSet,
    SolverFactory,
    TerminationCondition,
    Var,
    maximize,
    prod,
)

EPSILON = 1e-6  # Small safety margin to avoid numerical issues in (1 - p)**m when p is 0 or 1.


class MatchAou:
    """Solve a MATCH-AOU-style allocation problem as a MINLP in Pyomo.

    Notes
    -----
    - `y[j]` indicates whether task `j` is selected/fully allocated.
    - `x[i, j, k]` indicates whether agent `i` is assigned to step `k` of task `j`.
    - Precedence relations are enforced at the *selection* level:
      if (j1, j2) is in precedence_relations (j1 must precede j2),
      then task j2 cannot be selected unless task j1 is selected.
      This is modeled as: y[j2] <= y[j1].

    This implementation intentionally does NOT model time/scheduling variables; a topological ordering
    of selected tasks can be computed in post-processing.
    """

    def __init__(self, agents, tasks, precedence_relations=None, risk_factor: float = 0.0):
        """Initialize the model.

        Parameters
        ----------
        agents : list
            List of agent objects.
        tasks : list
            List of task objects (each task contains a list of steps).
        precedence_relations : list[tuple[int, int]] | None
            List of tuples (j1, j2) meaning task j1 must precede task j2.
        risk_factor : float
            Risk factor in [0, 1) applied to budgets as a conservative margin.
        """
        self.agents = agents
        self.tasks = tasks
        self.precedence_relations = precedence_relations if precedence_relations else []
        self.risk_factor = risk_factor

        # Create Pyomo model
        self.model = ConcreteModel()

        # Sets
        self.model.A = RangeSet(0, len(agents) - 1)  # Agents
        self.model.T = RangeSet(0, len(tasks) - 1)  # Tasks
        self.model.S = RangeSet(0, max(len(task.steps) for task in tasks) - 1)  # Step index (max over tasks)

        # Decision variables
        # x(i,j,k)=1 -> agent i is assigned to step k of task j
        self.model.x = Var(self.model.A, self.model.T, self.model.S, domain=Binary)
        # y(j)=1 -> task j is selected (and all its steps must be allocated)
        self.model.y = Var(self.model.T, domain=Binary)

        self._add_objective()
        self._add_constraints()

    def _add_objective(self) -> None:
        """Maximize expected utility of selected tasks (success probability across steps)."""

        def objective_function(model):
            return sum(model.y[j] * self.tasks[j].utility
                * prod(
                    [1 - (1 - step.probability + EPSILON) ** sum(model.x[i, j, k] for i in model.A)
                        for k, step in enumerate(self.tasks[j].steps)]
                )
                for j in model.T
            )

        self.model.obj = Objective(rule=objective_function, sense=maximize)

    def _add_constraints(self) -> None:
        """Add constraints to the model."""

        # ---------------------------------------------------------------------
        # Capability: agent can only be assigned to steps it can perform
        # ---------------------------------------------------------------------
        def capability_constraint(model, i, j, k):
            if k >= len(self.tasks[j].steps):
                # Step index k does not exist for task j.
                return Constraint.Skip

            agent = self.agents[i]
            step = self.tasks[j].steps[k]

            return Constraint.Skip if agent.has_capabilities(step.capabilities) else model.x[i, j, k] == 0

        self.model.capability = Constraint(self.model.A, self.model.T, self.model.S, rule=capability_constraint)

        # ---------------------------------------------------------------------
        # Task precedence as a *selection dependency*
        # If task j1 must precede task j2, we cannot select j2 unless we select j1.
        # ---------------------------------------------------------------------
        self.model.dependency = ConstraintList()
        for j1, j2 in self.precedence_relations:
            self.model.dependency.add(self.model.y[j2] <= self.model.y[j1])

        # ---------------------------------------------------------------------
        # If a task is selected, every step must be allocated to >= 1 agent
        # ---------------------------------------------------------------------
        def task_step_allocation_constraint(model, j, k):
            if k >= len(self.tasks[j].steps):
                # Step index k does not exist for task j.
                return Constraint.Skip
            return sum(model.x[i, j, k] for i in model.A) >= model.y[j]

        self.model.task_step_allocation = Constraint(self.model.T, self.model.S, rule=task_step_allocation_constraint)

        # ---------------------------------------------------------------------
        # If a task is not selected, no steps can be assigned to it
        # ---------------------------------------------------------------------
        def task_full_allocation_constraint(model, j):
            # Big-M upper bound: if y[j] = 0 -> sum x = 0; if y[j] = 1 -> allow assignments.
            m = len(self.agents) * len(self.tasks[j].steps)
            return (
                sum(model.x[i, j, k] for i in model.A for k in range(len(self.tasks[j].steps)))
                <= model.y[j] * m
            )

        self.model.task_full_allocation = Constraint(self.model.T, rule=task_full_allocation_constraint)

        # ---------------------------------------------------------------------
        # Movement budget constraint (approximation)
        #
        # IMPORTANT:
        # The original implementation computed a *full task route* cost and applied it to each x[i,j,k].
        # That is overly conservative when tasks are split across agents or when you deliberately assign
        # multiple agents to the same step (MATCH-AOU redundancy).
        #
        # Here we use a per-agent aggregate travel approximation:
        #   sum_{j,k} travel_cost(i, step_{j,k}) * x[i,j,k] <= budget_i
        #
        # travel_cost(i, step) is estimated from agent.start_location -> step.location.
        # ---------------------------------------------------------------------
        def movement_budget_constraint(model, i):
            agent = self.agents[int(i)]
            start_loc = getattr(agent, "location", None)
            if start_loc is None:
                return Constraint.Skip

            total_cost = 0.0
            for j in model.T:
                steps = self.tasks[int(j)].steps
                for k, step in enumerate(steps):
                    step_loc = getattr(step, "location", None)
                    if step_loc is None:
                        continue
                    total_cost += agent.move_cost(destination=step_loc, source=start_loc) * model.x[i, j, k]

            return total_cost <= agent.budget * (1 - self.risk_factor)

        self.model.movement_budget = Constraint(self.model.A, rule=movement_budget_constraint)

        # ---------------------------------------------------------------------
        # Optional: weapon budget constraint (currently disabled)
        # ---------------------------------------------------------------------
        # def weapon_budget_constraint(model, i):
        #     ...
        # self.model.weapon_budget = Constraint(self.model.A, rule=weapon_budget_constraint)

    def solve(self, solver_name: str = "bonmin"):
        """Solve the model.

        Returns
        -------
        solution : dict | None
            Mapping: agent_id -> list[(task_idx, step_idx)] for assigned steps.
            Returns None if not solved to acceptable optimality.
        results : SolverResults
            Pyomo solver results object.
        unselected_tasks : list[int]
            Task indices with y[j] == 0 (not selected).
        """
        solver = SolverFactory(solver_name)
        results = solver.solve(self.model, tee=False)

        ok_conditions = {
            TerminationCondition.optimal,
            TerminationCondition.locallyOptimal,
        }
        if results.solver.termination_condition not in ok_conditions:
            print("Model not solved to acceptable optimality. Check constraints and inputs.")
            return None, results, []

        # Tasks not selected by the model
        unselected_tasks: list[int] = []
        for j in self.model.T:
            y_val = self.model.y[j].value
            if y_val is None or y_val <= 0.5:
                unselected_tasks.append(int(j))

        # Extract assignment solution (only valid step indices)
        solution: dict = {}
        for i in self.model.A:
            agent_id = self.agents[int(i)].id
            for j in self.model.T:
                j_int = int(j)
                for k in range(len(self.tasks[j_int].steps)):
                    x_val = self.model.x[i, j, k].value
                    if x_val is not None and x_val > 0.5:
                        solution.setdefault(agent_id, []).append((j_int, int(k)))

        return solution, results, unselected_tasks

    def display_solution(self, solution) -> None:
        """Pretty-print the solution."""
        if solution is None:
            print("No solution found or problem is infeasible.")
            return

        print("Assigned Tasks:")
        for agent_id, assignments in solution.items():
            print(f"Agent {agent_id} assigned to steps:")
            for task_id, step_id in assignments:
                print(f"  Task {task_id}, Step {step_id}")

        print("\nUnassigned Tasks:")
        for j in self.model.T:
            if (self.model.y[j].value or 0) < 0.5:
                print(f"Task {j} is unassigned (Utility: {self.tasks[j].utility})")

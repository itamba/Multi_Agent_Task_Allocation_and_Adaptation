import json
import csv
from models import Agent, Task, Step, StepType, Location, Capability


def load_data(file_path):
    """
    Load data from a file (JSON or CSV) and parse it into Agent and Task objects.
    :param file_path: Path to the file.
    :return: Tuple (agents, tasks)
    """
    if file_path.endswith('.json'):
        return load_json(file_path)
    elif file_path.endswith('.csv'):
        return load_csv(file_path)
    else:
        raise ValueError("Unsupported file format. Use JSON or CSV.")


def load_json(file_path):
    """
    Load agents and tasks from a JSON file.
    """
    with open(file_path, 'r') as file:
        data = json.load(file)

    agents = [parse_agent(agent_data) for agent_data in data.get("agents", [])]
    tasks = [parse_task(task_data) for task_data in data.get("tasks", [])]

    return agents, tasks


def load_csv(file_path):
    """
    Load agents and tasks from a CSV file.
    Assumes a column "type" to distinguish between agents and tasks.
    """
    agents, tasks = [], []
    with open(file_path, newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['type'] == 'agent':
                agents.append(parse_agent(row))
            elif row['type'] == 'task':
                tasks.append(parse_task(row))
    return agents, tasks


def parse_agent(data):
    """
    Convert JSON/CSV data to an Agent object.
    """
    location = Location(float(data['x']), float(data['y']), float(data['z']))

    # Handle capabilities as list or semicolon-separated string
    if isinstance(data['capabilities'], list):
        capabilities = [Capability(cap.strip()) for cap in data['capabilities']]
    else:
        capabilities = [Capability(cap.strip()) for cap in data['capabilities'].split(';')]

    budget = float(data['budget'])

    step_cost_functions = {step_type: lambda step: step.compute_step_cost() for step_type in
                           data.get("step_costs", {}).keys()}
    move_cost_function = lambda loc1, loc2: loc1.distance_to(loc2) * float(data.get("move_cost_factor", 1.0))
    return Agent(location, capabilities, budget, step_cost_functions, move_cost_function)


def parse_task(data):
    """
    Convert JSON/CSV data to a Task object.
    """
    steps = [parse_step(step) for step in data.get("steps", [])]
    utility = float(data['utility'])
    return Task(steps, utility)


def parse_step(data):
    """
    Convert JSON data to a Step object.
    """
    location = Location(float(data['x']), float(data['y']), float(data['z'])) if "x" in data else None

    # Handle capabilities as list or semicolon-separated string
    if isinstance(data['capabilities'], list):
        capabilities = [Capability(cap.strip()) for cap in data['capabilities']]
    else:
        capabilities = [Capability(cap.strip()) for cap in data['capabilities'].split(';')]

    step_type = StepType(data['step_type'], float(data['base_cost']))
    effort = float(data['effort'])
    probability = float(data['probability'])
    return Step(location, capabilities, step_type, effort, probability)

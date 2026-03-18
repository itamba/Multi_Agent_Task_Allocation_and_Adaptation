class Capability:
    """
    Represents a single capability, possibly part of an ontology hierarchy.
    """
    def __init__(self, name, properties=None):
        """
        Initialize a capability.
        :param name: Name of the capability (e.g., "HD Camera").
        :param properties: Dictionary of properties (e.g., {"resolution": "1080p", "color": "blue"}).
        """
        self.name = name
        self.properties = properties or {}

    def matches_requirement(self, requirement):
        """
        Check if this capability satisfies a given requirement.
        :param requirement: A dictionary of required properties (e.g., {"color": "blue"}).
        :return: True if all required properties match, False otherwise.
        """
        for key, value in requirement.items():
            if self.properties.get(key) != value:
                return False
        return True

    def __repr__(self):
        return f"Capability(name={self.name}, properties={self.properties})"

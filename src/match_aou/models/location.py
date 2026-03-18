try:
    from haversine import haversine
except ImportError:
    # Fallback haversine implementation
    import math
    def haversine(coord1, coord2):
        """Calculate distance between two (lat, lon) points in km."""
        lat1, lon1 = coord1
        lat2, lon2 = coord2
        R = 6371  # Earth radius in km
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (math.sin(dlat/2)**2 + 
             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * 
             math.sin(dlon/2)**2)
        c = 2 * math.asin(math.sqrt(a))
        return R * c

class Location:
    """
    Represents a geographical location with latitude, longitude and optional altitude.
    """
    def __init__(self, latitude, longitude, altitude=0):
        """
        Initialize a location.
        :param latitude: Latitude of the location in degrees.
        :param longitude: Longitude of the location in degrees.
        :param altitude: Altitude in meters or feet (optional).
        """
        self.latitude = latitude
        self.longitude = longitude
        self.altitude = altitude

    def distance_to(self, other_location):
        """
        Calculate the horizontal distance to another location using haversine formula.
        Altitude is not considered in this calculation.

        :param other_location: Another Location object.
        :return: Distance in kilometers.
        """

        loc1 = (self.latitude, self.longitude)
        loc2 = (other_location.latitude, other_location.longitude)
        return haversine(loc1, loc2)

    def __repr__(self):
        return f"Location(latitude={self.latitude}, longitude={self.longitude}, altitude={self.altitude})"

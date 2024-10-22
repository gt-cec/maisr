import sys
import time
import math
from SimConnect import *
from datetime import datetime


class MSFSConnector:
    def __init__(self):
        # Initialize SimConnect
        try:
            self.sm = SimConnect()
            self.aq = AircraftRequests(self.sm, _time=0)
            self.ae = AircraftEvents(self.sm)
            print("Successfully connected to Microsoft Flight Simulator")
        except:
            print("Unable to connect to Microsoft Flight Simulator")
            sys.exit(1)

        # Initialize AI aircraft
        self.ai_aircraft = self.sm.make_request_object()
        self.ai_aircraft.sendId = 0  # ID for the AI aircraft

        # Set up aircraft event handlers
        self.freeze = self.ae.find("FREEZE_LATITUDE_LONGITUDE_SET")
        self.pause = self.ae.find("PAUSE_SET")

        # Create requests for aircraft data
        self.requests = {
            "latitude": self.aq.find("PLANE_LATITUDE"),
            "longitude": self.aq.find("PLANE_LONGITUDE"),
            "altitude": self.aq.find("PLANE_ALTITUDE"),
            "heading": self.aq.find("PLANE_HEADING_DEGREES_TRUE"),
            "airspeed": self.aq.find("AIRSPEED_INDICATED"),
            "ai_latitude": None,
            "ai_longitude": None,
            "ai_altitude": None,
            "ai_heading": None
        }

    def convert_game_to_gps(self, x, y, game_size):
        """Convert game coordinates to GPS coordinates
        Assumes game coordinates are in a square centered on a reference point"""
        # Define reference point (adjust these for your desired location)
        ref_lat = 47.6062  # Seattle latitude
        ref_lon = -122.3321  # Seattle longitude

        # Calculate relative position from center
        center = game_size / 2
        rel_x = x - center
        rel_y = center - y  # Invert Y because game coordinates are top-down

        # Convert game units to degrees (adjust scale factor as needed)
        scale = 0.001  # This determines how large the game area is in MSFS
        delta_lon = rel_x * scale
        delta_lat = rel_y * scale

        return ref_lat + delta_lat, ref_lon + delta_lon

    def convert_gps_to_game(self, lat, lon, game_size):
        """Convert GPS coordinates back to game coordinates"""
        ref_lat = 47.6062
        ref_lon = -122.3321
        scale = 0.001

        delta_lat = lat - ref_lat
        delta_lon = lon - ref_lon

        center = game_size / 2
        x = center + (delta_lon / scale)
        y = center - (delta_lat / scale)  # Invert Y coordinate

        return x, y

    def update_ai_aircraft(self, x, y, heading, game_size, altitude=1000):
        """Update AI aircraft position based on game coordinates"""
        lat, lon = self.convert_game_to_gps(x, y, game_size)

        # Set AI aircraft position
        self.sm.set_aircraft_position(
            self.ai_aircraft.sendId,
            lat,  # latitude
            lon,  # longitude
            altitude,  # altitude in feet
            heading,  # heading in degrees
            0,  # pitch (level)
            0  # bank (level)
        )

    def get_player_position(self, game_size):
        """Get player aircraft position in game coordinates"""
        lat = self.requests["latitude"].get()
        lon = self.requests["longitude"].get()
        heading = self.requests["heading"].get()

        if lat is not None and lon is not None:
            x, y = self.convert_gps_to_game(lat, lon, game_size)
            return x, y, heading
        return None, None, None

    def spawn_ai_aircraft(self):
        """Spawn the AI aircraft in MSFS"""
        try:
            # Remove any existing AI aircraft
            self.sm.remove_aircraft(self.ai_aircraft.sendId)
            time.sleep(0.1)

            # Create new AI aircraft
            self.sm.spawn_aircraft(
                "AI",  # Title
                "TT:ATCCOM.AC_MODEL_C172",  # Aircraft type (Cessna 172)
                "AI Aircraft",  # Tail number
                0,  # Aircraft ID
                47.6062,  # Initial latitude
                -122.3321,  # Initial longitude
                1000,  # Initial altitude
                0  # Initial heading
            )
            print("AI aircraft spawned successfully")
            return True
        except:
            print("Failed to spawn AI aircraft")
            return False

    def cleanup(self):
        """Clean up SimConnect resources"""
        try:
            self.sm.remove_aircraft(self.ai_aircraft.sendId)
            self.sm.exit()
            print("Successfully cleaned up MSFS connection")
        except:
            print("Error cleaning up MSFS connection")
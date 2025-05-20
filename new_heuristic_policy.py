import math
import numpy as np


class AutonomousPolicy:
    def __init__(self, env, aircraft_id):
        """
        Initialize the autonomous policy for a given environment and aircraft

        Args:
            env: MAISR environment instance
            aircraft_id: ID of the aircraft controlled by this policy
        """
        self.env = env
        self.aircraft_id = aircraft_id

        self.target_point = (0, 0)
        self.action = None

        # For displaying on agent status window
        self.low_level_rationale = ''  # Identify unknown target, identify unknown WEZ, or evade threat
        self.high_level_rationale = ''  # Why agent is doing what it's doing at high level
        self.quadrant_rationale = ''  # Why agent is searching in current quadrant
        self.nearby_threats = []  # List of up to 2 nearest threats
        self.three_upcoming_targets = []  # Target IDs of nearest unknown targets
        self.current_target_distance = 0  # Distance to current target if pursuing one

        self.risk_level = 'LOW'

        # Agent priorities to follow during search (can be overridden by human)
        self.search_quadrant = 'full'  # Auto-selected by policy unless search_quadrant_override is not 'none'
        self.search_type = 'target'  # Auto-selected by policy unless search_type_override is not 'none'

        # Human overrides for agent priorities
        self.search_quadrant_override = 'none'  # 'none' by default, NW/SW/NE/SE/full if human clicks a quadrant
        self.search_type_override = 'none'  # 'target' or 'wez' if human clicks buttons
        self.hold_commanded = False  # If hold button is clicked, goes to true
        self.waypoint_override = False  # If human sets a specific waypoint

        # UI options
        self.show_low_level_goals = True
        self.show_high_level_goals = True
        self.show_high_level_rationale = True
        self.show_tracked_factors = True

        # Update frequency control
        self.ticks_since_update = 0
        self.update_rate = 20

        # Previous state tracking
        self.previous_nearest_distance = None

    def act(self):
        """
        Determine and return the next action for the agent

        Returns:
            tuple: (x, y) coordinates, normalized to [-1, 1]
        """
        self.aircraft = self.env.agents[self.aircraft_id]

        if self.ticks_since_update > self.update_rate:
            self.ticks_since_update = 0
            self.calculate_risk_level()

            if self.hold_commanded:
                self.target_point = self.hold_policy()
                self.low_level_rationale = 'Holding position'
                self.high_level_rationale = '(Human command)'

            elif self.waypoint_override:
                self.target_point = self.human_waypoint(self.waypoint_override)
                self.low_level_rationale = 'Moving to designated waypoint'
                self.high_level_rationale = '(Human command)'

            else:
                self.calculate_priorities()

                target_point, closest_distance = self.basic_search()

                # Convert to normalized coordinates expected by the environment
                # First normalize to [0, 1]
                target_point_normalized = (
                    target_point[0] / self.env.config["gameboard_size"],
                    target_point[1] / self.env.config["gameboard_size"]
                )

                # Then scale to [-1, 1] for the continuous-normalized action space
                if self.env.action_type == 'continuous-normalized':
                    self.target_point = (
                        2 * target_point_normalized[0] - 1,
                        2 * target_point_normalized[1] - 1
                    )
                else:
                    self.target_point = target_point_normalized

                # Update info for display
                self.current_target_distance = closest_distance
                if closest_distance:
                    self.low_level_rationale = f'Identifying target {int(closest_distance)} units away'
                else:
                    self.low_level_rationale = 'Searching for unknown targets'

                if self.search_type_override != 'none':
                    self.high_level_rationale = '(Human command)'

            self.update_agent_info()
        else:
            self.ticks_since_update += 1

        # Return normalized action for the environment
        self.action = (self.target_point[0], self.target_point[1])
        return self.action

    def basic_search(self):
        """
        Find the nearest target that matches the current search criteria

        Returns:
            tuple: ((x, y) waypoint coordinates, distance to nearest target)
        """
        current_target_distances = {}  # Will be {target_idx: distance}
        closest_distance = None

        gameboard_size = self.env.config["gameboard_size"]
        quadrant_bounds = {
            'full': (0, gameboard_size, 0, gameboard_size),
            'NW': (0, gameboard_size * 0.5, 0, gameboard_size * 0.5),
            'NE': (gameboard_size * 0.5, gameboard_size, 0, gameboard_size * 0.5),
            'SW': (0, gameboard_size * 0.5, gameboard_size * 0.5, gameboard_size),
            'SE': (gameboard_size * 0.5, gameboard_size, gameboard_size * 0.5, gameboard_size)
        }  # specifies (Min x, max x, min y, max y)

        # Find nearby threats for status display
        self.nearby_threats = []
        threat_distances = {}

        # Process targets based on search objectives
        for target_idx in range(self.env.num_targets):
            target = self.env.targets[target_idx]
            target_id = int(target[0])
            target_value = target[1]  # 0 = regular, 1 = high-value
            info_level = target[2]  # 0 = unknown, 0.5 = low-quality, 1.0 = high-quality
            target_x = float(target[3])
            target_y = float(target[4])

            # Track high-value targets (potential threats) for display
            if info_level > 0 and target_value == 1.0:
                dist = math.hypot(self.aircraft.x - target_x, self.aircraft.y - target_y)
                threat_distances[target_id] = dist

            # Check if target is in current search quadrant
            in_quadrant = (
                    quadrant_bounds[self.search_quadrant][0] <= target_x <= quadrant_bounds[self.search_quadrant][1] and
                    quadrant_bounds[self.search_quadrant][2] <= target_y <= quadrant_bounds[self.search_quadrant][3]
            )

            if not in_quadrant:
                continue

            if self.search_type == 'target':
                # If set to target ID, only consider unknown targets (info_level = 0)
                if info_level == 0:
                    dist = math.hypot(self.aircraft.x - target_x, self.aircraft.y - target_y)
                    current_target_distances[target_id] = dist

                    if closest_distance is None or dist < closest_distance:
                        closest_distance = dist

            elif self.search_type == 'wez':
                # If set to WEZ ID, consider both unknown targets and targets with only low-quality info
                if info_level < 1.0:  # Either unknown (0) or low-quality (0.5)
                    dist = math.hypot(self.aircraft.x - target_x, self.aircraft.y - target_y)
                    current_target_distances[target_id] = dist

                    if closest_distance is None or dist < closest_distance:
                        closest_distance = dist

        # Get 2 closest threats for status display
        sorted_threats = sorted(threat_distances.items(), key=lambda x: x[1])
        for threat_id, dist in sorted_threats[:2]:
            self.nearby_threats.append((threat_id, int(dist)))

        if current_target_distances:  # If there are targets nearby, set waypoint to nearest one
            nearest_target_id = min(current_target_distances, key=current_target_distances.get)

            # Find the target with this ID in the targets array
            for target_idx in range(self.env.num_targets):
                if self.env.targets[target_idx][0] == nearest_target_id:
                    target_waypoint = (float(self.env.targets[target_idx][3]), float(self.env.targets[target_idx][4]))
                    break
            else:
                # Fallback if target not found (shouldn't happen)
                target_waypoint = self.get_default_waypoint(self.search_quadrant, gameboard_size)

        else:  # If all targets ID'd, loiter in center of board or specified quadrant
            target_waypoint = self.get_default_waypoint(self.search_quadrant, gameboard_size)

        return target_waypoint, closest_distance

    def get_default_waypoint(self, quadrant, gameboard_size):
        """
        Helper method to get default waypoint positions by quadrant

        Args:
            quadrant: Quadrant name (NW, NE, SW, SE, full)
            gameboard_size: Size of the game board

        Returns:
            tuple: (x, y) coordinates for the center of the specified quadrant
        """
        quadrant_centers = {
            'full': (gameboard_size * 0.5, gameboard_size * 0.5),
            'NW': (gameboard_size * 0.25, gameboard_size * 0.25),
            'NE': (gameboard_size * 0.75, gameboard_size * 0.25),
            'SW': (gameboard_size * 0.25, gameboard_size * 0.75),
            'SE': (gameboard_size * 0.75, gameboard_size * 0.75)
        }
        return quadrant_centers[quadrant]

    def hold_policy(self):
        """
        Maintain current position

        Returns:
            tuple: Current (x, y) position normalized to [-1, 1]
        """
        current_pos = (self.aircraft.x, self.aircraft.y)
        # Normalize to [0, 1]
        pos_normalized = (
            current_pos[0] / self.env.config["gameboard_size"],
            current_pos[1] / self.env.config["gameboard_size"]
        )
        # Scale to [-1, 1]
        if self.env.action_type == 'continuous-normalized':
            return (2 * pos_normalized[0] - 1, 2 * pos_normalized[1] - 1)
        return pos_normalized

    def human_waypoint(self, waypoint_position):
        """
        Move to a human-specified waypoint

        Args:
            waypoint_position: (x, y) coordinates of the waypoint

        Returns:
            tuple: Waypoint position normalized for the action space
        """
        waypoint_threshold = 5
        target_waypoint = self.waypoint_override

        # Check if we've reached the waypoint
        dist_to_waypoint = math.sqrt(
            (target_waypoint[0] - self.aircraft.x) ** 2 +
            (target_waypoint[1] - self.aircraft.y) ** 2
        )

        if dist_to_waypoint <= waypoint_threshold:
            self.waypoint_override = False

        # Convert to normalized format
        pos_normalized = (
            target_waypoint[0] / self.env.config["gameboard_size"],
            target_waypoint[1] / self.env.config["gameboard_size"]
        )

        # Scale to [-1, 1] if needed
        if self.env.action_type == 'continuous-normalized':
            return (2 * pos_normalized[0] - 1, 2 * pos_normalized[1] - 1)
        return pos_normalized

    def update_agent_info(self):
        """Update the agent info display with current status"""
        if not hasattr(self.env, 'agent_info_display'):
            return

        # Format nearby threats text
        threats_text = "None detected"
        if self.nearby_threats:
            threats_text = ", ".join([f"{t[1]} units" for t in self.nearby_threats])

        # Build status text with conditional sections
        self.status_lines = []

        # Low level goals
        if self.show_low_level_goals:
            self.status_lines.append(f"CURRENT GOAL: {self.low_level_rationale}")

        # High level goals
        if self.show_high_level_goals:
            if self.search_type:
                if self.show_high_level_rationale:
                    self.status_lines.append(f"SEARCH TYPE: {self.search_type.upper()} {self.high_level_rationale}")
                else:
                    self.status_lines.append(f"SEARCH TYPE: {self.search_type.upper()}")
            if self.search_quadrant:
                if self.show_high_level_rationale:
                    self.status_lines.append(f"SEARCH AREA: {self.search_quadrant.upper()} {self.quadrant_rationale}")
                else:
                    self.status_lines.append(f"SEARCH AREA: {self.search_quadrant.upper()}")

        # Tracked factors
        if self.show_tracked_factors:
            self.status_lines.append(f"RISK LEVEL: {self.risk_level}")
            self.status_lines.append(f"NEARBY THREATS: {threats_text}")

        # Update the display
        if hasattr(self.env.agent_info_display, 'update_text'):
            self.env.agent_info_display.update_text(self.status_lines)

    def calculate_risk_level(self):
        """Calculates how risky the current situation is, as a function of agent health and number of nearby threats"""
        # Find high-value targets within a detection radius
        detection_radius = 100  # Adjust as needed
        hostile_nearby = 0

        for target_idx in range(self.env.num_targets):
            target = self.env.targets[target_idx]
            target_value = target[1]  # 0 = regular, 1 = high-value
            info_level = target[2]  # Information level
            target_x, target_y = float(target[3]), float(target[4])

            # Count nearby high-value targets as threats
            if target_value == 1.0:
                dist = math.hypot(self.aircraft.x - target_x, self.aircraft.y - target_y)
                if dist <= detection_radius:
                    hostile_nearby += 1

        # Get agent health
        agent_health = self.aircraft.health_points

        # Calculate risk level based on threats and health
        risk_factor = hostile_nearby * 10 - agent_health

        if risk_factor <= 0:
            self.risk_level = 'LOW'
        elif risk_factor <= 5:
            self.risk_level = 'MEDIUM'
        elif risk_factor <= 10:
            self.risk_level = 'HIGH'
        else:
            self.risk_level = 'EXTREME'

    def calculate_priorities(self):
        """Set search type (target or wez) and search quadrant based on the current situation"""
        # Set search_type
        if self.search_type_override != 'none':
            # If player has set a specific search type, use that
            self.search_type = self.search_type_override
            self.high_level_rationale = '(Human command)'
        else:
            # Choose according to current situation
            time_remaining = self.env.time_limit - self.env.display_time / 1000

            if time_remaining <= 30:
                self.search_type = 'wez'
                self.high_level_rationale = '(Critical time remaining)'
            elif self.aircraft.health_points > 5:
                self.search_type = 'wez'
                self.high_level_rationale = ''
            else:
                self.search_type = 'target'
                if self.aircraft.health_points <= 5:
                    self.high_level_rationale = '(Low health)'
                else:
                    self.high_level_rationale = ''

        # Set quadrant to search in
        quadrants = self.calculate_quadrant_densities()

        if self.search_quadrant_override != 'none':
            self.search_quadrant = self.search_quadrant_override
            self.quadrant_rationale = '(Human command)'
        elif quadrants and len(quadrants) >= 2 and quadrants[0][1] >= quadrants[1][1] + 3:
            # If the densest quadrant has significantly more targets than the second,
            # prioritize that quadrant
            self.search_quadrant = quadrants[0][0]
            self.quadrant_rationale = f'(Significant target grouping)'
        else:
            self.search_quadrant = 'full'
            self.quadrant_rationale = ''

    def calculate_quadrant_densities(self):
        """
        Calculate the number of unknown targets in each quadrant

        Returns:
            list: [(quadrant_name, count)] sorted by count in descending order
        """
        gameboard_size = self.env.config["gameboard_size"]

        # Calculate how many unknown targets are in each quadrant
        quadrant_counts = {'NW': 0, 'NE': 0, 'SW': 0, 'SE': 0, 'full': 0}

        for target_idx in range(self.env.num_targets):
            target = self.env.targets[target_idx]
            info_level = target[2]  # 0 = unknown
            target_x, target_y = float(target[3]), float(target[4])

            # Only count unknown targets
            if info_level == 0:
                # Add to total count
                quadrant_counts['full'] += 1

                # Determine which quadrant the target is in
                if target_x <= gameboard_size * 0.5 and target_y <= gameboard_size * 0.5:
                    quadrant_counts['NW'] += 1
                elif target_x <= gameboard_size * 0.5 and target_y > gameboard_size * 0.5:
                    quadrant_counts['SW'] += 1
                elif target_x > gameboard_size * 0.5 and target_y <= gameboard_size * 0.5:
                    quadrant_counts['NE'] += 1
                else:  # target_x > gameboard_size * 0.5 and target_y > gameboard_size * 0.5
                    quadrant_counts['SE'] += 1

        # Sort quadrants by number of unknown targets (descending)
        sorted_quadrants = sorted(quadrant_counts.items(), key=lambda x: x[1], reverse=True)
        return sorted_quadrants
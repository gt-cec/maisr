#Agent0 always runs autonomous policy.Buttons force specific constraints to behavior

import math


'''
In main.py, will need to hook up the buttons to set autonomous_policy.gameplan, .risk_tolerance, .search_quadrant, .target_priorities
'''

"""info['decision']['constraints'].append(f'High target density in {quadrant} quadrant')

elif self.risk_level == low or self.risk_level == medium:
aggressive_policy():
high_level_rationale = 'Prioritize targets'

elif self.risk_level == high or self.risk_level == extreme:
cautious_policy():
high_level_rationale = 'Preserve health'"""

# TODO:
#   Add risk tolerance buttons

class AutonomousPolicy:
    def __init__(self,env,aircraft_id):

        self.env = env
        #self.aircraft = env.agents[aircraft_id]
        self.aircraft_id = aircraft_id

        self.target_point = (0,0)

        # For displaying on agent status window
        self.low_level_rationale = ''  # Identify unknown target, identify unknown WEZ, or evade threat
        self.high_level_rationale = ''  # Why agent is doing what it's doing at high level
        self.quadrant_rationale = ''  # Why agent is searching in current quadrant
        self.nearby_threats = []  # List of up to 2 nearest threats
        self.three_upcoming_targets = []  # Target IDs of nearest unknown targets
        self.current_target_distance = 0  # Distance to current target if pursuing one

        self.risk_level = ''

        # Agent priorities to follow during search (can be overridden by human)
        self.search_quadrant = '' # Auto-selected by policy unless search_quadrant_override is not 'none'
        self.search_type = '' # Auto-selected by policy unless search_type_override is not 'none'
        self.collision_ok = False  # If False, collision avoidance executes normally.
        self.use_thread_the_needle = False

        # Human overrides for agent priorities
        self.risk_tolerance = 'medium'  # Override to low/high based on button clicks
        self.search_quadrant_override = 'none'  # 'none' by default, NW/SW/NE/SE/full if human clicks a quadrant. Resets back to auto if autonomous button clicked
        self.search_type_override = 'none'  # 'target' or 'wez' if human clicks buttons. Resets to auto if auto button clicked
        self.hold_commanded = False # If hold button is clicked, goes to true
        self.waypoint_override = False

        self.show_low_level_goals = True
        self.show_high_level_goals = True
        self.show_high_level_rationale = True
        self.show_tracked_factors = True


    def act(self):  # Execute policy based on chosen gameplan
        #print('act running')
        self.aircraft = self.env.agents[self.aircraft_id]
        self.calculate_risk_level()

        if self.hold_commanded:
            self.target_point = self.hold_policy()
            self.low_level_rationale = 'Holding position'
            self.high_level_rationale = 'Following human command'

        elif self.waypoint_override != False:
            # Execute the waypoint
            self.target_point = self.human_waypoint(self.waypoint_override)


        elif self.upcoming_collision() and not self.collision_ok:
            print('collison detected')
            self.target_point = self.collision_avoidance()
            self.low_level_rationale = 'Evade threat'
            self.high_level_rationale = 'Preserve health'

        elif self.use_thread_the_needle:
            self.calculate_priorities()
            scores = self.thread_the_needle()
            if scores:
                self.target_point = max(scores.items(), key=lambda x: x[1])[0]

        else:
            self.calculate_priorities()
            self.target_point, closest_target_distance = self.basic_search()

            # Populate agent info
            self.current_target_distance = closest_target_distance
            if closest_target_distance: self.low_level_rationale = f'Identifying target {int(closest_target_distance)} units away'
            else: self.low_level_rationale = f'Scanning for targets'
            if self.search_type_override != 'none':
                self.high_level_rationale = 'Following human command'
            elif self.search_type == 'target': self.high_level_rationale = 'Preserve health'
            else: self.high_level_rationale = 'Prioritize mission'
            #print('Target point set to %s' % (self.target_point,))

        self.update_agent_info()


    def basic_search(self):
        # TODO: Doesn't seem to route to the closest unknown target in the specified quadrant
        current_state = self.env.get_state()
        current_target_distances = {}  # Will be {agent_idx:distance}
        closest_distance = None

        gameboard_size = self.env.config["gameboard size"]
        quadrant_bounds = {'full': (0, gameboard_size, 0, gameboard_size),'NW': (0, gameboard_size * 0.5, 0, gameboard_size * 0.5),'NE': (gameboard_size * 0.5, gameboard_size, 0, gameboard_size * 0.5),'SW': (0, gameboard_size * 0.5, gameboard_size * 0.5, gameboard_size), 'SE': (gameboard_size * 0.5, gameboard_size, gameboard_size * 0.5,gameboard_size)}  # specifies (Min x, max x, min y, max y)

        # Find nearby threats for status display
        self.nearby_threats = []
        threat_distances = {}
        for ship_id in current_state['ships']:
            ship = current_state['ships'][ship_id]
            if ship['observed'] and ship['observed threat'] and ship['threat'] > 0:
                dist = math.hypot(self.aircraft.x - self.env.agents[ship_id].x,
                                  self.aircraft.y - self.env.agents[ship_id].y)
                threat_distances[ship_id] = dist

        # Get 2 closest threats
        sorted_threats = sorted(threat_distances.items(), key=lambda x: x[1])
        for threat_id, dist in sorted_threats[:2]:
            self.nearby_threats.append((threat_id, int(dist)))

        for ship_id in current_state['ships']: # Loop through all ships in environment, calculate distance, find closest unknown ship (or unknown WEZ), and set waypoint to that location
            if self.search_type == 'target':  # If set to target, only consider unknown targets
                if current_state['ships'][ship_id]['observed'] == False and (
                        quadrant_bounds[self.search_quadrant][0] <=  current_state['ships'][ship_id]['position'][0] <=
                        quadrant_bounds[self.search_quadrant][1]) and (quadrant_bounds[self.search_quadrant][2] <=  current_state['ships'][ship_id]['position'][1] <=quadrant_bounds[self.search_quadrant][3]):
                    dist = math.hypot(self.aircraft.x - self.env.agents[ship_id].x,self.aircraft.y - self.env.agents[ship_id].x)
                    current_target_distances[ship_id] = dist
                if closest_distance is None or dist < closest_distance:
                    closest_distance = dist

            elif self.search_type == 'wez':  # If set to wez, consider unknown targets AND known hostiles with unknown threat rings
                if (current_state['ships'][ship_id]['observed'] == False or current_state['ships'][ship_id]['observed threat'] == False) and (quadrant_bounds[self.search_quadrant][0] <= current_state['ships'][ship_id]['position'][0] <=quadrant_bounds[self.search_quadrant][1]) and (quadrant_bounds[self.search_quadrant][2] <= current_state['ships'][ship_id]['position'][1] <=quadrant_bounds[self.search_quadrant][3]):
                    dist = math.hypot(self.aircraft.x - self.env.agents[ship_id].x,self.aircraft.y - self.env.agents[ship_id].y)
                    current_target_distances[ship_id] = dist
                    if closest_distance is None or dist < closest_distance:
                        closest_distance = dist
        #print('current target distances:')
        #print(current_target_distances)

        if current_target_distances: # If there are targets nearby, set waypoint to the nearest one
            self.three_upcoming_targets = [item[0] for item in sorted(current_target_distances.items(), key=lambda x: x[1], reverse=True)[:3]] # Store the three closest target IDs for displaying later
            nearest_target_id = min(current_target_distances, key=current_target_distances.get)
            target_waypoint = tuple((self.env.agents[nearest_target_id].x, self.env.agents[nearest_target_id].y))

        else:  # If all targets ID'd, loiter in center of board or specified quadrant
            quadrant_centers = {'full': (gameboard_size * 0.5,gameboard_size * 0.5), 'NW': (gameboard_size * 0.25, gameboard_size * 0.25), 'NE':(gameboard_size * 0.75, gameboard_size * 0.25), 'SW':(gameboard_size * 0.25, gameboard_size * 0.75),'SE':(gameboard_size * 0.75, gameboard_size * 0.75)}
            target_waypoint =  quadrant_centers[self.search_quadrant]

        target_direction = math.atan2(target_waypoint[1] - self.aircraft.y,target_waypoint[0] - self.aircraft.x)
        return target_waypoint, closest_distance


    def collision_avoidance(self):
        """
        Modifies waypoint to avoid detected collision threats.
        Returns a new waypoint that routes around the closest threat.
        """
        aircraft = self.env.agents[self.aircraft_id]

        # If no target point, can't calculate avoidance
        if not self.target_point:
            return (aircraft.x, aircraft.y)

        # Calculate direction vector to target
        dx = self.target_point[0] - aircraft.x
        dy = self.target_point[1] - aircraft.y
        path_length = math.sqrt(dx * dx + dy * dy)

        if path_length == 0:
            return (aircraft.x, aircraft.y)

        # Normalize direction vector
        dx = dx / path_length
        dy = dy / path_length

        # Find closest threatening ship along path
        closest_ship = None
        closest_distance = float('inf')
        closest_projection = 0

        current_state = self.env.get_state()
        for ship_id in current_state['ships']:
            ship = self.env.agents[ship_id]

            # Only care about observed hostile ships
            if not (ship.observed and ship.observed_threat and ship.threat > 0):
                continue

            # Vector from aircraft to ship
            to_ship_x = ship.x - aircraft.x
            to_ship_y = ship.y - aircraft.y

            # Project ship position onto agent's path
            dot_product = (to_ship_x * dx + to_ship_y * dy)

            # If the projection is negative or beyond target, skip
            if dot_product < 0 or dot_product > path_length:
                continue

            # Find closest point on path to ship
            closest_x = aircraft.x + dx * dot_product
            closest_y = aircraft.y + dy * dot_product

            # Calculate perpendicular distance from ship to path
            distance = math.sqrt(
                (ship.x - closest_x) ** 2 +
                (ship.y - closest_y) ** 2
            )

            # Update closest ship if this one is closer
            if distance < closest_distance:
                closest_ship = ship
                closest_distance = distance
                closest_projection = dot_product

        if closest_ship is None:
            return self.target_point

        # Calculate deflection point
        # Find perpendicular vector to path (rotate 90 degrees)
        perp_dx = -dy
        perp_dy = dx

        # Determine which side to deflect to
        ship_side = (closest_ship.x - aircraft.x) * perp_dx + (closest_ship.y - aircraft.y) * perp_dy

        # Deflect in opposite direction of ship
        deflection_distance = 100  # pixels to deflect
        if ship_side > 0:
            deflection_dx = -perp_dx * deflection_distance
            deflection_dy = -perp_dy * deflection_distance
        else:
            deflection_dx = perp_dx * deflection_distance
            deflection_dy = perp_dy * deflection_distance

        # Calculate deflection point at 70% of the distance to the closest ship
        deflection_point_x = aircraft.x + dx * (closest_projection * 2) + deflection_dx
        deflection_point_y = aircraft.y + dy * (closest_projection * 2) + deflection_dy

        # Ensure deflection point stays within game boundaries
        margin = self.env.config['gameboard border margin']
        board_size = self.env.config['gameboard size']
        deflection_point_x = max(margin, min(board_size - margin, deflection_point_x))
        deflection_point_y = max(margin, min(board_size - margin, deflection_point_y))

        return (deflection_point_x, deflection_point_y)


    def hold_policy(self):
        print('Commanding hold')
        target_waypoint = self.aircraft.x, self.aircraft.y
        print(target_waypoint)
        target_direction = math.atan2(target_waypoint[1] - self.aircraft.y, target_waypoint[0] - self.aircraft.x)
        return target_waypoint

    def human_waypoint(self,waypoint_position):
        # Set the waypoint to waypoint_position and maintain until complete
        waypoint_threshold = 5
        target_waypoint = self.waypoint_override

        dist_to_waypoint = math.sqrt((target_waypoint[0] - self.aircraft.x) ** 2 +(target_waypoint[1] - self.aircraft.y) ** 2)
        if dist_to_waypoint <= waypoint_threshold:
            self.waypoint_override = False
        return target_waypoint


    # Push various types of information about the agent's decision-making to the AGENT STATUS window, if set.
    def update_agent_info(self):
        """Update the agent info display with current status"""
        if not hasattr(self.env, 'agent_info_display'):
            return

        # Format nearby threats text
        threats_text = "None detected"
        if self.nearby_threats:
            threats_text = ", ".join([f"Ship {t[0]} ({t[1]} units)" for t in self.nearby_threats])

        # Build status text with conditional sections
        self.status_lines = []

        # Low level goals
        if self.show_low_level_goals:
            self.status_lines.append(f"CURRENT GOAL: {self.low_level_rationale}")

        # High level goals
        if self.show_high_level_goals:
            if self.search_type:
                self.status_lines.append(f"SEARCH TYPE: {self.search_type.upper()}")
            if self.search_quadrant:
                self.status_lines.append(f"SEARCH AREA: {self.search_quadrant.upper()}")

        # High level rationale
        if self.show_high_level_rationale:
            self.status_lines.append(f"RATIONALE: {self.high_level_rationale}")
            self.status_lines.append(f"AREA RATIONALE: {self.quadrant_rationale}")

        # Tracked factors
        if self.show_tracked_factors:
            self.status_lines.append(f"RISK LEVEL: {self.risk_level}")
            self.status_lines.append(f"NEARBY THREATS: {threats_text}")

        # Build status text
        """self.status_lines = [
            f"CURRENT GOAL: {self.low_level_rationale}",
            f"SEARCH TYPE: {self.search_type.upper() if self.search_type else 'None'}",
            f"TYPE RATIONALE: {'Following human command' if self.search_type_override != 'none' else self.high_level_rationale}",
            f"SEARCH AREA: {self.search_quadrant.upper() if self.search_quadrant else 'None'}",
            f"AREA RATIONALE: {self.quadrant_rationale}",
            f"RISK LEVEL: {self.risk_level}",
            f"NEARBY THREATS: {threats_text}"
        ]"""

        # Update the display
        self.env.agent_info_display.text = self.status_lines
        self.env.agent_info_display.update_text(self.status_lines)


    # Helper functions to determine behavior:
    # TODO write
    def upcoming_collision(self):
        """
        Check if there is a hostile target directly along agent's trajectory.
        Returns True if a hostile ship is within 20 pixels of the agent's current path.
        """
        aircraft = self.env.agents[self.aircraft_id]

        # If we don't have a valid target point, no collision possible
        if not hasattr(self, 'target_point') or self.target_point is None:
            return False

        # If target point is same as current position, no collision possible
        if self.target_point == (aircraft.x, aircraft.y):
            return False

        # Calculate direction vector to target
        dx = self.target_point[0] - aircraft.x
        dy = self.target_point[1] - aircraft.y
        path_length = math.sqrt(dx * dx + dy * dy)

        if path_length == 0:
            return False

        # Normalize direction vector
        dx = dx / path_length
        dy = dy / path_length

        # Check each ship
        current_state = self.env.get_state()
        for ship_id in current_state['ships']:
            ship = self.env.agents[ship_id]

            # Only care about observed hostile ships
            if not (ship.observed and ship.observed_threat and ship.threat > 0):
                continue

            # Vector from aircraft to ship
            to_ship_x = ship.x - aircraft.x
            to_ship_y = ship.y - aircraft.y

            # Project ship position onto agent's path to find closest point
            dot_product = (to_ship_x * dx + to_ship_y * dy)

            # If the projection is negative, ship is behind aircraft
            if dot_product < 0:
                continue

            # If projection is longer than path to target, ship is beyond target
            if dot_product > path_length:
                continue

            # Find closest point on path to ship
            closest_x = aircraft.x + dx * dot_product
            closest_y = aircraft.y + dy * dot_product

            # Calculate perpendicular distance from ship to path
            distance = math.sqrt(
                (ship.x - closest_x) ** 2 +
                (ship.y - closest_y) ** 2
            )

            # If within 40 pixels, consider it a collision risk
            if distance < 40:
                return True

        return False


    def calculate_risk_level(self): # Calculates how risky the current situation is, as a function of agent health and number of nearby hostile targets

        hostile_targets_nearby = sum(1 for agent in self.env.agents
                                     if agent.agent_class == "ship" and agent.threat > 0 and agent.observed_threat and math.hypot(agent.x - self.aircraft.x, agent.y - self.aircraft.y) <= 30)

        risk_level_function = 10 * hostile_targets_nearby + self.env.agents[self.env.num_ships].damage
        self.risk_level = 'LOW' if risk_level_function <= 30 else 'MEDIUM' if risk_level_function <= 60 else 'HIGH' if risk_level_function <= 80 else 'EXTREME'


    def calculate_priorities(self): # Set search type (target or wez), search quadrant (One of the 4 quadrants or the full board), and whether to avoid collisions

        # Set search_type
        if self.search_type_override != 'none': # If player has set a specific search type, use that
            self.search_type = self.search_type_override
        else: # Choose according to selected risk tolerance
            if self.risk_tolerance == 'low': self.search_type = 'target'

            elif self.risk_tolerance == 'medium':
                if self.aircraft.damage <= 50: self.search_type = 'wez'
                else: self.search_type = 'target'

            elif self.risk_tolerance == 'high': self.search_type = 'wez'

        # Set quadrant to search in
        # If the densest quadrant is substantially denser than the second most dense, prioritize that quadrant. Otherwise do not prioritize a quadrant.
        quadrants = self.calculate_quadrant_densities()
        if self.search_quadrant_override != 'none':
            self.search_quadrant = self.search_quadrant_override
            self.quadrant_rationale = 'Following human command'
        elif quadrants[0][1] >= quadrants[1][1] + 7:
            self.search_quadrant = quadrants[0][0]
            self.quadrant_rationale = f'Significant target grouping in {self.search_quadrant}'
        else:
            self.search_quadrant = 'full'
            self.quadrant_rationale = 'No significant groupings detected'

        # Set if collisions okay
        if self.risk_tolerance == 'low': self.collision_okay = False
        if self.risk_tolerance == 'medium': self.collision_okay = False
        if self.risk_tolerance == 'high': self.collision_okay = True


    def calculate_quadrant_densities(self):
        """Returns a list of (quadrant, num ships in quadrant) for all 4 quadrants, sorted in descending order of numtargets"""
        current_state = self.env.get_state()
        gameboard_size = self.env.config["gameboard size"]
        quadrant_bounds = {'full': (0, gameboard_size, 0, gameboard_size),
                           'NW': (0, gameboard_size * 0.5, 0, gameboard_size * 0.5),
                           'NE': (gameboard_size * 0.5, gameboard_size, 0, gameboard_size * 0.5),
                           'SW': (0, gameboard_size * 0.5, gameboard_size * 0.5, gameboard_size), 'SE': (
                gameboard_size * 0.5, gameboard_size, gameboard_size * 0.5,
                gameboard_size)}  # specifies (Min x, max x, min y, max y)

        # Calculate how many unknown ships are in each quadrant
        ship_quadrants = {'NW': 0, 'NE': 0, 'SW': 0, 'SE': 0,'full': 0}
        for ship_id in current_state['ships']:
            if not current_state['ships'][ship_id]['observed']:
                if current_state['ships'][ship_id]['position'][0] <= gameboard_size * 0.5 and \
                        current_state['ships'][ship_id]['position'][1] <= gameboard_size * 0.5:
                    ship_quadrants['NW'] += 1
                elif current_state['ships'][ship_id]['position'][0] <= gameboard_size * 0.5 <= \
                        current_state['ships'][ship_id]['position'][1] <= gameboard_size:
                    ship_quadrants['SW'] += 1
                elif gameboard_size * 0.5 <= current_state['ships'][ship_id]['position'][
                    0] <= gameboard_size and gameboard_size * 0.5 <= current_state['ships'][ship_id]['position'][
                    1] <= gameboard_size:
                    ship_quadrants['SE'] += 1
                elif gameboard_size >= current_state['ships'][ship_id]['position'][0] >= gameboard_size * 0.5 >= \
                        current_state['ships'][ship_id]['position'][1]:
                    ship_quadrants['NE'] += 1

        ship_quadrants = sorted(ship_quadrants.items(), key=lambda x: x[1], reverse=True)
        return ship_quadrants

    def thread_the_needle(self):
        """
        Analyzes gameboard to find points that maximize target identification potential
        while staying outside threat rings and ensuring clear paths.
        Returns dictionary of points and their utility scores.
        """
        aircraft = self.env.agents[self.aircraft_id]
        current_state = self.env.get_state()
        scores = {}

        # Get game board dimensions
        board_size = self.env.config["gameboard size"]
        margin = self.env.config["gameboard border margin"]
        increment = 25  # Pixel increment for scanning

        # Get list of all threat rings to avoid
        threat_rings = []  # List of (x, y, radius) tuples
        for ship_id in current_state['ships']:
            ship = self.env.agents[ship_id]
            if ship.observed and ship.observed_threat and ship.threat > 0:
                # Calculate threat radius based on ship's threat level
                threat_radius = ship.width * self.env.AGENT_THREAT_RADIUS[ship.threat]
                threat_rings.append((ship.x, ship.y, threat_radius))

        def path_intersects_threat(start_x, start_y, end_x, end_y, threat_rings):
            """Check if line segment from start to end intersects any threat rings"""
            for tx, ty, tradius in threat_rings:
                # Get vector from start to end
                path_dx = end_x - start_x
                path_dy = end_y - start_y
                path_length = math.hypot(path_dx, path_dy)

                if path_length == 0:
                    return False

                # Normalize direction vector
                path_dx = path_dx / path_length
                path_dy = path_dy / path_length

                # Vector from start to threat center
                to_threat_x = tx - start_x
                to_threat_y = ty - start_y

                # Project threat center onto path
                dot_product = to_threat_x * path_dx + to_threat_y * path_dy

                # Find closest point on path to threat center
                if dot_product < 0:  # Threat is behind start
                    closest_x = start_x
                    closest_y = start_y
                elif dot_product > path_length:  # Threat is beyond end
                    closest_x = end_x
                    closest_y = end_y
                else:  # Threat is alongside path
                    closest_x = start_x + path_dx * dot_product
                    closest_y = start_y + path_dy * dot_product

                # Check if closest point is within threat radius
                threat_dist = math.hypot(tx - closest_x, ty - closest_y)
                if threat_dist <= tradius:
                    return True

            return False

        # Scan board in grid pattern
        for x in range(margin, board_size - margin, increment):
            for y in range(margin, board_size - margin, increment):

                # Skip if point is inside any threat ring
                inside_threat = False
                for tx, ty, tradius in threat_rings:
                    if math.hypot(x - tx, y - ty) <= tradius:
                        inside_threat = True
                        break

                if inside_threat:
                    continue

                # Skip if path to point intersects any threat rings
                if path_intersects_threat(aircraft.x, aircraft.y, x, y, threat_rings):
                    continue

                # Count unknown targets within ISR range of this point
                unknown_targets_in_range = 0
                for ship_id in current_state['ships']:
                    ship = self.env.agents[ship_id]
                    if not ship.observed:  # If target not yet identified
                        # Check if within ISR range of this point
                        if math.hypot(x - ship.x, y - ship.y) <= self.env.AIRCRAFT_ISR_RADIUS:
                            unknown_targets_in_range += 1

                # Calculate distance from aircraft to this point
                dist_to_point = math.hypot(x - aircraft.x, y - aircraft.y)

                # Skip if distance is zero to avoid division by zero
                if dist_to_point == 0:
                    continue

                # Calculate utility score: targets/distance ratio
                utility = unknown_targets_in_range / (dist_to_point + 0.1)

                if unknown_targets_in_range > 0:  # Only store points that can see unknown targets
                    scores[(x, y)] = utility

        return scores

    def find_best_path_point(self):
        """
        Uses thread_the_needle to find the best next waypoint.
        Returns the point with highest utility score.
        """
        scores = self.thread_the_needle()

        if not scores:  # If no valid points found
            return None

        # Find point with maximum utility
        best_point = max(scores.items(), key=lambda x: x[1])[0]
        return best_point
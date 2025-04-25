import math


class AutonomousPolicy:
    def __init__(self,env,aircraft_id):

        self.env = env
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

        # Human overrides for agent priorities
        #self.risk_tolerance = 'medium'  # Override to low/high based on button clicks
        self.search_quadrant_override = 'none'  # 'none' by default, NW/SW/NE/SE/full if human clicks a quadrant. Resets back to auto if autonomous button clicked
        self.search_type_override = 'none'  # 'target' or 'wez' if human clicks buttons. Resets to auto if auto button clicked
        self.hold_commanded = False # If hold button is clicked, goes to true
        self.waypoint_override = False

        self.show_low_level_goals = True
        self.show_high_level_goals = True
        self.show_high_level_rationale = True
        self.show_tracked_factors = True

        self.ticks_since_update = 0
        self.update_rate = 20


    def act(self):  # Execute policy based on chosen gameplan
        self.aircraft = self.env.agents[self.aircraft_id]
        if self.ticks_since_update > self.update_rate:
            self.ticks_since_update = 0
            self.calculate_risk_level()

            if self.hold_commanded:
                self.target_point = self.hold_policy()
                self.low_level_rationale = 'Holding position'
                self.high_level_rationale = '(Human command)'

            elif self.waypoint_override != False: # Execute the waypoint
                self.target_point = self.human_waypoint(self.waypoint_override)

            else:
                self.calculate_priorities()
                self.target_point, closest_target_distance = self.basic_search()

                # Populate agent info
                self.current_target_distance = closest_target_distance
                if closest_target_distance: self.low_level_rationale = f'Identifying target {int(closest_target_distance)} units away'
                if self.search_type_override != 'none':
                    self.high_level_rationale = '(Human command)'

            self.update_agent_info()
        else:
            self.ticks_since_update += 1

        self.id_method = 0 # TODO hardcoded for now

        self.action = {'waypoint':self.target_point, 'id_method':self.id_method}
        return self.action


    def basic_search(self):
        current_state = self.env.get_state()
        current_target_distances = {}  # Will be {agent_idx:distance}
        #closest_distance = None
        dist = None

        gameboard_size = self.env.config["gameboard size"]
        quadrant_bounds = {'full': (0, gameboard_size, 0, gameboard_size),'NW': (0, gameboard_size * 0.5, 0, gameboard_size * 0.5),'NE': (gameboard_size * 0.5, gameboard_size, 0, gameboard_size * 0.5),'SW': (0, gameboard_size * 0.5, gameboard_size * 0.5, gameboard_size), 'SE': (gameboard_size * 0.5, gameboard_size, gameboard_size * 0.5,gameboard_size)}  # specifies (Min x, max x, min y, max y)

        # Find nearby threats for status display
        # self.nearby_threats = []
        # threat_distances = {}
        # for ship_id in current_state['ships']:
        #     ship = current_state['ships'][ship_id]
        #     if ship['observed'] and ship['observed threat'] and ship['threat'] > 0:
        #         dist = math.hypot(self.aircraft.x - self.env.agents[ship_id].x,self.aircraft.y - self.env.agents[ship_id].y)
        #         threat_distances[ship_id] = dist

        # Get 2 closest threats
        # sorted_threats = sorted(threat_distances.items(), key=lambda x: x[1])
        # for threat_id, dist in sorted_threats[:2]:
        #     self.nearby_threats.append((threat_id, int(dist)))

        for ship_id in current_state['ships']: # Loop through all ships in environment, calculate distance, find closest unknown ship (or unknown WEZ), and set waypoint to that location
            closest_distance = None
            if self.search_type == 'target':  # If set to target, only consider unknown targets

                if current_state['ships'][ship_id]['observed'] == False and (
                        quadrant_bounds[self.search_quadrant][0] <=  current_state['ships'][ship_id]['position'][0] <= quadrant_bounds[self.search_quadrant][1]) and (
                            quadrant_bounds[self.search_quadrant][2] <=  current_state['ships'][ship_id]['position'][1] <=quadrant_bounds[self.search_quadrant][3]):
                    
                    dist = math.hypot(self.aircraft.x - self.env.agents[ship_id].x,self.aircraft.y - self.env.agents[ship_id].y)
                    current_target_distances[ship_id] = dist

                if dist is not None:
                    if closest_distance is None or dist < closest_distance:
                        closest_distance = dist

            elif self.search_type == 'wez':  # If set to wez, consider unknown targets AND known hostiles with unknown threat rings
                closest_distance = None
                if (current_state['ships'][ship_id]['observed'] == False or current_state['ships'][ship_id]['observed threat'] == False) and (
                    quadrant_bounds[self.search_quadrant][0] <= current_state['ships'][ship_id]['position'][0] <=quadrant_bounds[self.search_quadrant][1]) and (
                        quadrant_bounds[self.search_quadrant][2] <= current_state['ships'][ship_id]['position'][1] <=quadrant_bounds[self.search_quadrant][3]):
                    
                    dist = math.hypot(self.aircraft.x - self.env.agents[ship_id].x,self.aircraft.y - self.env.agents[ship_id].y)
                    current_target_distances[ship_id] = dist
                    if dist is not None:
                        if closest_distance is None or dist < closest_distance:
                            closest_distance = dist

            elif self.search_type == 'tag team': # Only search unknown targets and WEZs within 300 pixels of the human ship (TODO testing)
                ship_to_human = math.hypot(self.env.agents[self.env.num_ships + 1].x - self.env.agents[ship_id].x, self.env.agents[self.env.num_ships + 1].y - self.env.agents[ship_id].y)
                if (current_state['ships'][ship_id]['observed'] == True and current_state['ships'][ship_id]['observed threat'] == False) and (ship_to_human < 300):
                    #dist = math.hypot(self.aircraft.x - self.env.agents[ship_id].x,self.aircraft.y - self.env.agents[ship_id].y)
                    dist = ship_to_human
                    current_target_distances[ship_id] = dist
                    if closest_distance is None or dist < closest_distance:
                        closest_distance = dist

        if current_target_distances: # If there are targets nearby, set waypoint to the nearest one
            nearest_target_id = min(current_target_distances, key=current_target_distances.get)
            target_waypoint = tuple((self.env.agents[nearest_target_id].x, self.env.agents[nearest_target_id].y))

        elif self.search_type == 'tag team':
            target_waypoint = (self.env.agents[self.env.num_ships + 1].x,self.env.agents[self.env.num_ships + 1].y)

        else:  # If all targets ID'd, loiter in center of board or specified quadrant
            quadrant_centers = {'full': (gameboard_size * 0.5,gameboard_size * 0.5), 'NW': (gameboard_size * 0.25, gameboard_size * 0.25), 'NE':(gameboard_size * 0.75, gameboard_size * 0.25), 'SW':(gameboard_size * 0.25, gameboard_size * 0.75),'SE':(gameboard_size * 0.75, gameboard_size * 0.75)}
            target_waypoint =  quadrant_centers[self.search_quadrant]

        return target_waypoint, closest_distance


    def hold_policy(self):
        target_waypoint = self.aircraft.x, self.aircraft.y
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
            threats_text = ", ".join([f"{t[1]} units" for t in self.nearby_threats])

        # Build status text with conditional sections
        self.status_lines = []

        # Low level goals
        if self.show_low_level_goals:
            self.status_lines.append(f"CURRENT GOAL: {self.low_level_rationale}")

        # High level goals
        if self.show_high_level_goals:
            if self.search_type:
                if self.show_high_level_rationale: self.status_lines.append(f"SEARCH TYPE: {self.search_type.upper()} {self.high_level_rationale}")
                else: self.status_lines.append(f"SEARCH TYPE: {self.search_type.upper()}")
            if self.search_quadrant:
                if self.show_high_level_rationale: self.status_lines.append(f"SEARCH AREA: {self.search_quadrant.upper()} {self.quadrant_rationale}")
                else: self.status_lines.append(f"SEARCH AREA: {self.search_quadrant.upper()}")

        # Tracked factors
        if self.show_tracked_factors:
            self.status_lines.append(f"RISK LEVEL: {self.risk_level}")
            self.status_lines.append(f"NEARBY THREATS: {threats_text}")

        # Update the display
        self.env.agent_info_display.text = self.status_lines
        self.env.agent_info_display.update_text(self.status_lines)


    def calculate_risk_level(self): # Calculates how risky the current situation is, as a function of agent health and number of nearby hostile targets
        hostile_targets_nearby = sum(1 for agent in self.env.agents if agent.agent_class == "ship" and agent.threat > 0 and agent.observed_threat and math.hypot(agent.x - self.aircraft.x, agent.y - self.aircraft.y) <= 30)

        risk_level_function = 10 * hostile_targets_nearby + self.env.agents[self.env.num_ships].damage
        self.risk_level = 'LOW' if risk_level_function <= 30 else 'MEDIUM' if risk_level_function <= 60 else 'HIGH' if risk_level_function <= 80 else 'EXTREME'


    def calculate_priorities(self): # Set search type (target or wez), search quadrant (One of the 4 quadrants or the full board)
        # Set search_type
        if self.search_type_override != 'none': # If player has set a specific search type, use that
            self.search_type = self.search_type_override
            self.high_level_rationale = '(Human command)'

        else: # Choose according to selected risk tolerance
            if abs(self.env.time_limit - self.env.display_time / 1000) <= 30:
                self.search_type = 'wez'
                self.high_level_rationale = '(Critical time remaining)'
            elif self.aircraft.damage <= 50:
                self.search_type = 'wez'
                self.high_level_rationale = ''
            else:
                self.search_type = 'target'
                if self.aircraft.damage > 50: self.high_level_rationale = '(Low health)'
                else: self.high_level_rationale = ''

        # Set quadrant to search in. If the densest quadrant is substantially denser than the second most dense, prioritize that quadrant. Otherwise do not prioritize a quadrant.
        quadrants = self.calculate_quadrant_densities()
        if self.search_quadrant_override != 'none':
            self.search_quadrant = self.search_quadrant_override
            self.quadrant_rationale = '(Human command)'
        elif quadrants[0][1] >= quadrants[1][1] + 7:
            self.search_quadrant = quadrants[0][0]
            self.quadrant_rationale = f'(Significant target grouping in {self.search_quadrant})'
        else:
            self.search_quadrant = 'full'
            self.quadrant_rationale = ''


    def calculate_quadrant_densities(self):
        """Returns a list of (quadrant, num ships in quadrant) for all 4 quadrants, sorted in descending order of numtargets"""
        current_state = self.env.get_state()
        gameboard_size = self.env.config["gameboard size"]
        quadrant_bounds = {'full': (0, gameboard_size, 0, gameboard_size),
                           'NW': (0, gameboard_size * 0.5, 0, gameboard_size * 0.5),
                           'NE': (gameboard_size * 0.5, gameboard_size, 0, gameboard_size * 0.5),
                           'SW': (0, gameboard_size * 0.5, gameboard_size * 0.5, gameboard_size),
                           'SE': (gameboard_size * 0.5, gameboard_size, gameboard_size * 0.5,gameboard_size)}  # specifies (Min x, max x, min y, max y)

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
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
        self.aircraft = env.agents[aircraft_id]

        self.target_point = (0,0)

        # For displaying on agent status window
        self.low_level_rationale = '' # Identify unknown target, identify unknown WEZ, or evade threat
        self.high_level_rationale = ''
        self.three_upcoming_targets = [] # Target IDs of the nearest three unknown targets, for displaying later (not implemented yet)

        self.risk_level = ''

        # Agent priorities to follow during search (can be overridden by human)
        self.search_quadrant = '' # Auto-selected by policy unless search_quadrant_override is not 'none'
        self.search_type = '' # Auto-selected by policy unless search_type_override is not 'none'
        self.collision_ok = False  # If False, collision avoidance executes normally.

        # Human overrides for agent priorities
        self.risk_tolerance = 'medium'  # Override to low/high based on button clicks
        self.search_quadrant_override = 'none'  # 'none' by default, NW/SW/NE/SE/full if human clicks a quadrant. Resets back to auto if autonomous button clicked
        self.search_type_override = 'none'  # 'target' or 'wez' if human clicks buttons. Resets to auto if auto button clicked
        self.hold_commanded = False # If hold button is clicked, goes to true

        # TODO overhaul this based on experiment variables
        # Low level goals and ratioanle, high level goals and rationale, etc
        self.agent_info = {
        'action': {'type': None, 'details': None},
        'risk': {'level': None, 'threats': [], 'safe_corridors': []},
        'targeting': {'priority': None, 'reason': None},
        'decision': {'strategy': None, 'constraints': []}
    }
        """
        Agent info types:
        
        """


    def act(self):  # Execute policy based on chosen gameplan
        print('act running')
        if self.hold_commanded:
            self.target_point = self.hold_policy()
        if self.upcoming_collision() and not self.collision_ok:
            self.target_point = self.collision_avoidance()
            self.low_level_rationale = 'EVADE THREAT'
        else:
            self.calculate_priorities()
            #print(self.search_quadrant)
            self.target_point, _ = self.basic_search()
            #print('Target point set to %s' % (self.target_point,))

        self.update_agent_info()


    def basic_search(self):
        # TODO: Doesn't seem to route to the closest unknown target in the specified quadrant

        current_state = self.env.get_state()
        current_target_distances = {}  # Will be {agent_idx:distance}

        gameboard_size = self.env.config["gameboard size"]
        quadrant_bounds = {'full': (0, gameboard_size, 0, gameboard_size),'NW': (0, gameboard_size * 0.5, 0, gameboard_size * 0.5),'NE': (gameboard_size * 0.5, gameboard_size, 0, gameboard_size * 0.5),'SW': (0, gameboard_size * 0.5, gameboard_size * 0.5, gameboard_size), 'SE': (gameboard_size * 0.5, gameboard_size, gameboard_size * 0.5,gameboard_size)}  # specifies (Min x, max x, min y, max y)

        print(self.search_quadrant)

        for ship_id in current_state['ships']: # Loop through all ships in environment, calculate distance, find closest unknown ship (or unknown WEZ), and set waypoint to that location
            if self.search_type == 'target':  # If set to target, only consider unknown targets
                if current_state['ships'][ship_id]['observed'] == False and (
                        quadrant_bounds[self.search_quadrant][0] <=  current_state['ships'][ship_id]['position'][0] <=
                        quadrant_bounds[self.search_quadrant][1]) and (quadrant_bounds[self.search_quadrant][2] <=  current_state['ships'][ship_id]['position'][1] <=quadrant_bounds[self.search_quadrant][3]):
                    dist = math.hypot(self.aircraft.x - self.env.agents[ship_id].x,self.aircraft.y - self.env.agents[ship_id].x)
                    current_target_distances[ship_id] = dist

            elif self.search_type == 'wez':  # If set to wez, consider unknown targets AND known hostiles with unknown threat rings
                if (current_state['ships'][ship_id]['observed'] == False or current_state['ships'][ship_id]['observed threat'] == False) and (quadrant_bounds[self.search_quadrant][0] <= current_state['ships'][ship_id]['position'][0] <=quadrant_bounds[self.search_quadrant][1]) and (quadrant_bounds[self.search_quadrant][2] <= current_state['ships'][ship_id]['position'][1] <=quadrant_bounds[self.search_quadrant][3]):
                    dist = math.hypot(self.aircraft.x - self.env.agents[ship_id].x,self.aircraft.y - self.env.agents[ship_id].y)
                    current_target_distances[ship_id] = dist
        print('current target distances:')
        print(current_target_distances)

        if current_target_distances: # If there are targets nearby, set waypoint to the nearest one
            self.three_upcoming_targets = [item[0] for item in sorted(current_target_distances.items(), key=lambda x: x[1], reverse=True)[:3]] # Store the three closest target IDs for displaying later
            nearest_target_id = min(current_target_distances, key=current_target_distances.get)
            target_waypoint = tuple((self.env.agents[nearest_target_id].x, self.env.agents[nearest_target_id].y))

        else:  # If all targets ID'd, loiter in center of board or specified quadrant
            quadrant_centers = {'full': (gameboard_size * 0.5,gameboard_size * 0.5), 'NW': (gameboard_size * 0.25, gameboard_size * 0.25), 'NE':(gameboard_size * 0.75, gameboard_size * 0.25), 'SW':(gameboard_size * 0.25, gameboard_size * 0.75),'SE':(gameboard_size * 0.75, gameboard_size * 0.75)}
            target_waypoint =  quadrant_centers[self.search_quadrant]

        target_direction = math.atan2(target_waypoint[1] - self.aircraft.y,target_waypoint[0] - self.aircraft.x)
        return target_waypoint, target_direction


    def collision_avoidance(self):
        # TODO implement
        # If    hostile is within X   pixels     straight     ahead, command     a    deflected    waypoint    off    to    the    side    Find    way    to    handle    being    surrounded
        pass


    def hold_policy(self):
        target_waypoint = self.aircraft.x, self.aircraft.y
        target_direction = math.atan2(target_waypoint[1] - self.aircraft.y, target_waypoint[0] - self.aircraft.x)
        return target_waypoint, target_direction


    # Push various types of information about the agent's decision-making to the AGENT STATUS window, if set.
    # TODO: Scrub through these and make sure they reflect what's actually going on
    def update_agent_info(self):
        current_state = self.env.get_state()

        if self.env.config['show_risk_info']:
            nearby_threats = []
            for ship_id in current_state['ships']:
                ship = self.env.agents[ship_id]
                if ship.threat > 0 and ship.observed_threat:
                    dist = math.hypot(self.aircraft.x - ship.x, self.aircraft.y - ship.y)
                    if dist < ship.width * self.env.AGENT_THREAT_RADIUS[ship.threat] * 2:
                        nearby_threats.append({
                            'position': (ship.x, ship.y),
                            'distance': dist,
                            'bearing': math.degrees(math.atan2(ship.y - self.aircraft.y, ship.x - self.aircraft.x))})

            self.agent_info['risk']['threats'] = nearby_threats
            self.agent_info['risk']['level'] = 'HIGH' if len(nearby_threats) > 1 else 'MEDIUM' if nearby_threats else 'LOW'

        # Gather targeting information
        if self.env.config['show_current_action']:
            unidentified_targets = []
            for ship_id in current_state['ships']:
                ship = self.env.agents[ship_id]
                if not ship.observed or (self.search_type == 'wez' and not ship.observed_threat):
                    dist = math.hypot(self.aircraft.x - ship.x, self.aircraft.y - ship.y)
                    unidentified_targets.append({
                        'id': ship_id,
                        'distance': dist,
                        'position': (ship.x, ship.y)})

            if unidentified_targets:
                closest_target = min(unidentified_targets, key=lambda x: x['distance'])
                self.agent_info['targeting']['priority'] = closest_target
                self.agent_info['action']['type'] = 'identify_target'
                self.agent_info['action']['details'] = f"Moving to identify target at {closest_target['distance']:.0f} units"

            # Decision rationale
            if self.env.config['show_decision_rationale']:
                unidentified_targets = []
                if len(nearby_threats) >= 2:
                    self.agent_info['decision']['strategy'] = 'threat_avoidance'
                    self.agent_info['decision']['constraints'].append('Multiple threats detected - calculating escape route')
                elif unidentified_targets:
                    self.agent_info['decision']['strategy'] = 'target_identification'
                    safe_approach = all(t['distance'] > 200 for t in nearby_threats)
                    self.agent_info['decision']['constraints'].append('Rationale: Direct approach available' if safe_approach else 'Maneuvering required for safe approach')


    # Helper functions to determine behavior:
    # TODO write
    def upcoming_collision(self):
        pass


    def calculate_risk_level(self): # Calculates how risky the current situation is, as a function of agent health and number of nearby hostile targets

        hostile_targets_nearby = sum(1 for agent in self.env.agents
                                     if agent.agent_class == "ship" and agent.threat > 0 and agent.observed_threat and math.hypot(agent.x - self.aircraft.x, agent.y - self.aircraft.y) <= 30)

        risk_level_function = 10 * hostile_targets_nearby + self.agents[self.num_ships].damage
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
        if quadrants[0][1] >= quadrants[1][1] + 7:
            self.search_quadrant = quadrants[0][0]
            self.agent_info['decision']['constraints'].append(f'High target density in {self.search_quadrant} quadrant')
        else:
            self.search_quadrant = 'full'
            self.agent_info['decision']['constraints'].append(f'No significant target clustering found, prioritizing full board')

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
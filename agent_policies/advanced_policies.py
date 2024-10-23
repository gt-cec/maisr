import math



def target_id_policy(env, aircraft_id, quadrant='full', id_type='target'):
    """
    Basic rule-based action policy for tactical HAI ISR project.
    Inputs:
        * Env: Game environment
        * aircraft_id: ID of the aircraft being moved
        * quadrant: Specifies whether agent is restricted to search in a specific map quadrant. Default is 'full' (all quadrants allowed). Alternatives are 'NW', 'NE', 'SW', 'SE' as strings.
        * id_type: 'target' (only ID unknown targets but not unknown WEZs of known hostiles) or 'wez' (ID unknown WEZs)
    Returns: Waypoint to the nearest unknown target
    """
    waypoints_to_show = env.config['show agent waypoint']
    gameboard_size = env.config[
        "gameboard size"]  # TODO: Currently set to full game window, not just inside the green bounds (10% to 90% of gameboard size)
    quadrant_bounds = {'full': (0, gameboard_size, 0, gameboard_size),
                       'NW': (0, gameboard_size * 0.5, 0, gameboard_size * 0.5),
                       'NE': (gameboard_size * 0.5, gameboard_size, 0, gameboard_size * 0.5),
                       'SW': (0, gameboard_size * 0.5, gameboard_size * 0.5, gameboard_size), 'SE': (
        gameboard_size * 0.5, gameboard_size, gameboard_size * 0.5,
        gameboard_size)}  # specifies (Min x, max x, min y, max y)

    current_state = env.get_state()
    current_target_distances = {}  # Will be {agent_idx:distance}


    # Information reporting (TODO testing)
    info = {
        'action': {'type': None, 'details': None},
        'risk': {'level': None, 'threats': [], 'safe_corridors': []},
        'targeting': {'priority': None, 'reason': None},
        'decision': {'strategy': None, 'constraints': []}
    }

    # Check for threats and gather risk information
    if env.config['show_risk_info']:
        nearby_threats = []
        for ship_id in current_state['ships']:
            ship = env.agents[ship_id]
            if ship.threat > 0 and ship.observed_threat:
                dist = math.hypot(env.agents[aircraft_id].x - ship.x, env.agents[aircraft_id].y - ship.y)
                if dist < ship.width * env.AGENT_THREAT_RADIUS[ship.threat] * 2:
                    nearby_threats.append({
                        'position': (ship.x, ship.y),
                        'distance': dist,
                        'bearing': math.degrees(math.atan2(ship.y - env.agents[aircraft_id].y, ship.x - env.agents[aircraft_id].x))
                    })

        info['risk']['threats'] = nearby_threats
        info['risk']['level'] = 'HIGH' if len(nearby_threats) > 1 else 'MEDIUM' if nearby_threats else 'LOW'

    # Gather targeting information
    if env.config['show_current_action']:
        unidentified_targets = []
        for ship_id in current_state['ships']:
            ship = env.agents[ship_id]
            if not ship.observed or (id_type == 'wez' and not ship.observed_threat):
                dist = math.hypot(env.agents[aircraft_id].x - ship.x, env.agents[aircraft_id].y - ship.y)
                unidentified_targets.append({
                    'id': ship_id,
                    'distance': dist,
                    'position': (ship.x, ship.y)
                })

        if unidentified_targets:
            closest_target = min(unidentified_targets, key=lambda x: x['distance'])
            info['targeting']['priority'] = closest_target
            info['action']['type'] = 'identify_target'
            info['action']['details'] = f"Moving to identify target at {closest_target['distance']:.0f} units"

    # Decision rationale
    if env.config['show_decision_rationale']:
        unidentified_targets = []
        if len(nearby_threats) >= 2:
            info['decision']['strategy'] = 'threat_avoidance'
            info['decision']['constraints'].append('Multiple threats detected - calculating escape route')
        elif unidentified_targets:
            info['decision']['strategy'] = 'target_identification'
            safe_approach = all(t['distance'] > 200 for t in nearby_threats)
            info['decision']['constraints'].append('Rationale: Direct approach available' if safe_approach else 'Maneuvering required for safe approach')


    for ship_id in current_state['ships']:
        # Loops through all ships in the environment, calculates distance from current aircraft position, finds the closest unknown ship (or unknown WEZ), and sets aircraft waypoint to that ship's location.
        if id_type == 'target':  # If set to target, only consider unknown targets
            if current_state['ships'][ship_id]['observed'] == False and (
                    quadrant_bounds[quadrant][0] <= current_state['ships'][ship_id]['position'][0] <=
                    quadrant_bounds[quadrant][1]) and (
                    quadrant_bounds[quadrant][2] <= current_state['ships'][ship_id]['position'][1] <=
                    quadrant_bounds[quadrant][3]):
                dist = math.hypot(env.agents[aircraft_id].x - env.agents[ship_id].x,
                                  env.agents[aircraft_id].y - env.agents[ship_id].y)
                current_target_distances[ship_id] = dist
        elif id_type == 'wez':  # If set to wez, consider unknown targets AND known hostiles with unknown threat rings
            if (current_state['ships'][ship_id]['observed'] == False or current_state['ships'][ship_id][
                'observed threat'] == False) and (
                    quadrant_bounds[quadrant][0] <= current_state['ships'][ship_id]['position'][0] <=
                    quadrant_bounds[quadrant][1]) and (
                    quadrant_bounds[quadrant][2] <= current_state['ships'][ship_id]['position'][1] <=
                    quadrant_bounds[quadrant][3]):
                dist = math.hypot(env.agents[aircraft_id].x - env.agents[ship_id].x,
                                  env.agents[aircraft_id].y - env.agents[ship_id].y)
                current_target_distances[ship_id] = dist

    if current_target_distances:
        nearest_target_id = min(current_target_distances, key=current_target_distances.get)
        target_waypoint = tuple((env.agents[nearest_target_id].x, env.agents[nearest_target_id].y))
        # print('Nearest unknown target is %s. Setting waypoint to %s' % (nearest_target_id, target_waypoint))


    else:  # If all targets ID'd, loiter in center of board or specified quadrant
        if quadrant == 'full':
            target_waypoint = (gameboard_size * 0.5,
                               gameboard_size * 0.5)  # If no more targets, return to center of game board TODO: Make this more robust
        elif quadrant == 'NW':
            target_waypoint = (gameboard_size * 0.25, gameboard_size * 0.25)
        elif quadrant == 'NE':
            target_waypoint = (gameboard_size * 0.75, gameboard_size * 0.25)
        elif quadrant == 'SW':
            target_waypoint = (gameboard_size * 0.25, gameboard_size * 0.75)
        elif quadrant == 'SE':
            target_waypoint = (gameboard_size * 0.75, gameboard_size * 0.75)

    target_direction = math.atan2(target_waypoint[1] - env.agents[aircraft_id].y,
                                  target_waypoint[0] - env.agents[aircraft_id].x)

    """if waypoints_to_show >= 2:
        # draw line from target waypoint to next closest unknown target
        second_nearest_target_id = sorted(current_target_distances, key=current_target_distances.get)[1]
        second_waypoint = tuple((env.agents[second_nearest_target_id].x, env.agents[second_nearest_target_id].y))
        pygame.draw.line(env.window, (0, 0, 0), (env.agents[nearest_target_id].x, env.agents[nearest_target_id].y), (env.agents[second_nearest_target_id].x, env.agents[second_nearest_target_id].y),2)  # Draw line from first to second target
        # TODO testing"""

    # if waypoints_to_show == 3:

    env.agent_info = info

    return target_waypoint, target_direction


def autonomous_policy(env,aircraft_id,quadrant='full',id_type='target'):
    """
    "Autonomous" policy that chooses waypoints based on what it deems to be best.
    If damage <= 50, attempts to identify closest...
    """
    gameboard_size = env.config["gameboard size"] # TODO: Currently set to full game window, not just inside the green bounds (10% to 90% of gameboard size)
    quadrant_bounds = {'full':(0,gameboard_size,0,gameboard_size), 'NW':(0,gameboard_size*0.5,0,gameboard_size*0.5),'NE':(gameboard_size*0.5,gameboard_size,0,gameboard_size*0.5),'SW':(0,gameboard_size*0.5,gameboard_size*0.5,gameboard_size),'SE':(gameboard_size*0.5,gameboard_size,gameboard_size*0.5,gameboard_size)} # specifies (Min x, max x, min y, max y)

    current_state = env.get_state()
    current_target_distances = {} # Will be {agent_idx:distance}

    info = {
        'action': {'type': None, 'details': None},
        'risk': {'level': None, 'threats': []},
        'decision': {'strategy': None, 'constraints': []}
    }

    # Assess nearby threats and overall risk
    nearby_threats = []
    for ship_id in current_state['ships']:
        ship = env.agents[ship_id]
        if ship.threat > 0 and ship.observed_threat:
            threat_radius = ship.width * env.AGENT_THREAT_RADIUS[ship.threat]
            dist = math.hypot(env.agents[aircraft_id].x - ship.x, env.agents[aircraft_id].y - ship.y)
            if dist < threat_radius * 2.5:
                nearby_threats.append({
                    'position': (ship.x, ship.y),
                    'distance': dist,
                    'bearing': math.degrees(math.atan2(ship.y - env.agents[aircraft_id].y, ship.x - env.agents[aircraft_id].x))
                })

    # Determine risk level based on threats and damage
    if env.agents[aircraft_id].damage >= 75:
        info['risk']['level'] = 'EXTREME'
    elif env.agents[aircraft_id].damage >= 50 or len(nearby_threats) >= 2:
        info['risk']['level'] = 'HIGH'
    elif env.agents[aircraft_id].damage >= 25 or len(nearby_threats) == 1:
        info['risk']['level'] = 'MEDIUM'
    else:
        info['risk']['level'] = 'LOW'

    info['risk']['threats'] = nearby_threats

    # Determine strategy based on risk level
    if info['risk']['level'] in ['HIGH', 'EXTREME']:
        # Prioritize survival and target identification only
        id_type = 'target'
        info['decision']['strategy'] = 'defensive_search'
        info['decision']['constraints'].append('Prioritizing survival over WEZ identification')
    else:
        # Can pursue more aggressive WEZ identification
        id_type = 'wez'
        info['decision']['strategy'] = 'aggressive_search'
        info['decision']['constraints'].append('Conditions permit WEZ identification')



    # Decide whether to ID targets or targets + WEZs
    if env.agents[aircraft_id].damage <= 50:
        #print('Autonomous policy prioritizing target+WEZ search')
        id_type = 'wez'
    else:
        #print('Autonomous policy prioritizing target search')
        id_type = 'target'

    # Determine which quadrant has most unknown targets (TODO: Very inefficient, combine with other for loop below
    ship_quadrants = {'NW':0,'NE':0,'SW':0,'SE':0,'full':0}  # For counting how many current unknown ships in each quadrant
    for ship_id in current_state['ships']:
        if current_state['ships'][ship_id]['observed'] == False:
            if current_state['ships'][ship_id]['position'][0] <= gameboard_size*0.5 and current_state['ships'][ship_id]['position'][1] <= gameboard_size*0.5:
                ship_quadrants['NW'] += 1
            elif current_state['ships'][ship_id]['position'][0] <= gameboard_size*0.5 and gameboard_size*0.5 <= current_state['ships'][ship_id]['position'][1] <= gameboard_size:
                ship_quadrants['SW'] += 1
            elif gameboard_size*0.5 <= current_state['ships'][ship_id]['position'][0] <= gameboard_size and gameboard_size * 0.5 <= current_state['ships'][ship_id]['position'][1] <= gameboard_size:
                ship_quadrants['SE'] += 1
            elif gameboard_size*0.5 <= current_state['ships'][ship_id]['position'][0] <= gameboard_size and current_state['ships'][ship_id]['position'][1] <= gameboard_size*0.5:
                ship_quadrants['NE'] += 1

    new_densest_quadrant = max(ship_quadrants, key=ship_quadrants.get) # Set search quadrant to the one with the most unknown ships
    info['decision']['constraints'].append(f'High target density in {quadrant} quadrant')
    # else: quadrant = 'full'
    # info['decision']['constraints'].append('No significant target clustering found')

    if ship_quadrants[new_densest_quadrant] > 3 + ship_quadrants[quadrant]:  # TODO: Bug: Spamming console because quadrant always re-initializes as 'full'. Need to fix.
        #densest_quadrant = new_densest_quadrant
        quadrant = new_densest_quadrant
        #print('Autonomous policy prioritizing quadrant %s' % (quadrant,))

    for ship_id in current_state['ships']:
        # Loops through all ships in the environment, calculates distance from current aircraft position, finds the closest unknown ship (or unknown WEZ), and sets aircraft waypoint to that ship's location.
        if id_type == 'target': # If set to target, only consider unknown targets
            if current_state['ships'][ship_id]['observed'] == False and (quadrant_bounds[quadrant][0] <= current_state['ships'][ship_id]['position'][0] <= quadrant_bounds[quadrant][1]) and (quadrant_bounds[quadrant][2] <= current_state['ships'][ship_id]['position'][1] <= quadrant_bounds[quadrant][3]):
                dist = math.hypot(env.agents[aircraft_id].x - env.agents[ship_id].x, env.agents[aircraft_id].y - env.agents[ship_id].y)
                current_target_distances[ship_id] = dist
        elif id_type == 'wez': # If set to wez, consider unknown targets AND known hostiles with unknown threat rings
            if (current_state['ships'][ship_id]['observed'] == False or current_state['ships'][ship_id]['observed threat'] == False) and (quadrant_bounds[quadrant][0] <= current_state['ships'][ship_id]['position'][0] <= quadrant_bounds[quadrant][1]) and (quadrant_bounds[quadrant][2] <= current_state['ships'][ship_id]['position'][1] <= quadrant_bounds[quadrant][3]):
                dist = math.hypot(env.agents[aircraft_id].x - env.agents[ship_id].x,env.agents[aircraft_id].y - env.agents[ship_id].y)
                current_target_distances[ship_id] = dist

    if current_target_distances:
        nearest_target_id = min(current_target_distances, key=current_target_distances.get)
        target_waypoint = tuple((env.agents[nearest_target_id].x, env.agents[nearest_target_id].y))
        nearest_target = env.agents[nearest_target_id]

        bearing = math.degrees(math.atan2(nearest_target.y - env.agents[aircraft_id].y,
                                          nearest_target.x - env.agents[aircraft_id].x))
        distance = current_target_distances[nearest_target_id]

        info['action']['type'] = 'identify_target'
        info['action'][
            'details'] = f"Moving to {'WEZ identify' if id_type == 'wez' else 'identify'} target at {bearing:.0f}°, {distance:.0f} units"


        #print('Nearest unknown target is %s. Setting waypoint to %s' % (nearest_target_id, target_waypoint))

    else: # If all targets ID'd, loiter in center of board or specified quadrant
        info['action']['type'] = 'returning'
        info['action']['details'] = f"Returning to {quadrant} center to wait for new targets"
        if quadrant == 'full': target_waypoint = (gameboard_size*0.5,gameboard_size*0.5) # If no more targets, return to center of game board TODO: Make this more robust
        elif quadrant == 'NW': target_waypoint = (gameboard_size*0.25,gameboard_size*0.25)
        elif quadrant == 'NE': target_waypoint = (gameboard_size * 0.75, gameboard_size * 0.25)
        elif quadrant == 'SW': target_waypoint = (gameboard_size * 0.25, gameboard_size * 0.75)
        elif quadrant == 'SE': target_waypoint = (gameboard_size * 0.75, gameboard_size * 0.75)

    target_direction = math.atan2(target_waypoint[1] - env.agents[aircraft_id].y,target_waypoint[0] - env.agents[aircraft_id].x)
    env.agent_info = info
    return target_waypoint, target_direction


"""NOTE: Policies below are not fully functional"""

# safe_target_id_policy
# Code below works fast but tends to get stick on the edge of a target's threat ring
def safe_target_id_policy(env, aircraft_id, quadrant='full', id_type='target'):
    # Get the original waypoint from the existing target_id_policy
    original_waypoint, original_direction = target_id_policy(env, aircraft_id, quadrant, id_type)

    aircraft = env.agents[aircraft_id]
    start = (aircraft.x, aircraft.y)

    # If we're already at the waypoint, no need to adjust
    if start == original_waypoint:
        return original_waypoint, original_direction

    # Check if the direct path to the waypoint is safe
    if is_path_safe(env, start, original_waypoint):
        return original_waypoint, original_direction

    # If not safe, find a safe intermediate waypoint
    safe_waypoint = find_safe_intermediate_waypoint(env, start, original_waypoint)

    if safe_waypoint:
        new_direction = math.atan2(safe_waypoint[1] - aircraft.y, safe_waypoint[0] - aircraft.x)
        return safe_waypoint, new_direction

    # If no safe waypoint found, move towards the original waypoint as far as safely possible
    safe_waypoint = move_towards_safely(env, start, original_waypoint)
    new_direction = math.atan2(safe_waypoint[1] - aircraft.y, safe_waypoint[0] - aircraft.x)
    return safe_waypoint, new_direction


def is_path_safe(env, start, end):
    # Check points along the path
    steps = 10
    for i in range(steps + 1):
        t = i / steps
        x = start[0] + t * (end[0] - start[0])
        y = start[1] + t * (end[1] - start[1])
        if not is_point_safe(env, (x, y)):
            return False
    return True


def is_point_safe(env, point):
    for agent in env.agents:
        if agent.agent_class == "ship" and agent.threat > 0:
            distance = math.hypot(point[0] - agent.x, point[1] - agent.y)
            if distance <= agent.width * env.AGENT_THREAT_RADIUS[agent.threat]:
                return False
    return True


def find_safe_intermediate_waypoint(env, start, end):
    vector = (end[0] - start[0], end[1] - start[1])
    perpendicular = (-vector[1], vector[0])
    magnitude = math.hypot(*perpendicular)
    if magnitude == 0:
        return None
    unit_perpendicular = (perpendicular[0] / magnitude, perpendicular[1] / magnitude)

    for fraction in [0.25, 0.5, 0.75]:  # Try different points along the path
        for distance in [50, 100, 150, -50, -100, -150]:  # Try different perpendicular distances
            intermediate = (
                start[0] + vector[0] * fraction + unit_perpendicular[0] * distance,
                start[1] + vector[1] * fraction + unit_perpendicular[1] * distance
            )
            if is_point_safe(env, intermediate) and is_path_safe(env, start, intermediate) and is_path_safe(env,
                                                                                                            intermediate,
                                                                                                            end):
                return intermediate

    return None


def move_towards_safely(env, start, end):
    vector = (end[0] - start[0], end[1] - start[1])
    distance = math.hypot(*vector)
    if distance == 0:
        return start

    unit_vector = (vector[0] / distance, vector[1] / distance)
    step_size = 10  # pixels

    for i in range(1, int(distance / step_size) + 1):
        point = (start[0] + unit_vector[0] * i * step_size,
                 start[1] + unit_vector[1] * i * step_size)
        if not is_point_safe(env, point):
            # Return the last safe point
            return (start[0] + unit_vector[0] * (i - 1) * step_size,
                    start[1] + unit_vector[1] * (i - 1) * step_size)

    # If the entire path is safe, return the end point
    return end



# Collision avoidance policy suite (includes collision_avoidance_target_id_policy, find_escape_route, intersects_threat_ring, and calculate_avoidance_waypoint
# Tends to get stuck between threats
def collision_avoidance_target_id_policy(env, aircraft_id, quadrant='full', id_type='target'):
    """
    Modified target ID policy that includes threat avoidance behavior.
    Returns waypoint to nearest unknown target while avoiding threat rings.
    """
    gameboard_size = env.config["gameboard size"]
    quadrant_bounds = {
        'full': (0, gameboard_size, 0, gameboard_size),
        'NW': (0, gameboard_size * 0.5, 0, gameboard_size * 0.5),
        'NE': (gameboard_size * 0.5, gameboard_size, 0, gameboard_size * 0.5),
        'SW': (0, gameboard_size * 0.5, gameboard_size * 0.5, gameboard_size),
        'SE': (gameboard_size * 0.5, gameboard_size, gameboard_size * 0.5, gameboard_size)
    }

    current_state = env.get_state()
    aircraft = env.agents[aircraft_id]
    aircraft_pos = (aircraft.x, aircraft.y)

    # First, check if we're trapped between threats
    nearby_threats = []
    for ship_id in current_state['ships']:
        ship = env.agents[ship_id]
        if ship.threat > 0 and ship.observed_threat:
            threat_radius = ship.width * env.AGENT_THREAT_RADIUS[ship.threat]
            dist_to_threat = math.hypot(aircraft.x - ship.x, aircraft.y - ship.y)

            # Consider threats that are relatively close
            if dist_to_threat < threat_radius * 2.5:
                nearby_threats.append((ship, dist_to_threat))

    # If multiple nearby threats, check if we need to escape
    if len(nearby_threats) >= 2:
        escape_point = find_escape_route(env,aircraft_pos, nearby_threats, gameboard_size)
        if escape_point:
            return escape_point, math.atan2(
                escape_point[1] - aircraft.y,
                escape_point[0] - aircraft.x
            )

    # If we're not trapped, proceed with normal target identification
    current_target_distances = {}
    for ship_id in current_state['ships']:
        ship = env.agents[ship_id]
        if ((id_type == 'target' and not ship.observed) or
                (id_type == 'wez' and (not ship.observed or not ship.observed_threat))):

            if (quadrant_bounds[quadrant][0] <= ship.x <= quadrant_bounds[quadrant][1] and
                    quadrant_bounds[quadrant][2] <= ship.y <= quadrant_bounds[quadrant][3]):
                dist = math.hypot(aircraft.x - ship.x, aircraft.y - ship.y)
                current_target_distances[ship_id] = dist

    # If no valid targets found, return to quadrant center
    if not current_target_distances:
        if quadrant == 'full':
            target_waypoint = (gameboard_size * 0.5, gameboard_size * 0.5)
        elif quadrant == 'NW':
            target_waypoint = (gameboard_size * 0.25, gameboard_size * 0.25)
        elif quadrant == 'NE':
            target_waypoint = (gameboard_size * 0.75, gameboard_size * 0.25)
        elif quadrant == 'SW':
            target_waypoint = (gameboard_size * 0.25, gameboard_size * 0.75)
        elif quadrant == 'SE':
            target_waypoint = (gameboard_size * 0.75, gameboard_size * 0.75)
    else:
        # Get closest valid target
        nearest_target_id = min(current_target_distances, key=current_target_distances.get)
        nearest_target = env.agents[nearest_target_id]
        target_waypoint = (nearest_target.x, nearest_target.y)

        # Check for threats along path to target
        for ship_id in current_state['ships']:
            ship = env.agents[ship_id]
            if ship.threat > 0 and ship.observed_threat:
                threat_radius = ship.width * env.AGENT_THREAT_RADIUS[ship.threat]
                ship_pos = (ship.x, ship.y)

                dist_to_threat = math.hypot(aircraft.x - ship.x, aircraft.y - ship.y)
                if dist_to_threat < threat_radius * 2:
                    if intersects_threat_ring(aircraft_pos, target_waypoint, ship_pos, threat_radius):
                        target_waypoint = calculate_avoidance_waypoint(
                            aircraft_pos,
                            target_waypoint,
                            ship_pos,
                            threat_radius,
                            gameboard_size
                        )

    target_direction = math.atan2(
        target_waypoint[1] - aircraft.y,
        target_waypoint[0] - aircraft.x
    )
    return target_waypoint, target_direction


def find_escape_route(env,aircraft_pos, nearby_threats, board_size):
    """
    Find a safe escape route when surrounded by multiple threats.
    Returns None if no escape route is needed.
    """
    # Calculate the center of mass of nearby threats
    threat_center_x = sum(threat[0].x for threat in nearby_threats) / len(nearby_threats)
    threat_center_y = sum(threat[0].y for threat in nearby_threats) / len(nearby_threats)

    # Check if we're actually trapped
    trapped = True
    for angle in range(0, 360, 45):  # Check 8 directions
        rad = math.radians(angle)
        test_point = (
            aircraft_pos[0] + math.cos(rad) * 100,  # Test point 100px away
            aircraft_pos[1] + math.sin(rad) * 100
        )

        # See if this direction is safe
        safe = True
        for threat, _ in nearby_threats:
            threat_radius = threat.width * env.AGENT_THREAT_RADIUS[threat.threat]
            if intersects_threat_ring(aircraft_pos, test_point, (threat.x, threat.y), threat_radius):
                safe = False
                break

        if safe:
            trapped = False
            break

    if not trapped:
        return None

    # We're trapped, find best escape route
    best_escape = None
    best_escape_score = float('-inf')

    # Try different escape angles
    for angle in range(0, 360, 15):  # Check more angles for better precision
        rad = math.radians(angle)

        # Try different distances
        for distance in [150, 200, 250]:  # Try multiple distances
            escape_x = aircraft_pos[0] + math.cos(rad) * distance
            escape_y = aircraft_pos[1] + math.sin(rad) * distance

            # Keep within board boundaries
            escape_x = max(35, min(board_size - 35, escape_x))
            escape_y = max(35, min(board_size - 35, escape_y))

            # Score this escape point:
            # - Higher score for points further from threat center
            # - Higher score for points closer to board center
            # - Lower score for points near any threat
            score = 0

            # Distance from threat center
            dist_from_threats = math.hypot(
                escape_x - threat_center_x,
                escape_y - threat_center_y
            )
            score += dist_from_threats * 2

            # Distance from board center
            dist_from_center = math.hypot(
                escape_x - board_size / 2,
                escape_y - board_size / 2
            )
            score -= dist_from_center

            # Check if path is safe
            path_safe = True
            for threat, _ in nearby_threats:
                threat_radius = threat.width * env.AGENT_THREAT_RADIUS[threat.threat]
                if intersects_threat_ring(
                        aircraft_pos,
                        (escape_x, escape_y),
                        (threat.x, threat.y),
                        threat_radius
                ):
                    path_safe = False
                    break

            if path_safe and score > best_escape_score:
                best_escape = (escape_x, escape_y)
                best_escape_score = score

    return best_escape or (board_size / 2, board_size / 2)  # Fall back to board center if no escape found


def intersects_threat_ring(start, end, center, radius):
    """Check if line segment from start to end intersects with threat circle."""
    dx = end[0] - start[0]
    dy = end[1] - start[1]

    cx = center[0] - start[0]
    cy = center[1] - start[1]

    path_length_sq = dx * dx + dy * dy

    if path_length_sq == 0:
        return math.hypot(cx, cy) <= radius

    t = max(0, min(1, (cx * dx + cy * dy) / path_length_sq))

    proj_x = start[0] + t * dx
    proj_y = start[1] + t * dy

    return math.hypot(center[0] - proj_x, center[1] - proj_y) <= radius + 20


def calculate_avoidance_waypoint(start, end, threat_pos, threat_radius, board_size):
    """Calculate waypoint that avoids threat ring while moving toward target."""
    dx = start[0] - threat_pos[0]
    dy = start[1] - threat_pos[1]

    dist = math.hypot(dx, dy)
    if dist == 0:
        dx, dy = 1, 0
    else:
        dx, dy = dx / dist, dy / dist

    perp_dx = -dy
    perp_dy = dx

    safe_dist = threat_radius + 40

    waypoint1 = (
        threat_pos[0] + dx * safe_dist + perp_dx * safe_dist,
        threat_pos[1] + dy * safe_dist + perp_dy * safe_dist
    )
    waypoint2 = (
        threat_pos[0] + dx * safe_dist - perp_dx * safe_dist,
        threat_pos[1] + dy * safe_dist - perp_dy * safe_dist
    )

    dist1 = math.hypot(waypoint1[0] - end[0], waypoint1[1] - end[1])
    dist2 = math.hypot(waypoint2[0] - end[0], waypoint2[1] - end[1])

    waypoint = waypoint1 if dist1 < dist2 else waypoint2

    margin = 35
    x = max(margin, min(board_size - margin, waypoint[0]))
    y = max(margin, min(board_size - margin, waypoint[1]))

    return (x, y)



# Defensive policy function suite (includes defensive_policy, find_safest_target, calculate_risk, approach_safely, and defensive_patrol
# Tends to get stuck between multiple threats
def defensive_policy(env, aircraft_id, quadrant='full', id_type='target'):
    """
    Defensive policy that prioritizes survival over efficiency.
    Only approaches targets when explicitly directed by human input.
    """
    aircraft = env.agents[aircraft_id]
    board_size = env.config["gameboard size"]

    # Get the list of dangerous areas to avoid
    danger_zones = []
    for agent in env.agents:
        if agent.agent_class == "ship":
            # For unidentified ships, assume maximum possible threat radius
            threat_radius = agent.width * env.AGENT_THREAT_RADIUS[3]
            # For identified hostile ships, use actual threat radius
            if agent.observed_threat and agent.threat > 0:
                threat_radius = agent.width * env.AGENT_THREAT_RADIUS[agent.threat]
            danger_zones.append((agent.x, agent.y, threat_radius))

    # Define safe margins
    SAFETY_MARGIN = 50  # Extra distance to keep from threat circles
    BOARD_MARGIN = env.config["gameboard border margin"] + 50  # Stay away from edges

    current_position = (aircraft.x, aircraft.y)

    # If in autonomous mode, be more aggressive
    if env.button_latch_dict['autonomous']:
        SAFETY_MARGIN = 20
        return autonomous_search(env, aircraft_id, danger_zones, SAFETY_MARGIN)

    # If in target ID or WEZ ID mode, approach carefully
    if env.button_latch_dict['target_id'] or env.button_latch_dict['wez_id']:
        target = find_safest_target(env, current_position, danger_zones, quadrant, id_type)
        if target:
            return approach_safely(env,current_position, target, danger_zones, SAFETY_MARGIN)

    # Default behavior: defensive patrolling
    return defensive_patrol(current_position, board_size, BOARD_MARGIN, danger_zones, SAFETY_MARGIN, quadrant)


def find_safest_target(env, current_pos, danger_zones, quadrant, id_type):
    """Find the target that can be observed with minimal risk."""
    safest_target = None
    min_risk = float('inf')

    for agent in env.agents:
        if agent.agent_class != "ship":
            continue

        # Skip if not in specified quadrant
        if not is_in_quadrant((agent.x, agent.y), quadrant, env.config["gameboard size"]):
            continue

        # Skip if already identified based on id_type
        if id_type == 'target' and agent.observed:
            continue
        if id_type == 'wez' and agent.observed_threat:
            continue

        # Calculate risk based on proximity to danger zones
        risk = calculate_risk((agent.x, agent.y), danger_zones)

        if risk < min_risk:
            min_risk = risk
            safest_target = (agent.x, agent.y)

    return safest_target

def calculate_risk(position, danger_zones):
    """Calculate risk level of a position based on proximity to threats."""
    risk = 0
    for x, y, radius in danger_zones:
        distance = math.hypot(position[0] - x, position[1] - y)
        if distance < radius:
            risk += 1000  # Heavily penalize being inside threat circles
        else:
            risk += 1 / (distance - radius)  # Add smaller risk for being close
    return risk

def approach_safely(env,current_pos, target_pos, danger_zones, safety_margin):
    """Find a safe path to approach the target."""
    # Try to find a safe observation point
    angles = list(range(0, 360, 45))  # Check 8 different angles
    best_point = None
    min_risk = float('inf')

    for angle in angles:
        rad = math.radians(angle)
        # Try different distances
        for dist in range(100, 300, 50):
            test_x = target_pos[0] + math.cos(rad) * dist
            test_y = target_pos[1] + math.sin(rad) * dist

            risk = calculate_risk((test_x, test_y), danger_zones)

            if risk < min_risk:
                min_risk = risk
                best_point = (test_x, test_y)

    if best_point:
        direction = math.atan2(best_point[1] - current_pos[1],
                               best_point[0] - current_pos[0])
        return best_point, direction

    # If no safe approach found, retreat
    return defensive_patrol(current_pos, env.config["gameboard size"],
                            env.config["gameboard border margin"] + 50,
                            danger_zones, safety_margin, 'full')

def defensive_patrol(current_pos, board_size, margin, danger_zones, safety_margin, quadrant):
    """Generate a defensive patrol pattern that avoids threats."""
    # Define patrol points based on quadrant
    if quadrant == 'full':
        patrol_points = [
            (margin, margin),
            (board_size - margin, margin),
            (board_size - margin, board_size - margin),
            (margin, board_size - margin)
        ]
    else:
        # Define smaller patrol pattern for specific quadrant
        half_size = board_size / 2
        if quadrant == 'NW':
            patrol_points = [
                (margin, margin),
                (half_size - margin, margin),
                (half_size - margin, half_size - margin),
                (margin, half_size - margin)
            ]
        elif quadrant == 'NE':
            patrol_points = [
                (half_size + margin, margin),
                (board_size - margin, margin),
                (board_size - margin, half_size - margin),
                (half_size + margin, half_size - margin)
            ]
        elif quadrant == 'SW':
            patrol_points = [
                (margin, half_size + margin),
                (half_size - margin, half_size + margin),
                (half_size - margin, board_size - margin),
                (margin, board_size - margin)
            ]
        else:  # SE
            patrol_points = [
                (half_size + margin, half_size + margin),
                (board_size - margin, half_size + margin),
                (board_size - margin, board_size - margin),
                (half_size + margin, board_size - margin)
            ]

    # Find the closest patrol point that's safe
    closest_safe_point = None
    min_distance = float('inf')

    for point in patrol_points:
        risk = calculate_risk(point, danger_zones)
        if risk < 0.1:  # Threshold for "safe enough"
            dist = math.hypot(point[0] - current_pos[0],
                              point[1] - current_pos[1])
            if dist < min_distance:
                min_distance = dist
                closest_safe_point = point

    if closest_safe_point:
        direction = math.atan2(closest_safe_point[1] - current_pos[1],
                               closest_safe_point[0] - current_pos[0])
        return closest_safe_point, direction

    # If no safe patrol point found, retreat to nearest corner
    corner = (margin, margin)  # Default to top-left
    min_risk = calculate_risk(corner, danger_zones)

    for point in [(margin, board_size - margin),
                  (board_size - margin, margin),
                  (board_size - margin, board_size - margin)]:
        risk = calculate_risk(point, danger_zones)
        if risk < min_risk:
            min_risk = risk
            corner = point

    direction = math.atan2(corner[1] - current_pos[1],
                           corner[0] - current_pos[0])
    return corner, direction

def autonomous_search(env, aircraft_id, danger_zones, safety_margin):
    """More aggressive search pattern when autonomous mode is enabled."""
    aircraft = env.agents[aircraft_id]
    current_pos = (aircraft.x, aircraft.y)

    # Find nearest unidentified target that can be approached safely
    target = find_safest_target(env, current_pos, danger_zones, 'full', 'wez')

    if target:
        # Calculate a closer approach than normal
        return approach_safely(env,current_pos, target, danger_zones, safety_margin)

    # If no safe targets found, maintain defensive patrol
    return defensive_patrol(current_pos, env.config["gameboard size"],
                            env.config["gameboard border margin"] + 30,
                            danger_zones, safety_margin, 'full')


def is_in_quadrant(pos, quadrant, board_size):
    """Check if position is in specified quadrant."""
    x, y = pos
    half_size = board_size / 2

    if quadrant == 'full':
        return True
    elif quadrant == 'NW':
        return x < half_size and y < half_size
    elif quadrant == 'NE':
        return x >= half_size and y < half_size
    elif quadrant == 'SW':
        return x < half_size and y >= half_size
    elif quadrant == 'SE':
        return x >= half_size and y >= half_size
    return False
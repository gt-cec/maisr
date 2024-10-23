"""def astar_policy(env, aircraft_id, target_waypoint):
    aircraft = env.agents[aircraft_id]
    start = (aircraft.x, aircraft.y)
    goal = target_waypoint

    def heuristic(a, b):
        return math.hypot(b[0] - a[0], b[1] - a[1])

    def get_neighbors(pos):
        x, y = pos
        neighbors = [
            (x + 30, y), (x - 30, y), (x, y + 30), (x, y - 30),
            (x + 21, y + 21), (x - 21, y + 21), (x + 21, y - 21), (x - 21, y - 21)
        ]
        return [(nx, ny) for nx, ny in neighbors if
                0 <= nx < env.config["gameboard size"] and 0 <= ny < env.config["gameboard size"]]

    def is_valid(pos):
        x, y = pos
        for ship in env.agents:
            if ship.agent_class == "ship" and ship.threat > 0:
                dist = math.hypot(ship.x - x, ship.y - y)
                if dist <= ship.width * env.AGENT_THREAT_RADIUS[ship.threat]:
                    return False
        return True

    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    while open_set:
        current = heapq.heappop(open_set)[1]

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        for neighbor in get_neighbors(current):
            if not is_valid(neighbor):
                continue

            tentative_g_score = g_score[current] + heuristic(current, neighbor)

            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return None  # No path found

def astar_waypoint_policy(env, aircraft_id, quadrant='full', id_type='target'):
    current_state = env.get_state()
    aircraft = env.agents[aircraft_id]

    # Use the existing target_id_policy to get the target waypoint
    target_waypoint, _ = target_id_policy(env, aircraft_id, quadrant, id_type)

    # Use A* to find a path to the target waypoint
    path = astar_policy(env, aircraft_id, target_waypoint)

    if path:
        # Return the next waypoint in the path
        next_waypoint = path[1] if len(path) > 1 else path[0]
        target_direction = math.atan2(next_waypoint[1] - aircraft.y, next_waypoint[0] - aircraft.x)
        return next_waypoint, target_direction
    else:
        # If no path is found, return the current position
        return (aircraft.x, aircraft.y), aircraft.direction
"""
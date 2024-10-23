import math


def hold_policy(env,aircraft_id,quadrant='full',id_type='target'):
    # Note: kwargs not currently used.
    target_waypoint = env.agents[aircraft_id].x, env.agents[aircraft_id].y
    target_direction = math.atan2(target_waypoint[1] - env.agents[aircraft_id].y,target_waypoint[0] - env.agents[aircraft_id].x)
    return target_waypoint, target_direction


def mouse_waypoint_policy(env,aircraft_id):
    # TODO Currently this is implemented in main.py. Might move it here.
    pass
    #return target_waypoint, target_direction
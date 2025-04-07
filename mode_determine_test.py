import numpy as np

def new_determine_agent_mode(gameplan_command_history):
    search_area = 'auto full'
    search_type = 'auto weapon'

    area_mode_history = []
    type_mode_history = []

    area_mode_history.append(search_area)
    type_mode_history.append(search_type)

    last_cmd_time = 0

    # TODO: Account for waypoint override ending

    for cmd in gameplan_command_history:
        time_diff = np.floor(cmd[0] - last_cmd_time)
        print(f'Looking at command {cmd} ({time_diff} seconds after previous command)')

        # If [priority_mode = waypoint override] in the gamestate timestamp closest to the command timestamp:
            # search_area = 'waypoint_override'
            # search_type = 'waypoint_override'

        if cmd[1] in ['NW', 'NE', 'SW', 'SE']:
            search_area = 'manual quadrant'
        elif cmd[1] == 'full':
            search_area = 'manual full'

        elif cmd[1] == 'autonomous':
            search_area = 'auto full'

        if cmd[1] == 'wez_id':
            search_type = 'manual weapon'
        elif cmd[1] == 'target_id':
            search_type = 'manual target'
        elif cmd[1] == 'autonomous':
            search_type = 'auto target'
        elif cmd[1] == 'waypoint override':
            search_type = 'waypoint override'
        elif cmd[1] == 'hold':
            search_type = 'hold'

        print(f'Appending modes for {time_diff} seconds')
        for i in range(time_diff):
            area_mode_history.append(search_area)
            type_mode_history.append(search_type)

        last_cmd_time = cmd[0]
        previous_was_waypoint = False

    return area_mode_history, type_mode_history, search_area


test_history = [[5.398, "NW"], [6.012, "target_id"], [95.625, "SW"], [158.993, "wez_id"], [177.855, "NW"], [5.546, "NW"], [6.864, "target_id"], [9.369, "wez_id"], [104.408, "NE"], [200.367, "SE"]]

new_determine_agent_mode(test_history)
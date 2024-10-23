"""
README:
To set up experiment run, first enter a subject_id and modify env_config depending on the experiment configuration:
show_agent_waypoint = 0,1,2, or 3
show_current_action = True/False
show_decision_rationale = True/False
show_Risk_info = True/False

"""

log_data = True  # Set to false if you don't want to save run data to a json file
subject_id = '001'

# environment configuration, use this for the gameplay parameters
env_config = {
    "gameboard size": 700, # NOTE: The rest of the GUI doesn't dynamically scale with different gameboard sizes. Stick to 700 for now
    "num aircraft": 2,  # supports any number of aircraft, colors are set in env.py:AIRCRAFT_COLORS (NOTE: Many aspects of the game currently only support 2 aircraft
    "gameplay color": "white",
    "gameboard border margin": 35,
    "targets iteration": "D",
    "motion iteration": "F",
    "search pattern": "ladder",
    "verbose": False,
    "window size": (1800,850), # width,height
    'show agent waypoint':1, # For SA-based agent transparency study TODO change to 0, 1, 2, 3
    'show agent location':'persistent', # For SA-based agent transparency. 'persistent', 'spotty', 'none' TODO not implemented
}

# Configure what information the agent shows the human (for SA-based transparency study
agent_config = {
    'show_low_level_goals' : True,
    'show_high_level_goals' : True,
    'show_high_level_rationale' : True,
    'show_tracked_factors' : True
}
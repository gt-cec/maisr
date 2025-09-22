# MAISR: Multi-Agent Intelligence, Surveillance, and Reconnaissance

This is the codebase for the Multi-Agent Intelligence, Surveillance, and Reconnaissance (MAISR) environment for multi-agent reinforcement learning training, including teamwork with human players. This version of the environment supports RL training and deployment and is compliant with the gymnasium API.

This codebase is a PyGame variant of the [CEC ISR domain](https://github.com/gt-cec/onr-isr). This branch is the version of the MAISR environment used in the "Generalizing to Human Teammates in Multi-Agent Reinforcement Learning By Modeling Human Behavior Styles" in (September 2025), submitted for publication in the IEEE ICRA 2026 conference.



# Quick Start

## Play the env as a human

## Train new agents
Run ``train_ppo.py`` to train a new policy using PPO. 

## Run the env with 1 human + 1 loaded agents


## Run the env with 2 loaded agents


# Observations, Actions, Rewards
## Observation spaces
``nearest``: 

* Observation is a vector containing the relative x and y distance to nearest config[num_observed_targets] unknown targets, nearest config[num_observed_threats] unknown threats,  and optionally the position of the teammate (agent 1) and the teammate's current goal (whether it is heading towards a target or a threat)

* Size: self.obs_size = 2 * [num_observed_targets] + 2 * [num_observed_threats] + (2 if [observe_teammate]) + (2 if [observe_teammate_direction]) + (1 if [observe_teammate_priority])

``full``: 
* Observation is a vector containing the relative x and y distance to all unknown targets and all unknown threats,  and optionally the position of the teammate (agent 1) and the teammate's current goal (whether it is heading towards a target or a threat)
* Size: self.obs_size = 2 * [num_targets] + 2 * [num_threats] + (2 if [observe_teammate]) + (2 if [observe_teammate_direction]) + (1 if [observe_teammate_priority])

``pixel``: TODO


## Action spaces
``Discrete8``

``Discrete16``

``continuous-normalized``

``target_index``


## Rewards
The reward function is defined in ``base_env.py:get_reward()``

# Folder sctructure
env_wrapper: Wraps the environment in an additional layer to handle the second agent, 
base_env: The main gymnasium environment that defines the core game mechanics


# To modify the environment
## Add more agents

## Change game dynamics: Edit env.step()

## Modify the heuristics

## Modify the teammate league


# Useful code snippets
## Instantiate the environment 


# Tips
Observations are normalized by default in ``train.py``. When you run the agent during evaluation or deployment, you must load the correct .pkl file that was saved from the VecNormalize env wrapper.

# Recommended hyperparameters

# Old stuff to fold in

### How to play
The dark blue aircraft is the "human", controlled by clicking on the map to command waypoints.

The light blue aircraft is the agent, which follows a simple rule-based policy (fly towards the nearest eligible unknown target based on gameplan constraints) and obeys commands set by the human using the buttons to the right of the gameboard.

(TODO CHECK) Press F1 at any time to exit the game. Press space to pause, and right click to unpause.


### Configuration
The majority of the game's configurable parameters are stored in json files inside the ./config_files/ folder. 

TODO Describe the config parameters

Aspects of the game:
* `num ships`: How many targets are spawned.
* `time limit`: How long the player has to complete the scenario. Default 120 (seconds)
* `game speed`: The speed multiplier for the aircraft. Default = 0.2. 
* `agent speed`: Agent aircraft's speed
* `human speed`: Human aircraft's speed
* `seed`: Determines target location, type, and threat class

For debugging:
* `verbose`: `True` or `False`, for printing state information during the game
* `infinite health`: `True` or `False`

Many other aspects of the game can be modified inside `env.py`, main.py, agents.py, etc.

### Data logging
(TODO) If log_data is set to True in `/config.py`, the game will automatically create a .jsonl file that logs the game state every 5 seconds. subject_id (set in config.py) is automatically included in the log filename for easy identification later. 

### Data analysis
(TODO) The .jsonl log file generated from a game round contains all the information required to completely reconstruct the round (at least in 5 second increments). You can do this visually using `isr_playback.py`. 

You can also use isr_gamedata_processor.py to convert the gamedata into a tidyverse-format excel sheet where each row is one participant's game data, and each column is a feature from that participant's game data. This script is currently hardcoded to require 5 .jsonl files per participant, corresponding to rounds 0 (training) and 1-4. You will need to modify the script if you want to process different numbers or arrangements of rounds.

# File organization

## Main files

``train_ppo.py``

``base_env.py``
* Processes and renders the game environment

``utility/localsearch_training_wrapper.py``
* Wrapper around the base env for training. Mainly exists to manage the teammate for two-player gameplay. Could be refactored to absorb these functions into base_env in the future.


## Utilities:
``league_management.py``
Contains these classes: 
* LeagueManager: Manages the league of teammates experienced by the ego agent during training.
* Heuristic

``agents.py``
* Defines the aircraft that are controlled by humans and agents. The agents themselves are defined elsewhere

``env_checker.py``


## Play scripts
``play_2agents.py``, ``play_agents_with_human.py``, 


(TODO) #### `main.py` 
* Initializes pygame and handles the main game loop.
* Starts the data logging function for saving experiment data
* Handles each aircraft's actions (The agent, AKA agent_0, has a policy that is called at the beginning of each game step. The human is controlled via mouse click event handler.)
* Handles GUI events such as button clicks (much of this code will eventually move into gui.py)
  
#### `env.py`

* Renders the GUI (this may move into a class inside gui.py later)

#### `agents.py`
 

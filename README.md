# MAISR: Multi-Agent Intelligence, Surveillance, and Reconnaissance
This codebase is a PyGame variant of the [CEC ISR domain](https://github.com/gt-cec/onr-isr). This branch is the version of the MAISR environment used in the team familiarization study in January 2025, submitted for publication in the IEEE RO-MAN 2025 conference.

### Quick Start
The environment can be run in one of two ways:
1. Set desired parameters in `config.py` (or leave the defaults), then run `main.py`. We recommend this method if you just want to test out the environment.
   
2. Alternatively, `run.bat` has been set up to streamline the loading process during user studies. When this script is run, it asks for the participant's ID, the study condition, starting round (0-4) and whether to log data. These parameters are used to initialize the game rather than the parameters in config.py (To use this method, first modify line 20 in `run.bat` to point to your python environment's activate.bat file. Then run run.bat from the terminal.

### How to play
The dark blue aircraft is the "human", controlled by clicking on the map to command waypoints.

The light blue aircraft is the agent, which follows a simple rule-based policy (fly towards the nearest eligible unknown target based on gameplan constraints) and obeys commands set by the human using the buttons to the right of the gameboard.

Press F1 at any time to exit the game. Press space to pause, and right click to unpause.


### Configuration
The majority of the game's configurable parameters are stored in json files inside the ./config_files/ folder. 

./config.py is used to set the subject ID (for experiments) and the config filename to load.

#### The following features are currently configurable:
Aspects of the game:
* `num ships`: How many targets are spawned. 30 seems to be a good setting.
* `time limit`: How long the player has to complete the scenario. Default 120 (seconds)
* `game speed`: The speed multiplier for the aircraft. Default = 0.2. 
* `agent speed`: Agent aircraft's speed
* `human speed`: Human aircraft's speed
* `seed`: Determines target location, type, and threat class

For debugging:
* `verbose`: `True` or `False`, for printing state information during the game
* `infinite health`: `True` or `False`

The following features are technically configurable using config files but modification is not supported (things will break:)
* `gameboard size`: pixel dimension of the gameboard (square board) 
* `window size`: The size of the full application window (gameboard + other GUI elements)
* `num aircraft`: number of aircraft. Currently 

The following features are from the original ISR application and are not used:
* `gameplay color`: one of `red, yellow`, as in ONR-ISR
* `motion iteration`: one of `F, G, H, I, J`, as in ONR-ISR
* `search pattern`: one of `square, ladder, hold`, as in ONR-ISR

Many other aspects of the game can be modified inside `env.py`, main.py, agents.py, or autonomous_policy.py.

### Data logging
If log_data is set to True in `/config.py`, the game will automatically create a .jsonl file that logs the game state every 5 seconds. subject_id (set in config.py) is automatically included in the log filename for easy identification later. 

### Data analysis
The .jsonl log file generated from a game round contains all the information required to completely reconstruct the round (at least in 5 second increments). You can do this visually using `isr_playback.py`. 

You can also use isr_gamedata_processor.py to convert the gamedata into a tidyverse-format excel sheet where each row is one participant's game data, and each column is a feature from that participant's game data. This script is currently hardcoded to require 5 .jsonl files per participant, corresponding to rounds 0 (training) and 1-4. You will need to modify the script if you want to process different numbers or arrangements of rounds.

### File organization
The primary files are:

#### `main.py` 
* Initializes pygame and handles the main game loop.
* Starts the data logging function for saving experiment data
* Handles each aircraft's actions (The agent, AKA agent_0, has a policy that is called at the beginning of each game step. The human is controlled via mouse click event handler.)
* Handles GUI events such as button clicks (much of this code will eventually move into gui.py)
  
#### `env.py`
* Processes and renders the game environment
* Renders the GUI (this may move into a class inside gui.py later)

#### `agents.py`
* Defines the aircraft and ship agents.
* NOTE: Agent policies are currentlyhandled inside autonomous_policy.py, not agents.py. This may change in the future.

#### `autonomous_policy.py`
* Defines the policy used by the agent aircraft

#### `gui.py`
* Defines many of the GUI elements, such as gameplan buttons, the agent health windows, the score printout, etc.

### Integrating Reinforcement Learning
This version of the environment is nearly set up for reinforcement learning training. `main.py` and `env.py` follow typical reinforcement learning environment formats and naming conventions, but the environment does not completely follow gymnasium conventions for observations and actions. We plan to update this environment to be fully gymnasium-compliant and ready for off-the-shelf RL training during summer 2025.

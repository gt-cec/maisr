# MAISR: Multi-Agent Intelligence, Surveillance, and Reconnaissance

This codebase is a PyGame variant of the [CEC ISR domain](https://github.com/gt-cec/onr-isr). This allows for exploring multi-agent ISR and training reinforcement learning policies at a far higher framerate than is possible with the web browser domain.

### Quick Start

To run the environment with a PyGame visualization:

`python main.py`

To run the environment in headless mode:

`python main.py headless`

Note that the PyGame visualization is only a visualization of the state space, so the `headless` mode allows for (very fast) simulations for policy training.

### How to play
The dark blue aircraft is the "human", controlled by clicking on the map to command waypoints.

The light blue aircraft is the agent, which follows a simple rule-based policy (fly towards the nearest eligible unknown target based on gameplan constraints) and follows search constraints set by the human using the buttons to the right of the gameboard.

### Configuration

The majority of the game's configurable parameters are stored in json files inside the ./config_files/ folder. ./config.py is used to set the subject ID (for experiments) and the config filename to load.

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
If log_data is set to True in ./config.py, the game will automatically create a .jsonl file that logs the game state every 5 seconds. subject_id (set in config.py) is automatically included in the log filename for easy identification later.

### File organization
Currently the game files are not organized according to best coding practices (this is left as future work). But the main files are:

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
* NOTE: Agent policies are currently not handled inside agents.py. This may change in the future.

#### `autonomous_policy.py`
* Defines the policy used by the agent aircraft

#### `gui.py`
* defines many of the GUI elements, such as gameplan buttons, the agent health windows, the score window, and the agent status window.

### Integrating Reinforcement Learning

Virtually all of the basic environment setup is complete. `main.py` and `env.py` follow typical reinforcement learning environment formats and naming conventions.

See `main.py` for comments indicating where you can integrate a custom policy. The codebase should be ready for any tutorial on reinforcement learning with only slight modifications.

The state space's reward function can be set in `env.py:get_reward()`.

Using gym's observation and state space definitions is not integrated, however a starting point is given in `env.py:action_space` and `env.py:observation_space`.
# MAISR: Multi-Agent Intelligence, Surveillance, and Reconnaissance

This codebase is a PyGame variant of the [CEC ISR domain](https://github.com/gt-cec/onr-isr). This allows for exploring multi-agent ISR and training reinforcement learning policies at a far higher framerate than is possible with the web browser domain.

### Quick Start

To run the environment with a PyGame visualization:

`python main.py`

To run the environment in headless mode:

`python main.py headless`

Note that the PyGame visualization is only a visualization of the state space, so the `headless` mode allows for (very fast) simulations for policy training.

### Configuration

There are several areas of configuration:

`main.py` has a game configuration dictionary similar to the ONR-ISR domain:

* `gameboard size`: pixel dimension of the gameboard (square board)
* `num aircraft`: number of aircraft, supports any number of aircrafts
* `gameplay color`: one of `red, yellow`, as in ONR-ISR
* `targets iteration`: one of `A, B, C, D, E`, as in ONR-ISR
* `motion iteration`: one of `F, G, H, I, J`, as in ONR-ISR
* `search pattern`: one of `square, ladder, hold`, as in ONR-ISR
* `verbose`: one of `True, False`, for printing state information during the game

`env.py` has environment parameters such as agent scale and drawing parameters, agent colors, aircraft scanner and visual radii, and the search pattern flight plans.

### Integrating Reinforcement Learning

Virtually all of the basic environment setup is complete. `main.py` and `env.py` follow typical reinforcement learning environment formats and naming conventions.

See `main.py` for comments indicating where you can integrate a custom policy. The codebase should be ready for any tutorial on reinforcement learning with only slight modifications.

The state space's reward function can be set in `env.py:get_reward()`.

Using gym's observation and state space definitions is not integrated, however a starting point is given in `env.py:action_space` and `env.py:observation_space`.

# Summary of Ryan's changes
* Added "target_id_policy" agent policy that sets the agent's waypoint to the nearest unknown target (incomplete)
* Added "mouse_waypoint_policy" that sets agent's waypoint to mouse click location (incomplete)
* Made gameboard green buffer configurable within env_config.
* (More not listed)
* Added damage system 
* Agents get destroyed when damage > 100 (currently a little messy, various functions check for the aircraft.alive Bool)

### 10/20/2024
* Fixed bug where aircraft were taking damage when within ID range, not within ship threat range
* Fixed time incrementing incorrectly while paused
* Comm log complete
* Basic but flawed A* search policy added to avoid WEZs

### 10/21/2024
* GUI rearranged
* Agent status window added
* Score system complete
* Data logging system written
* Made agent's waypoint line optional (for SA-based transparency study)
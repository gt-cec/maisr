# MAISR: Multi-Agent Intelligence, Surveillance, and Reconnaissance

This is the codebase for the Multi-Agent Intelligence, Surveillance, and Reconnaissance (MAISR) environment for multi-agent reinforcement learning training, including teamwork with human players. This version of the environment supports RL training and deployment and is compliant with the gymnasium API.

This codebase is a PyGame variant of the [CEC ISR domain](https://github.com/gt-cec/onr-isr). This branch is the version of the MAISR environment used in the "Generalizing to Human Teammates in Multi-Agent Reinforcement Learning By Modeling Human Behavior Styles" in (September 2025), submitted for publication in the IEEE ICRA 2026 conference.

# Requirements
**Core:**
- Python 3.12
- Pytorch
- Gymnasium
- Pygame

**Can be replaced/removed:**
- Stable-Baselines3 (Not needed if using another RL library)
- WandB, Tensorboard (logging)
- Matplotlib, Pandas (for data logging and analysis)
- Flask, socketio (for remote user studies)

# Quick Start

## Train a new agent
Run ``train_ppo.py`` to train a new policy in the MAISR environment using PPO. You can run without arguments to train 
using default settings, but the script is also set up to allow modifying hyperparameters, reward function elements, the 
contents and schedule of the teammate training pool, various callbacks for WandB logging and evaluation, etc. It also 
allows you to set up multiple training types that can be accessed using the --version arg.

## Play the env
Run ``play_env.py`` to run the environment with two aircraft. You can modify the script to change who controls each 
aircraft: A human player (left-click the map to set waypoints), a pretrained RL policy, or a heuristic agent.  

## Run a user study
Run ``experiments/exp2_user_study/run_userstudy.py``. The current version of the script runs a flask server for remote studies, but can be reworked to run locally without much effort.


# File organization

## Main files
``train_ppo.py``: Primary training script.

``base_env.py``: Defines the main MAISR environment based on the gymnasium API.

## Utilities:
``/human_trajectories_for_training/``: Folder containing recorded human trajectories that can be instantiated as 
teammates during agent evaluation in training.

``agents.py``: Defines the aircraft that are controlled by humans and agents. The agents themselves are defined elsewhere.

``env_checker.py``: Contains some tests to make sure the environment works and to sanity-check different reward settings. Old and may not work out of the box.

``league_management.py``:
Defines the ``TeammateManager`` that controls which teammates the learning agent experiences during training. Also 
defines RL teammates and the `ConfigurableHeuristicTeammate` used in the September 2025 paper. 

``level_layouts.json``: Defines specific level layouts consisting of positions of regular and high-value targets and the starting positions of the two agents. 

``utility/localsearch_training_wrapper.py``: Wrapper around the base env for training. Mainly exists to manage the teammate for two-player gameplay. Could be refactored to absorb these functions into base_env in the future.

``server.py`` and ``sockets.py``: Used for remote user studies.


## Folders
``/configs/``: Contains .json files for configuring the environment's mechanics, training hyperparameters, curriculum learning schedules, action and observation types, reward function elements, and more.

``/experiments/``: Scripts and data from the experiments discussed in the paper.


# Observations, Actions, Rewards
## Observation spaces

### Option 1: `nearest` (Recommended)
Observation is a vector containing the relative x and y distance to the nearest N unknown targets and M unknown threats, plus optional teammate information.

**Features included:**
- For each of the N nearest unknown targets: `[dx, dy]` - relative position vector to target
- For each of the M nearest unknown threats: `[dx, dy]` - relative position vector to threat  
- If `observe_teammate` is True: `[dx, dy]` - relative position of teammate
- If `observe_teammate_direction` is True: `[heading_x, heading_y]` - teammate's movement direction
- If `observe_teammate_priority` is True: `[0 or 1]` - whether teammate is flying toward a threat (1) or target (0)

**Size:** `2 * num_observed_targets + 2 * num_observed_threats + (2 if observe_teammate) + (2 if observe_teammate_direction) + (1 if observe_teammate_priority)`

**Example:** With `num_observed_targets=3`, `num_observed_threats=1`, and all teammate observations enabled, the observation vector would be size 11 and contain: `[target1_dx, target1_dy, target2_dx, target2_dy, target3_dx, target3_dy, threat1_dx, threat1_dy, teammate_dx, teammate_dy, teammate_heading_x, teammate_heading_y, teammate_priority]`

### Option 2: `full`
Observation is a vector containing information about ALL targets and threats in the environment, regardless of distance. In our experience, the agent struggles to learn an effective policy with an observation this large, but we include it in case others would like to do more testing.

**Features included:**
- Agent position: `[agent_x, agent_y]` - normalized agent position (-1 to +1)
- For each target: `[info_level, target_x, target_y]` - identification status (0=unknown, 1=identified) and normalized position
- For each threat: `[threat_x, threat_y]` - normalized position
- Optional teammate information (same as `nearest` mode)

**Size:** `2 + 3 * num_targets + 2 * num_threats + (2 if observe_teammate) + (2 if observe_teammate_direction) + (1 if observe_teammate_priority)`

### Option 3: `pixel` 
Observation is a grayscale 84x84 pixel image suitable for CNN-based agents. We have not tested this mode extensively, but encourage others to do so.

**Visual elements:**
- Background: white (255)
- Aircraft: dark gray (100) 
- Unknown targets: dark gray (50)
- Identified targets: light gray (200)
- Threats: medium gray (150) with triangle markers

**Size:** `(84, 84, 1)` - Height, Width, Channels (grayscale)

## Action spaces

### Option 1: `Discrete8`
8 discrete movement directions corresponding to compass directions.
- Actions 0-7 map to: N, NE, E, SE, S, SW, W, NW. Each action moves the agent a fixed distance (50 pixels) in the chosen direction
- **Action space:** `gym.spaces.Discrete(8)`

### Option 2: `Discrete16` (Recommended)
16 discrete movement directions for finer control.
- Actions 0-15 map to directions in 22.5° increments (N, NNE, NE, ENE, E, ESE, SE, SSE, S, SSW, SW, WSW, W, WNW, NW, NNW). Each action moves the agent a fixed distance (50 pixels) in the chosen direction
- **Action space:** `gym.spaces.Discrete(16)`

### Option 3: `continuous-normalized`
Continuous 2D movement with normalized coordinates.
- Action is `[x, y]` where both values are in range `[-1, 1]`
- Actions are denormalized to map coordinates: `actual_coord = action * map_half_size`
- Allows precise positioning anywhere within the map bounds
- **Action space:** `gym.spaces.Box(low=[-1, -1], high=[1, 1], dtype=np.float32)`

### Option 4:`target_index`
Discrete action space where each action corresponds to flying toward a specific target or threat. We find that this action type does not present enough of a challenge to the agent for us to see meaningful performance differences.
- Actions 0 to `num_observed_targets-1`: fly toward the Nth nearest unknown target
- Actions `num_observed_targets` to `num_observed_targets + num_observed_threats - 1`: fly toward the Nth nearest unknown threat
- If the selected target/threat doesn't exist, agent stays at current position
- **Action space:** `gym.spaces.Discrete(num_observed_targets + num_observed_threats)`

**Example:** With `num_observed_targets=3` and `num_observed_threats=1`:
- Action 0: fly to nearest unknown target
- Action 1: fly to 2nd nearest unknown target  
- Action 2: fly to 3rd nearest unknown target
- Action 3: fly to nearest unknown threat

## Rewards

The reward function is defined in `base_env.py:get_reward()` and includes:

**Positive rewards:**
- Target identification: `base_env_target_id_reward` points per target identified
- Threat identification: `threat_id_reward` points per threat identified (up to `max_threat_ids`)
- Early completion: `shaping_coeff_earlyfinish` × remaining steps if all objectives completed
- Potential-based shaping: rewards for moving closer to unknown targets/threats
- Team coordination: spread bonus for maintaining optimal distance between teammates

**Negative rewards:**
- Proximity penalty: for getting too close to teammate
- Excess threat identification: penalty for identifying more threats than `max_threat_ids`
- Detection: penalties when agent is detected by threats (if `prob_detect > 0`)

**Key parameters:**
- `base_env_target_id_reward`: Base reward for identifying targets
- `threat_id_reward`: Reward for identifying threats  
- `teammate_reward_scale`: Multiplier for rewards earned by teammate
- `target_potential_coeff`: Weight for target approach shaping
- `threat_potential_coeff`: Weight for threat approach shaping


# To modify the environment

## Change game dynamics
Most game mechanics are defined in ``base_env:_single_step()`` and can be changed as needed. 

## Modify the heuristic teammates
The heuristic agents' configurable policies are defined in ``league_management.py``. The primary class is 
``ConfigurableHeuristicTeammate``, which can be configured manually or procedurally using the ``TeammateManager`` class.   
The ``ConfigurableHeuristicTeammate`` uses subpolicies (LocalSearch, GoToThreat, etc.) and the ``PolicySelector`` to choose between them.

All of these subpolicies and the rules for their selection are configurable based on 5 dimensions: ``risk_tolerance``, 
``spatial_coord``(ination), ``decision_speed``, ``action_stability``, and ``planning_horizon``. Each dimension has 
several options that change how certain subpolicies behave. You are encouraged to modify these to induce new behaviors.

You can also modify the ``TeammateManager`` class to change which teammates are included in the pool, such as to weight
some heuristics more than others or to mix in self-play or pretrained RL teammates.

## Other modifications
- Change agent aircraft appearance: Edit aircraft.draw() in ``agents.py``

# Tips
Observations are normalized by default in ``train_ppo.py``. When you run the agent during evaluation or deployment, you must load the correct .pkl file that was saved from the VecNormalize env wrapper.

# Recommended hyperparameters
See /configs/main_config.json for the hyperparameters we used in the paper. More suggestions are coming soon.


# Future work
Coming soon

# Citation
If you use this repository in your work, please cite:

(Coming soon)
# MAISR Codebase Overview

This document summarizes the purpose of each file in the codebase.

### run_experiment.py
Handles the execution of user study episodes in the MAISR environment. Manages pygame rendering, countdowns, progress bars, bottom info bar, human subpolicy control (keyboard and mouse), agent actions, and episode logging through `ExperimentDataLogger`. Integrates with a SocketIO server for streaming frames.

### instructional_screens.py
Defines various pygame-based instructional and survey screens for the user study. Includes welcome, instructions, between-episode, workload survey (NASA-TLX style), and teammate preference survey screens with mouse/keyboard interactivity.

### rl_data_logger.py
Implements `ExperimentDataLogger` for saving user study data. Logs timestep-level data, events, and episode summaries to JSON. Manages session data, survey responses, and teammate survey results, with safe resume functionality.

### train_generic.py
Provides a generic training script for MAISR agents using Stable-Baselines3. Includes utility functions for finding latest checkpoints, a league type transition callback, and a comprehensive WandB logging callback with evaluation and curriculum learning support.

### agents.py
Defines `Agent` and `Aircraft` classes used in the MAISR environment. Handles movement, waypoint overrides, and custom drawing logic in pygame, including visual markers for different aircraft appearances.

### league_management.py
Implements `TeammateManager` and various teammate policy classes (heuristic and RL-based). Manages league types (selfplay, strategy-diverse, mixed, FCP), checkpoint loading, normalization stats, and heuristic teammate behavior for MARL training.

### localsearch_training_wrapper.py
Provides `MaisrLocalSearchWrapper`, a Gym environment wrapper for training mode selector policies. Adds teammate management, observation processing, action space configuration, stuck-agent detection with override behavior, and teammate action logic.

### env_multi_new.py
Defines `MAISREnvVec`, the core multi-agent ISR environment for MARL training and user studies. Implements Gym API with step/reset, reward shaping, curriculum difficulty, threat/target placement, agent movement, scoring, and pygame-based rendering.


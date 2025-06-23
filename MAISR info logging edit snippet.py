################################################################################################
In env.step:

def step(self, action):
    """ Skip frames by repeating the action multiple times """
	
	...
	
	

    total_reward, info, total_potential_gain = 0, None, 0
    
    # Track status from prior step for info gathering later
    prev_targets_identified = self.targets_identified
    prev_threats_identified = self.num_threats_identified
    prev_detections = self.detections

    # Initialize consolidation containers
    consolidated_new_identifications = []
    consolidated_score_breakdown = {
        "target_points": 0, 
        "threat_points": 0, 
        "time_points": 0, 
        "completion_points": 0, 
        "penalty_points": 0
    }
    consolidated_reward_components = {}
    final_info = None

    for frame in range(self.config['frame_skip']):
        observation, reward, self.terminated, self.truncated, info = self._single_step(action)
        total_reward += reward
        total_potential_gain += info["potential_gain"]
        
        # Accumulate new identifications
        if "new_identifications" in info:
            consolidated_new_identifications.extend(info["new_identifications"])
        
        # Accumulate score breakdown
        if "score_breakdown" in info:
            for key, value in info["score_breakdown"].items():
                consolidated_score_breakdown[key] += value
        
        # Accumulate reward components
        if "reward_components" in info:
            for key, value in info["reward_components"].items():
                consolidated_reward_components[key] = consolidated_reward_components.get(key, 0) + value
        
        # Keep the final info as base
        final_info = info.copy()

        # Break early if the episode is done to avoid unnecessary computation
        if self.terminated or self.truncated:
            break

    self.step_count_outer += 1
    
    # Update final info with consolidated data
    final_info["outerstep_potential_gain"] = total_potential_gain
    final_info['new_target_ids'] = self.targets_identified - prev_targets_identified
    final_info['new_threat_ids'] = self.num_threats_identified - prev_threats_identified
    final_info['new_detections'] = self.detections - prev_detections
    
    # Add consolidated data
    final_info["new_identifications"] = consolidated_new_identifications
    final_info["score_breakdown"] = consolidated_score_breakdown
    final_info["reward_components"] = consolidated_reward_components
    final_info["frame_skip_steps"] = frame if self.terminated or self.truncated else self.config['frame_skip']

    return observation, total_reward, self.terminated, self.truncated, final_info
	
	
################################################################################################
################################################################################################
################################################################################################

In wrapped env.step:

def step(self, action: np.int32):
    
	# ... existing code for subpolicy selection ...
	

    ############################################ Step the environment ############################################

    # Initialize accumulation variables for action_rate consolidation
    macrostep_reward = 0
    total_new_target_ids = 0
    total_new_threat_ids = 0
    total_new_detections = 0
    consolidated_new_identifications = []
    consolidated_score_breakdown = {
        "target_points": 0, 
        "threat_points": 0, 
        "time_points": 0, 
        "completion_points": 0, 
        "penalty_points": 0
    }
    consolidated_reward_components = {}
    final_base_info = None
    
    # Track initial state for wrapper reward calculation
    initial_targets_identified = self.env.targets_identified
    initial_threats_identified = self.env.num_threats_identified
    initial_detections = self.env.detections
    
    # Execute action_rate steps
    for action_step in range(self.action_rate):
        base_obs, base_reward, base_terminated, base_truncated, base_info = self.env.step(subpolicy_action)
        macrostep_reward += base_reward
        
        # Accumulate new achievements
        total_new_target_ids += base_info.get('new_target_ids', 0)
        total_new_threat_ids += base_info.get('new_threat_ids', 0)
        total_new_detections += base_info.get('new_detections', 0)
        
        # Accumulate new identifications
        if "new_identifications" in base_info:
            consolidated_new_identifications.extend(base_info["new_identifications"])
        
        # Accumulate score breakdown
        if "score_breakdown" in base_info:
            for key, value in base_info["score_breakdown"].items():
                consolidated_score_breakdown[key] += value
        
        # Accumulate reward components
        if "reward_components" in base_info:
            for key, value in base_info["reward_components"].items():
                consolidated_reward_components[key] = consolidated_reward_components.get(key, 0) + value
        
        # Keep final info as base for episode tracking
        final_base_info = base_info.copy()
        
        # Break early if episode ends
        if base_terminated or base_truncated:
            break

    # Create consolidated info for wrapper reward calculation
    wrapper_reward_info = {
        'new_target_ids': total_new_target_ids,
        'new_threat_ids': total_new_threat_ids, 
        'new_detections': total_new_detections,
        'steps_left': final_base_info.get('steps_left', 0),
        'done': final_base_info.get('done', False),
        'failed': final_base_info.get('failed', False)
    }

    # Convert base_env elements to wrapper elements
    observation = self.get_observation(0)
    reward = self.get_reward(wrapper_reward_info)  # Use consolidated info
    self.episode_reward += reward  # Track wrapper reward, not base reward
    
    # Build final info dictionary
    info = final_base_info.copy()
    
    # Add wrapper-specific consolidated data
    info['policy_switches'] = self.total_switches
    info['final_subpolicy'] = self.subpolicy_choice
    info['threat_ids'] = self.env.num_threats_identified
    
    # Add accumulated achievements
    info['wrapper_new_target_ids'] = total_new_target_ids
    info['wrapper_new_threat_ids'] = total_new_threat_ids
    info['wrapper_new_detections'] = total_new_detections
    info['wrapper_new_identifications'] = consolidated_new_identifications
    info['wrapper_score_breakdown'] = consolidated_score_breakdown
    info['wrapper_reward_components'] = consolidated_reward_components
    info['action_rate_steps'] = action_step + 1 if base_terminated or base_truncated else self.action_rate
    
    # Update episode tracking with wrapper values
    info['episode'] = {
        'r': self.episode_reward,  # Wrapper episode reward
        'l': self.env.step_count_outer  # Environment step count
    }
    
    terminated = base_terminated
    truncated = base_truncated

    # ... rest of existing code ...
	
	
################################################################################################
################################################################################################
################################################################################################

In wrapper:

def get_reward(self, info):
    """Generate reward for the mode selector using consolidated info from action_rate steps"""
    
    target_reward = info.get('new_target_ids', 0) * self.reward_per_target_id
    threat_reward = info.get('new_threat_ids', 0) * self.reward_per_threat_id
    finish_reward = info.get('steps_left', 0) * self.reward_per_step_early if info.get('done', False) else 0
    fail_penalty = self.fail_penalty if info.get('failed', False) else 0
    switch_penalty = -self.switched_policies * self.penalty_for_policy_switch  # Negative penalty
    detect_penalty = -info.get('new_detections', 0) * self.penalty_per_detection  # Negative penalty

    total_reward = target_reward + threat_reward + finish_reward + fail_penalty + switch_penalty + detect_penalty
    
    # Optional: Add detailed reward breakdown to info for debugging
    reward_breakdown = {
        'target_reward': target_reward,
        'threat_reward': threat_reward, 
        'finish_reward': finish_reward,
        'fail_penalty': fail_penalty,
        'switch_penalty': switch_penalty,
        'detect_penalty': detect_penalty,
        'total': total_reward
    }
    
    return total_reward
class TargetSearchClusters(SubPolicy):
    """
    Sub-policy that optimizes for early target acquisition using a discount factor.
    Instead of minimizing total distance, this maximizes discounted target value
    where earlier targets are worth more.
    """

    def __init__(self, search_radius=200, spatial_coord=False, discount_factor=0.9,
                 model_path: str = None, norm_stats_filepath: str = None):
        super().__init__("target_search_clusters")
        self.search_radius = search_radius
        self.spatial_coord = spatial_coord
        self.discount_factor = discount_factor  # How much to discount later targets (0.9 = 10% discount per step)

        self.recalculation_period = 1  # Recalculate route every N steps
        self.teammate_prediction_steps = 5

        # Route planning state
        self.current_waypoints = []
        self.current_waypoint_index = 0
        self.steps_since_recalculation = 0
        self.last_known_targets = set()

        # Teammate prediction
        self.teammate_last_positions = []
        self.teammate_prediction_history_length = 3

        # Fallback to original LocalSearch behavior
        self.fallback_policy = LocalSearch(model_path, norm_stats_filepath)

        # Direction mapping for discrete actions
        self.directions = np.array([
            (0, 1),  # North (0°)
            (0.383, 0.924),  # NNE (22.5°)
            (0.707, 0.707),  # NE (45°)
            (0.924, 0.383),  # ENE (67.5°)
            (1, 0),  # East (90°)
            (0.924, -0.383),  # ESE (112.5°)
            (0.707, -0.707),  # SE (135°)
            (0.383, -0.924),  # SSE (157.5°)
            (0, -1),  # South (180°)
            (-0.383, -0.924),  # SSW (202.5°)
            (-0.707, -0.707),  # SW (225°)
            (-0.924, -0.383),  # WSW (247.5°)
            (-1, 0),  # West (270°)
            (-0.924, 0.383),  # WNW (292.5°)
            (-0.707, 0.707),  # NW (315°)
            (-0.383, 0.924),  # NNW (337.5°)
        ], dtype=np.float32)

        # Anti-oscillation state
        self._last_action = None
        self._action_repeat_count = 0
        self._max_repeat_count = 3

    def act(self, observation, env=None, agent_id=0):
        """
        Main action method that follows optimized early-acquisition route
        """
        if env is None:
            return self.fallback_policy.act(observation, env, agent_id), None

        # Update teammate position tracking
        self._update_teammate_tracking(env, agent_id)

        # Check if we need to recalculate route
        if (self.steps_since_recalculation >= self.recalculation_period or
                len(self.current_waypoints) == 0 or
                self._targets_changed(env)):
            self._recalculate_early_acquisition_route(env, agent_id)
            self.steps_since_recalculation = 0

        # Follow current route or fallback
        if len(self.current_waypoints) > 0 and self.current_waypoint_index < len(self.current_waypoints):
            action = self._navigate_to_current_waypoint(env, agent_id)
        else:
            action = self.fallback_policy.act(observation, env, agent_id)

        self.steps_since_recalculation += 1
        return action, None

    def _recalculate_early_acquisition_route(self, env, agent_id):
        """
        Recalculate route optimized for early target acquisition using discount factor
        """
        agent_pos = np.array([
            env.agents[env.aircraft_ids[agent_id]].x,
            env.agents[env.aircraft_ids[agent_id]].y
        ])

        # Get nearby unknown targets
        nearby_targets = self._get_nearby_unknown_targets(env, agent_pos)

        if len(nearby_targets) == 0:
            self.current_waypoints = []
            self.current_waypoint_index = 0
            return

        # Filter out targets teammate will visit if spatial coordination enabled
        if self.spatial_coord:
            teammate_will_visit = self._predict_teammate_targets(env, agent_id)
            filtered_targets = [t for t in nearby_targets if t['id'] not in teammate_will_visit]
        else:
            filtered_targets = nearby_targets

        if len(filtered_targets) == 0:
            filtered_targets = nearby_targets

        # Solve for early acquisition optimization
        if len(filtered_targets) == 1:
            self.current_waypoints = [filtered_targets[0]['position']]
        else:
            self.current_waypoints = self._solve_early_acquisition_route(agent_pos, filtered_targets)

        self.current_waypoint_index = 0
        print(f"[Clusters] Calculated early-acquisition route with {len(self.current_waypoints)} waypoints")

    def _solve_early_acquisition_route(self, start_pos, targets):
        """
        Solve for route that maximizes discounted target value (early targets worth more)
        """
        if len(targets) <= 1:
            return [t['position'] for t in targets]

        # Use greedy approach with value-to-distance ratio for larger problems
        if len(targets) > 8:
            return self._solve_greedy_value_distance_ratio(start_pos, targets)
        else:
            return self._solve_exact_early_acquisition(start_pos, targets)

    def _solve_exact_early_acquisition(self, start_pos, targets):
        """
        Solve exactly for small problems by trying all permutations
        """
        best_value = float('-inf')
        best_route = None

        # Try all possible routes
        for perm in itertools.permutations(targets):
            route_value = self._calculate_discounted_route_value(start_pos, perm)

            if route_value > best_value:
                best_value = route_value
                best_route = [t['position'] for t in perm]

        return best_route if best_route else [t['position'] for t in targets]

    def _solve_greedy_value_distance_ratio(self, start_pos, targets):
        """
        Greedy approach: always pick the target with best value-to-distance ratio
        considering the discount factor
        """
        route = []
        remaining_targets = targets.copy()
        current_pos = start_pos
        time_step = 0

        while remaining_targets:
            best_target = None
            best_score = float('-inf')

            for target in remaining_targets:
                distance = np.linalg.norm(target['position'] - current_pos)

                # Estimate time to reach this target (assuming constant speed)
                estimated_time_to_reach = time_step + distance / 10.0  # Adjust speed factor as needed

                # Calculate discounted value
                discounted_value = (self.discount_factor ** estimated_time_to_reach)

                # Add bonus for high-value targets
                target_base_value = 2.0 if target.get('high_value', False) else 1.0

                # Score combines discounted value with inverse distance
                score = (discounted_value * target_base_value) / max(distance, 1.0)

                if score > best_score:
                    best_score = score
                    best_target = target

            if best_target:
                route.append(best_target['position'])
                remaining_targets.remove(best_target)
                current_pos = best_target['position']
                time_step += np.linalg.norm(best_target['position'] - current_pos) / 10.0

        return route

    def _calculate_discounted_route_value(self, start_pos, target_sequence):
        """
        Calculate the total discounted value of visiting targets in the given sequence
        """
        total_value = 0.0
        current_pos = start_pos
        cumulative_distance = 0.0

        for i, target in enumerate(target_sequence):
            distance_to_target = np.linalg.norm(target['position'] - current_pos)
            cumulative_distance += distance_to_target

            # Estimate time step based on distance (assuming constant speed)
            time_step = cumulative_distance / 10.0  # Adjust speed factor as needed

            # Calculate discounted value
            discounted_value = (self.discount_factor ** time_step)

            # Add bonus for high-value targets
            base_value = 2.0 if target.get('high_value', False) else 1.0

            total_value += discounted_value * base_value
            current_pos = target['position']

        return total_value

    def _get_nearby_unknown_targets(self, env, agent_pos):
        """Get all unknown targets within search radius with enhanced target info"""
        targets = []
        target_positions = env.targets[:env.config['num_targets'], 3:5]
        target_info_levels = env.targets[:env.config['num_targets'], 2]
        target_values = env.targets[:env.config['num_targets'], 1]  # High-value flag

        for i, (pos, info_level, value) in enumerate(zip(target_positions, target_info_levels, target_values)):
            if info_level < 1.0:  # Unknown target
                distance = np.linalg.norm(pos - agent_pos)
                if distance <= self.search_radius:
                    targets.append({
                        'id': i,
                        'position': pos.copy(),
                        'distance': distance,
                        'high_value': value > 0  # Flag for high-value targets
                    })

        return targets

    # Include all the helper methods from the original class
    def _update_teammate_tracking(self, env, agent_id):
        """Update teammate position history for movement prediction"""
        if env.config['num_aircraft'] < 2:
            return

        teammate_id = 1 if agent_id == 0 else 0
        teammate_pos = np.array([
            env.agents[env.aircraft_ids[teammate_id]].x,
            env.agents[env.aircraft_ids[teammate_id]].y
        ])

        self.teammate_last_positions.append(teammate_pos)
        if len(self.teammate_last_positions) > self.teammate_prediction_history_length:
            self.teammate_last_positions.pop(0)

    def _targets_changed(self, env):
        """Check if the set of unknown targets has changed significantly"""
        current_targets = set()
        target_positions = env.targets[:env.config['num_targets'], 3:5]
        target_info_levels = env.targets[:env.config['num_targets'], 2]

        agent_pos = np.array([
            env.agents[env.aircraft_ids[0]].x,
            env.agents[env.aircraft_ids[0]].y
        ])

        for i, (pos, info_level) in enumerate(zip(target_positions, target_info_levels)):
            if info_level < 1.0:  # Unknown target
                distance = np.linalg.norm(pos - agent_pos)
                if distance <= self.search_radius:
                    current_targets.add(i)

        changed = len(current_targets.symmetric_difference(self.last_known_targets)) > 0
        self.last_known_targets = current_targets
        return changed

    def _predict_teammate_targets(self, env, agent_id):
        """Predict which targets the teammate will likely visit"""
        if env.config['num_aircraft'] < 2 or len(self.teammate_last_positions) < 2:
            return set()

        teammate_id = 1 if agent_id == 0 else 0
        teammate_pos = np.array([
            env.agents[env.aircraft_ids[teammate_id]].x,
            env.agents[env.aircraft_ids[teammate_id]].y
        ])

        # Predict teammate movement direction
        teammate_velocity = np.array([0.0, 0.0])
        if len(self.teammate_last_positions) >= 2:
            teammate_velocity = self.teammate_last_positions[-1] - self.teammate_last_positions[-2]

        predicted_pos = teammate_pos + teammate_velocity * self.teammate_prediction_steps

        # Find targets the teammate is likely to visit
        targets_teammate_will_visit = set()
        target_positions = env.targets[:env.config['num_targets'], 3:5]
        target_info_levels = env.targets[:env.config['num_targets'], 2]

        teammate_target_distances = []
        for i, (pos, info_level) in enumerate(zip(target_positions, target_info_levels)):
            if info_level < 1.0:  # Unknown target
                distance_to_predicted = np.linalg.norm(pos - predicted_pos)
                distance_to_current = np.linalg.norm(pos - teammate_pos)
                teammate_target_distances.append((i, min(distance_to_predicted, distance_to_current)))

        teammate_target_distances.sort(key=lambda x: x[1])
        max_teammate_targets = min(3, len(teammate_target_distances))

        for i in range(max_teammate_targets):
            targets_teammate_will_visit.add(teammate_target_distances[i][0])

        return targets_teammate_will_visit

    def _navigate_to_current_waypoint(self, env, agent_id):
        """Navigate to the current waypoint in the route"""
        if self.current_waypoint_index >= len(self.current_waypoints):
            return 0

        agent_pos = np.array([
            env.agents[env.aircraft_ids[agent_id]].x,
            env.agents[env.aircraft_ids[agent_id]].y
        ])

        target_pos = self.current_waypoints[self.current_waypoint_index]

        # Check if we've reached the current waypoint
        distance_to_waypoint = np.linalg.norm(target_pos - agent_pos)
        if distance_to_waypoint <= 30.0:  # Waypoint reached threshold
            self.current_waypoint_index += 1
            if self.current_waypoint_index >= len(self.current_waypoints):
                return 0  # All waypoints visited
            target_pos = self.current_waypoints[self.current_waypoint_index]

        # Calculate direction to target
        direction_to_target = target_pos - agent_pos
        target_norm = np.linalg.norm(direction_to_target)

        if target_norm == 0:
            return 0

        direction_to_target_norm = direction_to_target / target_norm

        # Find best matching action
        dot_products = np.dot(self.directions, direction_to_target_norm)
        best_action = np.argmax(dot_products)

        # Anti-oscillation logic
        if (self._last_action is not None and
                self._action_repeat_count < self._max_repeat_count and
                self._last_action != best_action):

            last_dot_product = dot_products[self._last_action]
            if last_dot_product > 0.5:
                best_action = self._last_action
                self._action_repeat_count += 1
            else:
                self._action_repeat_count = 0
        else:
            self._action_repeat_count = 0

        # Prevent direct opposite actions
        if (self._last_action is not None and abs(self._last_action - best_action) == 8):
            adjacent_actions = [(self._last_action + 1) % 16, (self._last_action - 1) % 16]
            adjacent_dots = [dot_products[a] for a in adjacent_actions]
            best_adjacent_idx = np.argmax(adjacent_dots)
            best_action = adjacent_actions[best_adjacent_idx]

        self._last_action = best_action
        return np.int32(best_action)
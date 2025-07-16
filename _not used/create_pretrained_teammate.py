    def _create_pretrained_rl_teammate(self):
        """
        Create pretrained RL mode selector teammate
        TODO: Load actual pretrained RL model
        """
        
        if self.pretrained_models_dir is None:
            print("Warning: No pretrained_models_dir specified, falling back to heuristic teammate")
            return self._create_fallback_heuristic_teammate("PretrainedRL_NoModelsDir_Fallback")
        
        if not os.path.exists(self.pretrained_models_dir):
            print(f"Warning: Pretrained models directory {self.pretrained_models_dir} does not exist, falling back to heuristic")
            return self._create_fallback_heuristic_teammate("PretrainedRL_NoModelsDir_Fallback")
        
        # Look for pretrained models with specific naming conventions
        model_patterns = [
            # Look for models with league-specific naming
            os.path.join(self.pretrained_models_dir, f"*{self.league_type}*.zip"),
            os.path.join(self.pretrained_models_dir, f"**/*{self.league_type}*.zip"),
            
            # Look for general pretrained models
            os.path.join(self.pretrained_models_dir, "pretrained_*.zip"),
            os.path.join(self.pretrained_models_dir, "**/pretrained_*.zip"),
            
            # Look for final/best models
            os.path.join(self.pretrained_models_dir, "*final*.zip"),
            os.path.join(self.pretrained_models_dir, "*best*.zip"),
            os.path.join(self.pretrained_models_dir, "**/*final*.zip"),
            os.path.join(self.pretrained_models_dir, "**/*best*.zip"),
            
            # General model search
            os.path.join(self.pretrained_models_dir, "*.zip"),
            os.path.join(self.pretrained_models_dir, "**/*.zip"),
        ]
        
        all_models = []
        for pattern in model_patterns:
            all_models.extend(glob.glob(pattern, recursive=True))
        
        
        if not all_models:
            print(f"Warning: No pretrained model files found in {self.pretrained_models_dir}, falling back to heuristic")
            return self._create_fallback_heuristic_teammate("PretrainedRL_NoModels_Fallback")
           
        
        selected_model = random.choice(all_models)
        
        print(f"Loading pretrained RL model: {os.path.basename(selected_model)} (strategy: {strategy_name})")
        pretrained_model = PPO.load(selected_model)
        
        # Create RL teammate policy using the loaded model
        rl_teammate = RLTeammatePolicy(
            model=pretrained_model,
            env=None,  # Will be set later if needed
            local_search_policy=self.subpolicies.get('local_search'),
            go_to_highvalue_policy=self.subpolicies.get('go_to_threat'),
            change_region_subpolicy=self.subpolicies.get('change_region'),
        )

        # Extract model identifier for naming
        model_name = os.path.splitext(os.path.basename(selected_model))[0]
        rl_teammate.name = f"PretrainedRL_{strategy_name}_{model_name}"
        
        self.current_teammate = rl_teammate
        return rl_teammate
        
        
def _create_fallback_heuristic_teammate(self, fallback_name):
        """
        Create a heuristic teammate as fallback when RL model loading fails
        """
        if self.league_type == "vanilla":
            teammate = self._create_vanilla_teammate()
        elif self.league_type == "strategy_diverse":
            teammate = self._create_strategy_diverse_teammate()
        else:
            teammate = self._create_baseline_teammate()
        
        teammate.name = fallback_name
        return teammate
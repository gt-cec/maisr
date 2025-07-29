import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from env_multi_new import MAISREnvVec
from utility.data_logging import load_env_config
from utility.league_management import TargetSearchLocalTSP


class TSPRouteAnalyzer:
    """Analyzer to directly test and visualize TSP route calculations"""

    def __init__(self, config_filename='configs/Monolith_R5L_july8.json'):
        self.config = load_env_config(config_filename)
        self.results = []

    def create_test_environment(self, seed=42):
        """Create a test environment with known setup"""
        np.random.seed(seed)

        # Create minimal environment
        env = MAISREnvVec(
            config=self.config,
            render_mode='headless',
            run_name='tsp_test',
            tag='analysis_0'
        )

        obs = env.reset()
        return env

    def analyze_tsp_behavior(self, spatial_coord_values=[True, False], search_methods=["greedy"], num_tests=5):
        """Analyze TSP behavior across different configurations"""

        for spatial_coord in spatial_coord_values:
            for search_method in search_methods:
                print(f"\n{'=' * 60}")
                print(f"Testing: Spatial Coord = {spatial_coord}, Method = {search_method}")
                print(f"{'=' * 60}")

                test_results = []

                for test_idx in range(num_tests):
                    print(f"Test {test_idx + 1}/{num_tests}")

                    # Create fresh environment for each test
                    env = self.create_test_environment(seed=42 + test_idx)

                    # Create TSP policy
                    tsp_policy = TargetSearchLocalTSP(
                        search_radius=1000,
                        spatial_coord=spatial_coord,
                        search_method=search_method
                    )

                    # Get agent positions
                    agent_id = 0
                    agent_pos = np.array([
                        env.agents[env.aircraft_ids[agent_id]].x,
                        env.agents[env.aircraft_ids[agent_id]].y
                    ])

                    teammate_pos = None
                    if env.config['num_aircraft'] >= 2:
                        teammate_pos = np.array([
                            env.agents[env.aircraft_ids[1]].x,
                            env.agents[env.aircraft_ids[1]].y
                        ])

                    # Manually call TSP methods to get detailed info
                    nearby_targets = tsp_policy._get_nearby_unknown_targets(env, agent_pos)

                    predicted_teammate_targets = set()
                    if spatial_coord and env.config['num_aircraft'] >= 2:
                        # Simulate teammate tracking for prediction
                        tsp_policy.teammate_last_positions = []
                        if teammate_pos is not None:
                            tsp_policy.teammate_last_positions.append(teammate_pos)
                            tsp_policy.teammate_last_positions.append(
                                teammate_pos + np.array([5, 5]))  # Simulate movement

                        predicted_teammate_targets = tsp_policy._predict_teammate_targets_dynamic(env, agent_id)

                    # Get filtered targets
                    filtered_targets = nearby_targets
                    if spatial_coord:
                        filtered_targets = [t for t in nearby_targets if t['id'] not in predicted_teammate_targets]
                        if len(filtered_targets) == 0:
                            filtered_targets = nearby_targets

                    # Calculate route
                    if search_method == "clusters":
                        route = tsp_policy._solve_tsp_with_clustering(agent_pos, filtered_targets)
                    elif search_method == 'early_weighted':
                        route = tsp_policy._solve_weighted_tsp(agent_pos, filtered_targets)
                    else:  # "greedy"
                        if len(filtered_targets) == 1:
                            route = [filtered_targets[0]['position']]
                        elif len(filtered_targets) <= 8:
                            route = tsp_policy._solve_tsp_exact(agent_pos, filtered_targets)
                        else:
                            route = tsp_policy._solve_tsp_heuristic(agent_pos, filtered_targets)

                    # Store results
                    result = {
                        'test_idx': test_idx,
                        'spatial_coord': spatial_coord,
                        'search_method': search_method,
                        'agent_pos': agent_pos.copy(),
                        'teammate_pos': teammate_pos.copy() if teammate_pos is not None else None,
                        'all_targets': nearby_targets,
                        'predicted_teammate_targets': predicted_teammate_targets,
                        'filtered_targets': filtered_targets,
                        'calculated_route': route,
                        'num_targets_total': len(nearby_targets),
                        'num_targets_filtered': len(filtered_targets),
                        'num_predicted_for_teammate': len(predicted_teammate_targets),
                        'route_length': len(route),
                        'env_targets': env.targets[:env.config['num_targets'], :].copy(),
                        'env_threats': env.threats.copy() if hasattr(env, 'threats') else np.array([])
                    }

                    test_results.append(result)

                    print(f"  Targets: {result['num_targets_total']} total, {result['num_targets_filtered']} filtered")
                    print(f"  Predicted for teammate: {result['num_predicted_for_teammate']}")
                    print(f"  Route length: {result['route_length']}")

                    env.close()

                self.results.extend(test_results)

        return self.results

    def plot_comparison(self, test_idx=0, save_dir='tsp_analysis_plots'):
        """Plot comparison between spatial coord on vs off for same test"""

        os.makedirs(save_dir, exist_ok=True)

        # Find results for same test index
        spatial_on_results = [r for r in self.results if r['spatial_coord'] == True and r['test_idx'] == test_idx]
        spatial_off_results = [r for r in self.results if r['spatial_coord'] == False and r['test_idx'] == test_idx]

        if not spatial_on_results or not spatial_off_results:
            print(f"No matching results found for test {test_idx}")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

        # Plot spatial coord OFF
        self._plot_single_result(spatial_off_results[0], ax1, "Spatial Coordination OFF")

        # Plot spatial coord ON
        self._plot_single_result(spatial_on_results[0], ax2, "Spatial Coordination ON")

        plt.tight_layout()
        save_path = os.path.join(save_dir, f'tsp_comparison_test_{test_idx}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comparison plot saved to {save_path}")
        plt.show()
        plt.close()

    def _plot_single_result(self, result, ax, title):
        """Plot a single TSP result"""

        # Set map bounds
        map_half_size = self.config["gameboard_size"] / 2
        ax.set_xlim(-map_half_size, map_half_size)
        ax.set_ylim(-map_half_size, map_half_size)

        # Plot all environment targets
        env_targets = result['env_targets']
        for i, target in enumerate(env_targets):
            target_pos = target[3:5]
            info_level = target[2]

            if info_level < 1.0:  # Unknown target
                if i in result['predicted_teammate_targets']:
                    # Predicted for teammate - purple square
                    ax.scatter(target_pos[0], target_pos[1], c='purple', marker='s', s=150,
                               alpha=0.8, edgecolors='black', linewidth=2,
                               label='Predicted for Teammate' if i == list(result['predicted_teammate_targets'])[0] else None)
                else: # Regular unknown target - orange circle
                    ax.scatter(target_pos[0], target_pos[1], c='orange', marker='o', s=100,
                               alpha=0.8, edgecolors='black', linewidth=1,
                               label='Unknown Target' if i == 0 else "")
            else:
                # Known target - green circle
                ax.scatter(target_pos[0], target_pos[1], c='green', marker='o', s=100,
                           alpha=0.8, edgecolors='black', linewidth=1,
                           label='Known Target' if info_level == 1.0 and i == 0 else "")

        # Plot TSP route
        if result['calculated_route']:
            route_x = [pos[0] for pos in result['calculated_route']]
            route_y = [pos[1] for pos in result['calculated_route']]

            # Draw route line from agent to first waypoint, then between waypoints
            full_route_x = [result['agent_pos'][0]] + route_x
            full_route_y = [result['agent_pos'][1]] + route_y

            ax.plot(full_route_x, full_route_y, 'b-', linewidth=3, alpha=0.8, label='TSP Route')

            # Plot waypoints
            ax.scatter(route_x, route_y, c='blue', marker='X', s=120,
                       alpha=0.9, edgecolors='darkblue', linewidth=2, label='TSP Waypoints')

            # Number the waypoints
            for i, (x, y) in enumerate(zip(route_x, route_y)):
                ax.annotate(str(i + 1), (x, y), xytext=(8, 8), textcoords='offset points',
                            fontsize=12, fontweight='bold', color='blue',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        # Plot agent position
        ax.scatter(result['agent_pos'][0], result['agent_pos'][1],
                   c='red', marker='*', s=300, edgecolors='darkred', linewidth=2,
                   label='Agent Position', zorder=10)

        # Plot teammate position
        if result['teammate_pos'] is not None:
            ax.scatter(result['teammate_pos'][0], result['teammate_pos'][1],
                       c='cyan', marker='*', s=300, edgecolors='darkcyan', linewidth=2,
                       label='Teammate Position', zorder=10)

        # Plot threats
        if len(result['env_threats']) > 0:
            for threat_idx, threat in enumerate(result['env_threats']):
                threat_pos = threat[:2]
                threat_radius = self.config['threat_radius']

                # Draw threat circle
                circle = patches.Circle((threat_pos[0], threat_pos[1]), threat_radius,
                                        fill=False, color='gold', linewidth=3, alpha=0.7)
                ax.add_patch(circle)

                # Draw threat marker
                ax.scatter(threat_pos[0], threat_pos[1], c='gold', marker='v', s=200,
                           alpha=0.8, edgecolors='orange', linewidth=2,
                           label='Threat' if threat_idx == 0 else "")

        # Add grid and formatting
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X Position', fontsize=12)
        ax.set_ylabel('Y Position', fontsize=12)
        ax.set_aspect('equal')

        # Add title with statistics
        stats_text = f"Targets: {result['num_targets_total']} total, {result['num_targets_filtered']} filtered\n"
        stats_text += f"Predicted for teammate: {result['num_predicted_for_teammate']}, Route length: {result['route_length']}"

        ax.set_title(f"{title}\n{stats_text}", fontsize=14, pad=20)

        # Add legend
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc='upper right',
                      bbox_to_anchor=(1.15, 1), fontsize=10)

    def generate_summary_report(self, save_path='tsp_analysis_report.txt'):
        """Generate a summary report of all tests"""

        if not self.results:
            print("No results to summarize")
            return

        with open(save_path, 'w') as f:
            f.write("TSP Route Analysis Summary Report\n")
            f.write("=" * 50 + "\n\n")

            # Group results by configuration
            spatial_on_results = [r for r in self.results if r['spatial_coord'] == True]
            spatial_off_results = [r for r in self.results if r['spatial_coord'] == False]

            f.write(f"Total tests conducted: {len(self.results)}\n")
            f.write(f"Tests with spatial coordination ON: {len(spatial_on_results)}\n")
            f.write(f"Tests with spatial coordination OFF: {len(spatial_off_results)}\n\n")

            # Analyze spatial coordination OFF
            if spatial_off_results:
                f.write("SPATIAL COORDINATION OFF:\n")
                f.write("-" * 30 + "\n")

                avg_targets_total = np.mean([r['num_targets_total'] for r in spatial_off_results])
                avg_targets_filtered = np.mean([r['num_targets_filtered'] for r in spatial_off_results])
                avg_route_length = np.mean([r['route_length'] for r in spatial_off_results])
                avg_predicted = np.mean([r['num_predicted_for_teammate'] for r in spatial_off_results])

                f.write(f"Average targets total: {avg_targets_total:.2f}\n")
                f.write(f"Average targets filtered: {avg_targets_filtered:.2f}\n")
                f.write(f"Average predicted for teammate: {avg_predicted:.2f}\n")
                f.write(f"Average route length: {avg_route_length:.2f}\n\n")

            # Analyze spatial coordination ON
            if spatial_on_results:
                f.write("SPATIAL COORDINATION ON:\n")
                f.write("-" * 30 + "\n")

                avg_targets_total = np.mean([r['num_targets_total'] for r in spatial_on_results])
                avg_targets_filtered = np.mean([r['num_targets_filtered'] for r in spatial_on_results])
                avg_route_length = np.mean([r['route_length'] for r in spatial_on_results])
                avg_predicted = np.mean([r['num_predicted_for_teammate'] for r in spatial_on_results])

                f.write(f"Average targets total: {avg_targets_total:.2f}\n")
                f.write(f"Average targets filtered: {avg_targets_filtered:.2f}\n")
                f.write(f"Average predicted for teammate: {avg_predicted:.2f}\n")
                f.write(f"Average route length: {avg_route_length:.2f}\n\n")

            # Detailed breakdown by test
            f.write("DETAILED TEST BREAKDOWN:\n")
            f.write("-" * 30 + "\n")

            for result in self.results:
                f.write(
                    f"Test {result['test_idx']}, Spatial: {result['spatial_coord']}, Method: {result['search_method']}\n")
                f.write(
                    f"  Targets: {result['num_targets_total']} total -> {result['num_targets_filtered']} filtered\n")
                f.write(f"  Predicted for teammate: {result['num_predicted_for_teammate']}\n")
                f.write(f"  Route length: {result['route_length']}\n")
                f.write(f"  Agent pos: ({result['agent_pos'][0]:.1f}, {result['agent_pos'][1]:.1f})\n")
                if result['teammate_pos'] is not None:
                    f.write(f"  Teammate pos: ({result['teammate_pos'][0]:.1f}, {result['teammate_pos'][1]:.1f})\n")
                f.write("\n")

        print(f"Summary report saved to {save_path}")

    def plot_all_tests(self, save_dir='tsp_analysis_plots'):
        """Plot all individual test results"""

        os.makedirs(save_dir, exist_ok=True)

        for result in self.results:
            fig, ax = plt.subplots(1, 1, figsize=(12, 10))

            title = f"Test {result['test_idx']} - Spatial Coord: {result['spatial_coord']} - {result['search_method']}"
            self._plot_single_result(result, ax, title)

            plt.tight_layout()
            filename = f"test_{result['test_idx']}_spatial_{result['spatial_coord']}_{result['search_method']}.png"
            save_path = os.path.join(save_dir, filename)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"Plot saved: {filename}")


def main():
    """Main function to run TSP analysis"""

    print("TSP Route Analysis Tool")
    print("=" * 50)

    # Create analyzer
    analyzer = TSPRouteAnalyzer()

    # Run analysis
    print("Running TSP behavior analysis...")
    results = analyzer.analyze_tsp_behavior(
        spatial_coord_values=[False, True],
        search_methods=["greedy"],
        num_tests=3
    )

    print(f"\nAnalysis complete! {len(results)} tests conducted.")

    # Generate summary report
    analyzer.generate_summary_report('tsp_analysis_report.txt')

    # Plot comparisons for each test
    print("\nGenerating comparison plots...")
    for test_idx in range(3):
        try:
            analyzer.plot_comparison(test_idx=test_idx)
        except Exception as e:
            print(f"Error plotting test {test_idx}: {e}")

    # Plot all individual tests
    print("\nGenerating individual test plots...")
    analyzer.plot_all_tests()

    print("\nAnalysis complete! Check the following files:")
    print("- tsp_analysis_report.txt: Summary statistics")
    print("- tsp_analysis_plots/: Individual and comparison plots")


if __name__ == "__main__":
    main()
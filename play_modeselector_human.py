import ctypes
import pygame
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import gymnasium as gym
from env_multi_new import MAISREnvVec
from training_wrappers.modeselector_training_wrapper import MaisrModeSelectorWrapper
from utility.data_logging import load_env_config
from policies.league_management import GenericTeammatePolicy, SubPolicy, LocalSearch, ChangeRegions, GoToNearestThreat, \
    EvadeDetection, TeammateManager
import json
import datetime
import math
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def calculate_spatial_coverage(positions, gameboard_size):
    """Calculate what percentage of the map was visited"""
    if not positions:
        return 0

    # Create a simple grid-based coverage metric
    grid_size = 20  # 20x20 grid
    visited_cells = set()
    cell_size = gameboard_size / grid_size

    for x, y in positions:
        # Convert to grid coordinates
        grid_x = int((x + gameboard_size / 2) // cell_size)
        grid_y = int((y + gameboard_size / 2) // cell_size)
        visited_cells.add((grid_x, grid_y))

    total_cells = grid_size * grid_size
    coverage_percentage = len(visited_cells) / total_cells * 100
    return min(coverage_percentage, 100.0)

def calculate_coordination_score(human_sequence, teammate_usage):
    """Calculate a simple coordination score between human and teammate"""
    if not human_sequence or not teammate_usage:
        return 0

    # Simple metric: how often they used different subpolicies
    total_teammate_steps = sum(teammate_usage.values())
    if total_teammate_steps == 0:
        return 0

    # Calculate diversity - higher score when using different subpolicies
    human_dominant = max(set(human_sequence), key=human_sequence.count)
    teammate_dominant = max(teammate_usage.keys(), key=lambda k: teammate_usage[k])

    coordination_score = 100 if human_dominant != teammate_dominant else 50
    return coordination_score

def run_sanity_checks(episode_data):
    """Run sanity checks to validate environment behavior"""

    print("\n" + "=" * 50)
    print("RUNNING SANITY CHECKS")
    print("=" * 50)

    # Group episodes by threat IDs
    episodes_by_threat_ids = {}
    for ep in episode_data:
        threat_count = ep['threat_ids']
        if threat_count not in episodes_by_threat_ids:
            episodes_by_threat_ids[threat_count] = []
        episodes_by_threat_ids[threat_count].append(ep)

    # Group episodes by target IDs
    episodes_by_target_ids = {}
    for ep in episode_data:
        target_count = ep['target_ids']
        if target_count not in episodes_by_target_ids:
            episodes_by_target_ids[target_count] = []
        episodes_by_target_ids[target_count].append(ep)

    # Calculate average rewards for each group
    avg_rewards_by_threats = {}
    avg_rewards_by_targets = {}

    for threat_count, episodes in episodes_by_threat_ids.items():
        avg_rewards_by_threats[threat_count] = sum(ep['total_reward'] for ep in episodes) / len(episodes)

    for target_count, episodes in episodes_by_target_ids.items():
        avg_rewards_by_targets[target_count] = sum(ep['total_reward'] for ep in episodes) / len(episodes)

    # SANITY CHECK 1: Episodes with threat_ids == 2 should have more reward than threat_ids == 0
    check1_passed = False
    if 0 in avg_rewards_by_threats and 2 in avg_rewards_by_threats:
        reward_0_threats = avg_rewards_by_threats[0]
        reward_2_threats = avg_rewards_by_threats[2]
        check1_passed = reward_2_threats > reward_0_threats

        print(f"CHECK 1 - Threat ID Reward Scaling (0 vs 2):")
        print(f"  Episodes with 0 threat IDs: avg reward = {reward_0_threats:.2f}")
        print(f"  Episodes with 2 threat IDs: avg reward = {reward_2_threats:.2f}")
        print(f"  Expected: 2 threats > 0 threats")
        print(f"  Result: {'✓ PASS' if check1_passed else '✗ FAIL'}")
    else:
        print(f"CHECK 1 - SKIPPED (missing data for 0 or 2 threat IDs)")
        print(f"  Available threat ID counts: {list(avg_rewards_by_threats.keys())}")

    # SANITY CHECK 2: Episodes with threat_ids == 3 should have less reward than threat_ids == 2
    check2_passed = False
    if 2 in avg_rewards_by_threats and 3 in avg_rewards_by_threats:
        reward_2_threats = avg_rewards_by_threats[2]
        reward_3_threats = avg_rewards_by_threats[3]
        check2_passed = reward_2_threats > reward_3_threats

        print(f"\nCHECK 2 - Threat ID Penalty (2 vs 3):")
        print(f"  Episodes with 2 threat IDs: avg reward = {reward_2_threats:.2f}")
        print(f"  Episodes with 3 threat IDs: avg reward = {reward_3_threats:.2f}")
        print(f"  Expected: 2 threats > 3 threats (penalty for too many)")
        print(f"  Result: {'✓ PASS' if check2_passed else '✗ FAIL'}")
    else:
        print(f"\nCHECK 2 - SKIPPED (missing data for 2 or 3 threat IDs)")
        print(f"  Available threat ID counts: {list(avg_rewards_by_threats.keys())}")

    # SANITY CHECK 3: More target IDs should generally mean higher rewards
    check3_passed = True
    target_counts_sorted = sorted(avg_rewards_by_targets.keys())
    if len(target_counts_sorted) >= 2:
        print(f"\nCHECK 3 - Target ID Reward Progression:")
        for i in range(len(target_counts_sorted) - 1):
            current_targets = target_counts_sorted[i]
            next_targets = target_counts_sorted[i + 1]
            current_reward = avg_rewards_by_targets[current_targets]
            next_reward = avg_rewards_by_targets[next_targets]

            step_passed = next_reward > current_reward
            check3_passed = check3_passed and step_passed

            print(f"  {current_targets} targets: {current_reward:.2f} vs {next_targets} targets: {next_reward:.2f}")
            print(f"    {'✓' if step_passed else '✗'} More targets should have higher reward")

        print(f"  Overall Result: {'✓ PASS' if check3_passed else '✗ FAIL'}")
    else:
        print(f"\nCHECK 3 - SKIPPED (insufficient target ID variation)")
        check3_passed = None

    # SANITY CHECK 4: Successful episodes should have higher rewards than failed ones
    successful_episodes = [ep for ep in episode_data if ep['completed_successfully']]
    failed_episodes = [ep for ep in episode_data if not ep['completed_successfully']]

    check4_passed = False
    if successful_episodes and failed_episodes:
        avg_reward_success = sum(ep['total_reward'] for ep in successful_episodes) / len(successful_episodes)
        avg_reward_failed = sum(ep['total_reward'] for ep in failed_episodes) / len(failed_episodes)
        check4_passed = avg_reward_success > avg_reward_failed

        print(f"\nCHECK 4 - Success vs Failure Rewards:")
        print(f"  Successful episodes ({len(successful_episodes)}): avg reward = {avg_reward_success:.2f}")
        print(f"  Failed episodes ({len(failed_episodes)}): avg reward = {avg_reward_failed:.2f}")
        print(f"  Expected: Success > Failure")
        print(f"  Result: {'✓ PASS' if check4_passed else '✗ FAIL'}")
    else:
        print(f"\nCHECK 4 - SKIPPED (all episodes had same outcome)")
        check4_passed = None

    # SANITY CHECK 5: Episodes with fewer steps should generally have higher rewards (efficiency bonus)
    episodes_with_rewards = [ep for ep in episode_data if ep['total_reward'] > 0]
    if len(episodes_with_rewards) >= 2:
        # Sort by steps and check if reward generally decreases with more steps
        episodes_by_steps = sorted(episodes_with_rewards, key=lambda x: x['steps'])

        # Compare shortest 1/3 vs longest 1/3
        third = len(episodes_by_steps) // 3
        if third > 0:
            shortest_episodes = episodes_by_steps[:third]
            longest_episodes = episodes_by_steps[-third:]

            avg_reward_short = sum(ep['total_reward'] for ep in shortest_episodes) / len(shortest_episodes)
            avg_reward_long = sum(ep['total_reward'] for ep in longest_episodes) / len(longest_episodes)
            avg_steps_short = sum(ep['steps'] for ep in shortest_episodes) / len(shortest_episodes)
            avg_steps_long = sum(ep['steps'] for ep in longest_episodes) / len(longest_episodes)

            check5_passed = avg_reward_short > avg_reward_long

            print(f"\nCHECK 5 - Step Efficiency (Early Finish Bonus):")
            print(f"  Shortest episodes (avg {avg_steps_short:.1f} steps): avg reward = {avg_reward_short:.2f}")
            print(f"  Longest episodes (avg {avg_steps_long:.1f} steps): avg reward = {avg_reward_long:.2f}")
            print(f"  Expected: Shorter episodes > Longer episodes (efficiency bonus)")
            print(f"  Result: {'✓ PASS' if check5_passed else '✗ FAIL'}")
        else:
            print(f"\nCHECK 5 - SKIPPED (insufficient episode variation)")
            check5_passed = None
    else:
        print(f"\nCHECK 5 - SKIPPED (insufficient positive reward episodes)")
        check5_passed = None

    # SANITY CHECK 6: Episodes with high detection counts should have lower rewards
    episodes_with_detections = [ep for ep in episode_data if ep['total_detections'] > 0]
    episodes_no_detections = [ep for ep in episode_data if ep['total_detections'] == 0]

    check6_passed = False
    if episodes_with_detections and episodes_no_detections:
        avg_reward_detections = sum(ep['total_reward'] for ep in episodes_with_detections) / len(
            episodes_with_detections)
        avg_reward_no_detections = sum(ep['total_reward'] for ep in episodes_no_detections) / len(
            episodes_no_detections)
        check6_passed = avg_reward_no_detections > avg_reward_detections

        print(f"\nCHECK 6 - Detection Penalty:")
        print(f"  Episodes with detections ({len(episodes_with_detections)}): avg reward = {avg_reward_detections:.2f}")
        print(
            f"  Episodes without detections ({len(episodes_no_detections)}): avg reward = {avg_reward_no_detections:.2f}")
        print(f"  Expected: No detections > With detections")
        print(f"  Result: {'✓ PASS' if check6_passed else '✗ FAIL'}")
    else:
        print(f"\nCHECK 6 - SKIPPED (all episodes had same detection pattern)")
        check6_passed = None

    # SUMMARY
    checks = [
        ("Threat ID Scaling (0 vs 2)", check1_passed),
        ("Threat ID Penalty (2 vs 3)", check2_passed),
        ("Target ID Progression", check3_passed),
        ("Success vs Failure", check4_passed),
        ("Step Efficiency", check5_passed),
        ("Detection Penalty", check6_passed)
    ]

    passed_checks = sum(1 for _, result in checks if result is True)
    failed_checks = sum(1 for _, result in checks if result is False)
    skipped_checks = sum(1 for _, result in checks if result is None)

    print(f"\n" + "=" * 50)
    print(f"SANITY CHECK SUMMARY")
    print(f"=" * 50)
    print(f"✓ PASSED: {passed_checks}")
    print(f"✗ FAILED: {failed_checks}")
    print(f"- SKIPPED: {skipped_checks}")

    if failed_checks > 0:
        print(f"\n⚠️  WARNING: {failed_checks} checks failed!")
        print("This suggests potential issues with reward structure or environment logic.")
        print("Failed checks:")
        for name, result in checks:
            if result is False:
                print(f"  - {name}")
    else:
        print(f"\n✅ All executable checks passed! Environment appears to be working correctly.")

    # SANITY CHECK 7: Policy Oscillation Detection
    print(f"\nCHECK 7 - Policy Oscillation Detection:")
    oscillation_issues = []

    for ep in episode_data:
        episode_num = ep['episode']

        # Check human policy oscillation (if we have subpolicy sequence)
        if 'subpolicy_sequence' in ep and ep['subpolicy_sequence']:
            human_oscillations = detect_policy_oscillation(ep['subpolicy_sequence'], episode_num, "Human")
            oscillation_issues.extend(human_oscillations)

        # Check teammate policy oscillation (reconstruct from usage if needed)
        if 'teammate_subpolicy_usage' in ep:
            # This is a simplified check - in real implementation you'd want to track teammate sequence too
            teammate_switches = ep.get('teammate_switches', 0)
            teammate_steps = sum(ep['teammate_subpolicy_usage'].values())
            if teammate_steps > 0:
                switch_rate = teammate_switches / teammate_steps
                if switch_rate > 0.3:  # More than 30% switch rate indicates potential oscillation
                    oscillation_issues.append({
                        'episode': episode_num,
                        'agent': 'Teammate',
                        'issue': f'High switch rate: {switch_rate:.2f} ({teammate_switches} switches in {teammate_steps} steps)'
                    })

    if oscillation_issues:
        print(f"  ⚠️ Found {len(oscillation_issues)} potential oscillation issues:")
        for issue in oscillation_issues[:5]:  # Show first 5
            print(f"    Episode {issue['episode']} ({issue['agent']}): {issue['issue']}")
        if len(oscillation_issues) > 5:
            print(f"    ... and {len(oscillation_issues) - 5} more")
        check7_passed = False
    else:
        print(f"  ✓ No policy oscillation issues detected")
        check7_passed = True

    # SANITY CHECK 8: Agent Movement/Stuck Detection
    print(f"\nCHECK 8 - Agent Movement (Stuck Detection):")
    movement_threshold = 40.0  # pixels
    check_window = 25  # steps to check
    stuck_issues = []

    for ep in episode_data:
        episode_num = ep['episode']

        if 'positions_visited' in ep and ep['positions_visited']:
            positions = ep['positions_visited']
            stuck_periods = detect_stuck_agent(positions, movement_threshold, check_window, episode_num)
            stuck_issues.extend(stuck_periods)

    if stuck_issues:
        print(f"  ⚠️ Found {len(stuck_issues)} periods where agent appeared stuck:")
        for issue in stuck_issues[:3]:  # Show first 3
            print(f"    Episode {issue['episode']}: Steps {issue['start_step']}-{issue['end_step']} "
                  f"(moved {issue['distance_moved']:.1f} pixels in {issue['duration']} steps)")
        if len(stuck_issues) > 3:
            print(f"    ... and {len(stuck_issues) - 3} more")
        check8_passed = False
    else:
        print(f"  ✓ No stuck agent periods detected (threshold: {movement_threshold} pixels in {check_window} steps)")
        check8_passed = True

    # SANITY CHECK 9: Subpolicy Effectiveness Check
    print(f"\nCHECK 9 - Subpolicy Usage Validation:")
    subpolicy_names = {0: "Local Search", 1: "Change Region", 2: "Go to Threat"}

    # Check if all subpolicies are being used
    total_usage = {0: 0, 1: 0, 2: 0}
    for ep in episode_data:
        if 'subpolicy_usage' in ep:
            for policy_id, count in ep['subpolicy_usage'].items():
                total_usage[policy_id] += count

    total_steps = sum(total_usage.values())
    unused_policies = [policy_id for policy_id, count in total_usage.items() if count == 0]
    underused_policies = [policy_id for policy_id, count in total_usage.items()
                          if count > 0 and count < total_steps * 0.05]  # Less than 5% usage

    check9_passed = True
    if unused_policies:
        print(f"  ⚠️ Unused subpolicies: {[subpolicy_names[p] for p in unused_policies]}")
        check9_passed = False
    elif underused_policies:
        print(f"  ⚠️ Underused subpolicies (<5%): {[subpolicy_names[p] for p in underused_policies]}")
        for policy_id in underused_policies:
            percentage = (total_usage[policy_id] / total_steps) * 100
            print(f"    {subpolicy_names[policy_id]}: {percentage:.1f}% usage")
        check9_passed = False
    else:
        print(f"  ✓ All subpolicies used reasonably:")
        for policy_id, count in total_usage.items():
            percentage = (count / total_steps) * 100 if total_steps > 0 else 0
            print(f"    {subpolicy_names[policy_id]}: {percentage:.1f}% usage")

    # Update checks list
    checks.extend([
        ("Policy Oscillation", check7_passed),
        ("Agent Movement", check8_passed),
        ("Subpolicy Usage", check9_passed)
    ])

    # Update summary
    passed_checks = sum(1 for _, result in checks if result is True)
    failed_checks = sum(1 for _, result in checks if result is False)
    skipped_checks = sum(1 for _, result in checks if result is None)

    print(f"\n" + "=" * 50)
    print(f"SANITY CHECK SUMMARY")
    print(f"=" * 50)
    print(f"✓ PASSED: {passed_checks}")
    print(f"✗ FAILED: {failed_checks}")
    print(f"- SKIPPED: {skipped_checks}")

    if failed_checks > 0:
        print(f"\n⚠️  WARNING: {failed_checks} checks failed!")
        print("This suggests potential issues with reward structure or environment logic.")
        print("Failed checks:")
        for name, result in checks:
            if result is False:
                print(f"  - {name}")
    else:
        print(f"\n✅ All executable checks passed! Environment appears to be working correctly.")

    # Store detailed results for further analysis
    detailed_results = {
        'checks': checks,
        'passed': passed_checks,
        'failed': failed_checks,
        'skipped': skipped_checks,
        'oscillation_issues': oscillation_issues,
        'stuck_issues': stuck_issues,
        'subpolicy_usage': total_usage
    }

    return detailed_results

def detect_policy_oscillation(policy_sequence, episode_num, agent_name, min_oscillations=3):
    """
    Detect if an agent is oscillating between two policies

    Args:
        policy_sequence: List of policy IDs over time
        episode_num: Episode number for reporting
        agent_name: Name of the agent (Human/Teammate)
        min_oscillations: Minimum number of back-and-forth switches to consider oscillation

    Returns:
        List of oscillation issues found
    """
    issues = []

    if len(policy_sequence) < 6:  # Need at least 6 steps to detect meaningful oscillation
        return issues

    # Look for patterns like [A, B, A, B, A, B] indicating oscillation
    i = 0
    while i < len(policy_sequence) - 5:  # Need at least 6 elements to check
        current_policy = policy_sequence[i]

        # Check if we have an alternating pattern
        oscillation_length = 0
        j = i

        while j < len(policy_sequence) - 1:
            if j % 2 == i % 2:  # Even positions (relative to start)
                expected_policy = current_policy
            else:  # Odd positions
                expected_policy = policy_sequence[i + 1] if i + 1 < len(policy_sequence) else None

            if expected_policy is not None and policy_sequence[j] == expected_policy:
                oscillation_length += 1
                j += 1
            else:
                break

        # If we found a long enough oscillating pattern
        if oscillation_length >= min_oscillations * 2:  # *2 because we need both A and B
            policy_a = current_policy
            policy_b = policy_sequence[i + 1] if i + 1 < len(policy_sequence) else None

            issues.append({
                'episode': episode_num,
                'agent': agent_name,
                'issue': f'Oscillating between policies {policy_a} and {policy_b} for {oscillation_length} steps (starting at step {i + 1})'
            })

            i += oscillation_length  # Skip past this oscillation
        else:
            i += 1

    return issues

def detect_stuck_agent(positions, movement_threshold, check_window, episode_num):
    """
    Detect periods where the agent didn't move much (potentially stuck)

    Args:
        positions: List of (x, y) positions over time
        movement_threshold: Minimum distance that should be moved in check_window steps
        check_window: Number of steps to check for movement
        episode_num: Episode number for reporting

    Returns:
        List of stuck periods found
    """
    import math

    stuck_periods = []

    if len(positions) < check_window:
        return stuck_periods

    # Sliding window to check movement
    for i in range(len(positions) - check_window + 1):
        start_pos = positions[i]
        end_pos = positions[i + check_window - 1]

        # Calculate distance moved in this window
        distance_moved = math.sqrt((end_pos[0] - start_pos[0]) ** 2 + (end_pos[1] - start_pos[1]) ** 2)

        # Also check maximum distance from start position within the window
        max_distance_from_start = 0
        for j in range(i, i + check_window):
            pos = positions[j]
            dist = math.sqrt((pos[0] - start_pos[0]) ** 2 + (pos[1] - start_pos[1]) ** 2)
            max_distance_from_start = max(max_distance_from_start, dist)

        # Consider stuck if both end-to-end distance and max excursion are small
        if distance_moved < movement_threshold and max_distance_from_start < movement_threshold * 1.5:
            # Check if this is a continuation of a previous stuck period
            if stuck_periods and stuck_periods[-1]['end_step'] >= i:
                # Extend the previous stuck period
                stuck_periods[-1]['end_step'] = i + check_window - 1
                stuck_periods[-1]['duration'] = stuck_periods[-1]['end_step'] - stuck_periods[-1]['start_step'] + 1
                stuck_periods[-1]['distance_moved'] = min(stuck_periods[-1]['distance_moved'], distance_moved)
            else:
                # New stuck period
                stuck_periods.append({
                    'episode': episode_num,
                    'start_step': i + 1,  # +1 for human-readable step numbers
                    'end_step': i + check_window,
                    'duration': check_window,
                    'distance_moved': distance_moved,
                    'start_position': start_pos,
                    'end_position': end_pos
                })

    return stuck_periods

def create_analysis_plots(episode_data):
    """Create analysis plots for episode data"""

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))

    # Extract data for plotting
    episodes = [ep['episode'] for ep in episode_data]
    rewards = [ep['total_reward'] for ep in episode_data]
    target_ids = [ep['target_ids'] for ep in episode_data]
    threat_ids = [ep['threat_ids'] for ep in episode_data]

    # Calculate average teammate distance for each episode
    avg_teammate_distances = []
    for ep in episode_data:
        if ('positions_visited' in ep and ep['positions_visited'] and
                'teammate_positions_visited' in ep and ep['teammate_positions_visited']):

            agent_positions = ep['positions_visited']
            teammate_positions = ep['teammate_positions_visited']

            # Calculate distance at each time step (use minimum length in case of mismatch)
            min_length = min(len(agent_positions), len(teammate_positions))
            distances = []

            for i in range(min_length):
                agent_pos = agent_positions[i]
                teammate_pos = teammate_positions[i]
                distance = math.sqrt((agent_pos[0] - teammate_pos[0]) ** 2 +
                                     (agent_pos[1] - teammate_pos[1]) ** 2)
                distances.append(distance)

            # Calculate average distance for this episode
            avg_distance = sum(distances) / len(distances) if distances else 150
            avg_teammate_distances.append(avg_distance)
        else:
            raise ValueError
            # Fallback if data is missing
            #avg_teammate_distances.append(150)  # Default value

    # Plot 1: Reward vs Average Teammate Distance
    plt.subplot(2, 3, 1)
    plt.scatter(avg_teammate_distances, rewards, alpha=0.7, c=episodes, cmap='viridis')
    plt.xlabel('Average Teammate Distance (pixels)')
    plt.ylabel('Total Reward')
    plt.title('Reward vs Average Teammate Distance')
    plt.colorbar(label='Episode Number')

    # Add trend line
    if len(avg_teammate_distances) > 1:
        z = np.polyfit(avg_teammate_distances, rewards, 1)
        p = np.poly1d(z)
        plt.plot(sorted(avg_teammate_distances), p(sorted(avg_teammate_distances)), "r--", alpha=0.8)

    # Plot 2: Reward vs Target IDs
    plt.subplot(2, 3, 2)
    plt.scatter(target_ids, rewards, alpha=0.7, c=episodes, cmap='viridis', s=60)
    plt.xlabel('Target IDs Gained')
    plt.ylabel('Total Reward')
    plt.title('Reward vs Target IDs')
    plt.colorbar(label='Episode Number')

    # Add trend line
    if len(target_ids) > 1 and len(set(target_ids)) > 1:
        z = np.polyfit(target_ids, rewards, 1)
        p = np.poly1d(z)
        target_range = np.linspace(min(target_ids), max(target_ids), 100)
        plt.plot(target_range, p(target_range), "r--", alpha=0.8)

    # Show correlation coefficient
    if len(target_ids) > 1:
        corr = np.corrcoef(target_ids, rewards)[0, 1]
        plt.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=plt.gca().transAxes,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Plot 3: Reward vs Threat IDs
    plt.subplot(2, 3, 3)
    plt.scatter(threat_ids, rewards, alpha=0.7, c=episodes, cmap='viridis', s=60)
    plt.xlabel('Threat IDs Gained')
    plt.ylabel('Total Reward')
    plt.title('Reward vs Threat IDs')
    plt.colorbar(label='Episode Number')

    # Add trend line if there's variation in threat IDs
    if len(threat_ids) > 1 and len(set(threat_ids)) > 1:
        z = np.polyfit(threat_ids, rewards, 1)
        p = np.poly1d(z)
        threat_range = np.linspace(min(threat_ids), max(threat_ids), 100)
        plt.plot(threat_range, p(threat_range), "r--", alpha=0.8)

    # Show correlation coefficient
    if len(threat_ids) > 1:
        corr = np.corrcoef(threat_ids, rewards)[0, 1]
        plt.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=plt.gca().transAxes,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Plot 4: 3D Plot - Reward vs Target IDs and Threat IDs
    ax = fig.add_subplot(2, 3, 4, projection='3d')
    scatter = ax.scatter(target_ids, threat_ids, rewards, c=rewards, cmap='coolwarm', s=60, alpha=0.8)
    ax.set_xlabel('Target IDs Gained')
    ax.set_ylabel('Threat IDs Gained')
    ax.set_zlabel('Total Reward')
    ax.set_title('3D: Reward vs Target & Threat IDs')
    plt.colorbar(scatter, ax=ax, label='Episode Number', shrink=0.6)

    # Plot 5: Efficiency Analysis
    plt.subplot(2, 3, 5)
    efficiency_scores = [ep['efficiency_score'] for ep in episode_data]
    steps = [ep['steps'] for ep in episode_data]
    plt.scatter(steps, efficiency_scores, alpha=0.7, c=rewards, cmap='RdYlGn', s=60)
    plt.xlabel('Episode Steps')
    plt.ylabel('Efficiency Score (Targets/Step)')
    plt.title('Efficiency vs Episode Length')
    plt.colorbar(label='Total Reward')

    # Plot 6: Subpolicy Usage Effectiveness
    plt.subplot(2, 3, 6)

    # Calculate dominant subpolicy for each episode
    dominant_subpolicies = []
    for ep in episode_data:
        usage = ep.get('subpolicy_usage', {0: 0, 1: 0, 2: 0})
        dominant = max(usage.keys(), key=lambda k: usage[k])
        dominant_subpolicies.append(dominant)

    # Group rewards by dominant subpolicy
    subpolicy_names = {0: "Local Search", 1: "Change Region", 2: "Go to Threat"}
    colors = {0: 'blue', 1: 'green', 2: 'red'}

    for policy_id in [0, 1, 2]:
        policy_episodes = [i for i, dom in enumerate(dominant_subpolicies) if dom == policy_id]
        if policy_episodes:
            policy_rewards = [rewards[i] for i in policy_episodes]
            policy_targets = [target_ids[i] for i in policy_episodes]
            plt.scatter(policy_targets, policy_rewards,
                        label=subpolicy_names[policy_id],
                        color=colors[policy_id], alpha=0.7, s=60)

    plt.xlabel('Target IDs Gained')
    plt.ylabel('Total Reward')
    plt.title('Reward vs Targets by Dominant Subpolicy')
    plt.legend()

    # Adjust layout and save
    plt.tight_layout()

    # Save the figure
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f"./logs/env_tests/MS_analysis_plots_{timestamp}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\nAnalysis plots saved to {plot_filename}")

    # Additional statistical summary
    print("\n=== CORRELATION ANALYSIS ===")
    if len(set(target_ids)) > 1:
        target_reward_corr = np.corrcoef(target_ids, rewards)[0, 1]
        print(f"Target IDs vs Reward correlation: {target_reward_corr:.3f}")

    if len(set(threat_ids)) > 1:
        threat_reward_corr = np.corrcoef(threat_ids, rewards)[0, 1]
        print(f"Threat IDs vs Reward correlation: {threat_reward_corr:.3f}")

    if len(set(avg_teammate_distances)) > 1:
        distance_reward_corr = np.corrcoef(avg_teammate_distances, rewards)[0, 1]
        print(f"Teammate Distance vs Reward correlation: {distance_reward_corr:.3f}")

    efficiency_reward_corr = np.corrcoef(efficiency_scores, rewards)[0, 1]
    print(f"Efficiency vs Reward correlation: {efficiency_reward_corr:.3f}")

    # Subpolicy effectiveness summary
    print(f"\n=== SUBPOLICY EFFECTIVENESS ===")
    for policy_id in [0, 1, 2]:
        policy_episodes = [ep for ep in episode_data if
                           max(ep['subpolicy_usage'].keys(), key=lambda k: ep['subpolicy_usage'][k]) == policy_id]
        if policy_episodes:
            avg_reward = sum(ep['total_reward'] for ep in policy_episodes) / len(policy_episodes)
            avg_targets = sum(ep['target_ids'] for ep in policy_episodes) / len(policy_episodes)
            print(
                f"{subpolicy_names[policy_id]}: {len(policy_episodes)} episodes, avg reward: {avg_reward:.2f}, avg targets: {avg_targets:.1f}")




if __name__ == "__main__":

    config_filename = 'configs/june20_leagues.json'
    league_type = 'strategy_diverse'
    num_episodes = 30

    localsearch_model_path = None  # 'trained_models/local_search_2000000.0timesteps_0.1threatpenalty_0615_1541_6envs_maisr_trained_model.zip'
    localsearch_normstats_path = 'trained_models/local_search_2000000.0timesteps_0.1threatpenalty_0615_1541_6envslocal_search_norm_stats.npy'


    config = load_env_config(config_filename)
    print(f'LOADED CONFIG {config_filename}')
    pygame.display.init()
    pygame.font.init()
    clock = pygame.time.Clock()
    ctypes.windll.user32.SetProcessDPIAware()
    window_width, window_height = config['window_size'][0], config['window_size'][1]
    config['tick_rate'] = 30
    window = pygame.display.set_mode((window_width, window_height), flags=pygame.NOFRAME)
    pygame.display.set_caption("MAISR Human Interface")

    base_env = MAISREnvVec(
        config=config,
        clock=clock,
        window=window,
        render_mode='human',
        run_name='hrl_test',
        tag=f'test0',
    )

    subpolicies = {
        'local_search': LocalSearch(model_path=None),  # Using heuristic
        'change_region': ChangeRegions(model_path=None),  # Using heuristic
        'go_to_threat': GoToNearestThreat(model_path=None)  # Using heuristic
    }

    env = MaisrModeSelectorWrapper(
        base_env,
        local_search_policy = LocalSearch(model_path = localsearch_model_path, norm_stats_filepath = localsearch_normstats_path),
        go_to_highvalue_policy=GoToNearestThreat(model_path=None),
        change_region_subpolicy = ChangeRegions(model_path=None),
        evade_policy = EvadeDetection(model_path=None),
        teammate_manager = TeammateManager(league_type=league_type, subpolicies=subpolicies)
    )


    ###################################################################################################################

    key_to_action = {pygame.K_1: 0, pygame.K_2: 1, pygame.K_3: 2}
    all_observations = []
    episode_rewards = []
    episode_observations = []
    episode_actions = []
    all_actions = []

    episode_data = []  # Store data for all episodes
    # Modified main loop with comprehensive logging
    for episode in range(num_episodes):
        obs = env.reset()[0]
        episode_reward = 0

        # Initialize all tracking variables
        episode_steps = 0
        initial_threat_ids = env.env.num_threats_identified
        initial_target_ids = env.env.targets_identified

        # Subpolicy tracking
        subpolicy_usage = {0: 0, 1: 0, 2: 0}
        subpolicy_switches = 0
        last_action = None
        subpolicy_sequence = []

        # Teammate tracking
        teammate_subpolicy_usage = {0: 0, 1: 0, 2: 0}
        teammate_switches = 0
        last_teammate_action = None

        # Performance tracking
        detection_events = []
        identification_events = []
        distance_traveled = 0
        last_position = None
        positions_visited = []
        teammate_positions_visited = []
        target_discovery_times = {}
        threat_discovery_times = {}

        # Timing tracking
        episode_start_time = pygame.time.get_ticks()

        done = False
        action = 0

        print(f"\nStarting human episode {episode + 1}/10")

        while not done:
            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                    break
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        done = True
                        break
                    elif event.key in key_to_action:
                        action = key_to_action[event.key]

            if done:
                break

            # Track subpolicy usage
            subpolicy_usage[action] += 1
            subpolicy_sequence.append(action)
            if last_action is not None and last_action != action:
                subpolicy_switches += 1
            last_action = action

            # Track teammate behavior
            ai_subpolicy_id, ai_subpolicy_name = env.get_teammate_subpolicy_info()
            teammate_subpolicy_usage[ai_subpolicy_id] += 1
            if last_teammate_action is not None and last_teammate_action != ai_subpolicy_id:
                teammate_switches += 1
            last_teammate_action = ai_subpolicy_id

            # Take step
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
            episode_steps += 1

            # Track position and distance
            current_pos = (env.env.agents[env.env.aircraft_ids[0]].x, env.env.agents[env.env.aircraft_ids[0]].y)
            teammate_current_pos = (env.env.agents[env.env.aircraft_ids[1]].x, env.env.agents[env.env.aircraft_ids[1]].y)
            positions_visited.append(current_pos)
            teammate_positions_visited.append(teammate_current_pos)

            if last_position is not None:
                distance_traveled += math.sqrt((current_pos[0] - last_position[0]) ** 2 +
                                               (current_pos[1] - last_position[1]) ** 2)
            last_position = current_pos

            # Track events
            if 'new_detections' in info and info['new_detections'] > 0:
                detection_events.append({
                    'step': episode_steps,
                    'count': info['new_detections'],
                    'position': current_pos
                })

            if 'new_target_ids' in info and info['new_target_ids'] > 0:
                identification_events.append({
                    'step': episode_steps,
                    'type': 'target',
                    'count': info['new_target_ids']
                })
                target_discovery_times[len(target_discovery_times)] = episode_steps

            if 'new_threat_ids' in info and info['new_threat_ids'] > 0:
                identification_events.append({
                    'step': episode_steps,
                    'type': 'threat',
                    'count': info['new_threat_ids']
                })
                threat_discovery_times[len(threat_discovery_times)] = episode_steps

            # Render
            env.render()
            human_subpolicy_id, human_subpolicy_name = env.get_current_subpolicy_info()
            ai_subpolicy_id, ai_subpolicy_name = env.get_teammate_subpolicy_info()
            env.env.render_subpolicy_indicators(human_subpolicy_id, human_subpolicy_name, ai_subpolicy_id,
                                                ai_subpolicy_name)
            pygame.display.flip()

        # Calculate episode duration
        episode_end_time = pygame.time.get_ticks()
        episode_duration_ms = episode_end_time - episode_start_time

        # Calculate final counts
        final_threat_ids = env.env.num_threats_identified
        final_target_ids = env.env.targets_identified
        threat_ids_gained = final_threat_ids - initial_threat_ids
        target_ids_gained = final_target_ids - initial_target_ids

        # Store comprehensive episode data
        episode_info = {
            # Basic metrics
            'episode': episode + 1,
            'steps': episode_steps,
            'threat_ids': threat_ids_gained,
            'target_ids': target_ids_gained,
            'total_reward': episode_reward,
            'duration_ms': episode_duration_ms,

            # Subpolicy analytics
            'subpolicy_usage': subpolicy_usage.copy(),
            'subpolicy_switches': subpolicy_switches,
            'subpolicy_percentages': {
                k: (v / episode_steps * 100) if episode_steps > 0 else 0
                for k, v in subpolicy_usage.items()
            },

            # Teammate analytics
            'teammate_subpolicy_usage': teammate_subpolicy_usage.copy(),
            'teammate_switches': teammate_switches,
            'coordination_score': calculate_coordination_score(subpolicy_sequence, teammate_subpolicy_usage),
            'positions_visited': positions_visited,
            'teammate_positions_visited': teammate_positions_visited,

            # Performance metrics
            'distance_traveled': distance_traveled,
            'efficiency_score': target_ids_gained / max(episode_steps, 1),
            'spatial_coverage_percent': calculate_spatial_coverage(positions_visited, env.env.config['gameboard_size']),
            'avg_distance_per_step': distance_traveled / max(episode_steps, 1),

            # Event tracking
            'detection_events': detection_events,
            'identification_events': identification_events,
            'total_detections': env.env.detections,
            'num_detection_events': len(detection_events),

            # Timing analysis
            'target_discovery_times': target_discovery_times,
            'threat_discovery_times': threat_discovery_times,
            'time_to_first_target': min(target_discovery_times.values()) if target_discovery_times else None,
            'time_to_last_target': max(target_discovery_times.values()) if target_discovery_times else None,

            # Episode outcome
            'completed_successfully': env.env.all_targets_identified,
            'failed': getattr(env, 'failed', False),
            'termination_reason': 'success' if env.env.all_targets_identified else
            'failed' if getattr(env, 'failed', False) else 'timeout',

            # Environment state
            'final_threat_count': final_threat_ids,
            'final_target_count': final_target_ids,
            'num_targets_total': env.env.config['num_targets'],
            'num_threats_total': env.env.config['num_threats'],
            'gameboard_size': env.env.config['gameboard_size'],
            'max_steps_allowed': env.env.max_steps,
        }

        episode_data.append(episode_info)

        # Print detailed episode summary
        print(f"\nEpisode {episode + 1} completed:")
        print(f"  Steps: {episode_steps}")
        print(f"  Duration: {episode_duration_ms / 1000:.1f}s")
        print(f"  Threat IDs: {threat_ids_gained}, Target IDs: {target_ids_gained}")
        print(f"  Total reward: {episode_reward:.2f}")
        print(
            f"  Subpolicy usage: Local:{subpolicy_usage[0]}, Change:{subpolicy_usage[1]}, Threat:{subpolicy_usage[2]}")
        print(f"  Subpolicy switches: {subpolicy_switches}")
        print(f"  Distance traveled: {distance_traveled:.1f}")
        print(f"  Spatial coverage: {episode_info['spatial_coverage_percent']:.1f}%")
        print(f"  Efficiency: {episode_info['efficiency_score']:.3f} targets/step")
        print(f"  Termination: {episode_info['termination_reason']}")

    env.close()


def convert_to_json_serializable(obj):
    """Convert numpy types and other non-serializable objects to JSON-serializable types"""
    import numpy as np

    if isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_json_serializable(item) for item in obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


# Convert episode_data to JSON-serializable format
serializable_episode_data = convert_to_json_serializable(episode_data)

# Save episode data to JSON file
import os

os.makedirs('./logs/env_tests', exist_ok=True)  # Create directory if it doesn't exist

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"./logs/env_tests/MS_test_episode_data_{timestamp}.json"

try:
    with open(filename, 'w') as f:
        json.dump(serializable_episode_data, f, indent=2)
    print(f"\nEpisode data saved to {filename}")
except Exception as e:
    print(f"Error saving JSON file: {e}")
    # Fallback: save as pickle if JSON fails
    import pickle

    pickle_filename = filename.replace('.json', '.pkl')
    with open(pickle_filename, 'wb') as f:
        pickle.dump(episode_data, f)
    print(f"Saved as pickle instead: {pickle_filename}")

# Print summary statistics
print("\n=== EPISODE SUMMARY ===")
total_steps = sum(ep['steps'] for ep in episode_data)
total_threats = sum(ep['threat_ids'] for ep in episode_data)
total_targets = sum(ep['target_ids'] for ep in episode_data)
total_reward = sum(ep['total_reward'] for ep in episode_data)

print(f"Total episodes: {len(episode_data)}")
print(f"Average steps per episode: {total_steps / len(episode_data):.1f}")
print(f"Total threat IDs: {total_threats}")
print(f"Total target IDs: {total_targets}")
print(f"Average reward per episode: {total_reward / len(episode_data):.2f}")

# Run sanity checks
sanity_results = run_sanity_checks(episode_data)

# Run the plotting function
create_analysis_plots(episode_data)
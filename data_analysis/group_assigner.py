import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from scipy import stats

# TODO Untested

""" Input:
new_participant = {
    'age': 32,  # numeric value based on age range
    'gender': 1,  # 0=Male, 1=Female, 2=Non-binary, 3=Prefer not to say
    'ai_experience': 3,  # 1-5 scale
    'gaming_experience': 2,  # 1-5 scale
    'trust_score': 3.5  # average of trust questions (1-5 scale)
}
"""

def preprocess_demographics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the demographics data from the Excel file.
    """
    # Select and rename relevant columns
    demographics = df[[
        'Q1',  # age
        'Q2',  # gender
        'Q6',  # AI experience
        'Q3',  # video game experience
        'Q13', 'Q14', 'Q15', 'Q16', 'Q17', 'Q18'  # trust in automation questions
    ]].copy()

    # Convert age ranges to numeric
    age_mapping = {
        '18-24': 21,
        '25-39': 32,
        '40-54': 47,
        '55+': 60
    }
    demographics['age'] = demographics['Q1'].map(age_mapping)

    # Convert gender to numeric (for calculation purposes)
    gender_mapping = {
        'Male': 0,
        'Female': 1,
        'Non-binary': 2,
        'Prefer not to say': 3
    }
    demographics['gender'] = demographics['Q2'].map(gender_mapping)

    # Convert AI experience to numeric scale (1-5)
    ai_exp_mapping = {
        'I have never used AI/ML tools': 1,
        'I have used AI tools (like ChatGPT) but have no formal training': 2,
        'I have taken 1-2 AI/ML courses (university or online), OR I have used AI development tools in a professional/research setting': 3,
        'I have taken 3+ AI/ML courses, OR I regularly work with AI/ML': 4,
        'I am an AI/ML researcher or professional': 5
    }
    demographics['ai_experience'] = demographics['Q6'].map(ai_exp_mapping)

    # Convert gaming experience to numeric scale (1-5)
    gaming_exp_mapping = {
        'Almost never': 1,
        'A few times per year': 2,
        '1-5 hours per week': 3,
        'More than 5 hours per week': 4,
        'More than 15 hours per week': 5
    }
    demographics['gaming_experience'] = demographics['Q3'].map(gaming_exp_mapping)

    # Calculate trust in automation score (average of Q13-Q18, with Q16 reversed)
    trust_columns = ['Q13', 'Q14', 'Q15', 'Q16', 'Q17', 'Q18']
    demographics[trust_columns] = demographics[trust_columns].apply(pd.to_numeric, errors='coerce')
    demographics['Q16_reversed'] = 6 - demographics['Q16']  # Reverse score Q16
    trust_cols_final = ['Q13', 'Q14', 'Q15', 'Q16_reversed', 'Q17', 'Q18']
    demographics['trust_score'] = demographics[trust_cols_final].mean(axis=1)

    return demographics[['age', 'gender', 'ai_experience', 'gaming_experience', 'trust_score']]


def calculate_group_stats(group_data: pd.DataFrame) -> Dict:
    """
    Calculate mean and variance for each demographic variable in a group.
    """
    if len(group_data) == 0:
        return {col: {'mean': 0, 'var': 0} for col in group_data.columns}

    return {
        col: {
            'mean': group_data[col].mean(),
            'var': group_data[col].var()
        }
        for col in group_data.columns
    }


def calculate_balance_score(group_stats: List[Dict], new_participant: pd.Series) -> float:
    """
    Calculate how well balanced the groups would be if new participant joins.
    Lower score means better balance.
    """
    balance_score = 0

    # Compare means and variances across groups
    for var in group_stats[0].keys():
        means = [stats['mean'] for stats in group_stats]
        vars = [stats['var'] for stats in group_stats]

        # Calculate coefficient of variation for means and variances
        mean_cv = np.std(means) / (np.mean(means) if np.mean(means) != 0 else 1)
        var_cv = np.std(vars) / (np.mean(vars) if np.mean(vars) != 0 else 1)

        balance_score += mean_cv + var_cv

    return balance_score


def assign_group(excel_path: str, new_participant: Dict) -> str:
    """
    Determine the best group assignment for a new participant.

    Args:
        excel_path: Path to Excel file with current participant data
        new_participant: Dictionary with new participant's demographics

    Returns:
        str: Recommended group assignment ('control', 'model card', or 'in situ')
    """
    # Read and preprocess existing data
    df = pd.read_excel(excel_path)
    demographics = preprocess_demographics(df)

    # Count current group sizes
    group_sizes = df['Group'].value_counts()
    smallest_group_size = group_sizes.min() if not group_sizes.empty else 0
    largest_group_size = group_sizes.max() if not group_sizes.empty else 0

    # If groups are significantly unbalanced, assign to smallest group
    if largest_group_size - smallest_group_size >= 2:
        return group_sizes.idxmin()

    # Prepare new participant data
    new_participant_processed = pd.Series({
        'age': new_participant['age'],
        'gender': new_participant['gender'],
        'ai_experience': new_participant['ai_experience'],
        'gaming_experience': new_participant['gaming_experience'],
        'trust_score': new_participant['trust_score']
    })

    # Calculate current stats for each group
    groups = ['control', 'model card', 'in situ']
    best_group = None
    best_balance_score = float('inf')

    # Try adding participant to each group and calculate resulting balance
    for test_group in groups:
        # Create temporary group assignments including new participant
        temp_assignments = df['Group'].copy()
        temp_demographics = demographics.copy()

        # Add new participant to test group
        temp_assignments = pd.concat([temp_assignments, pd.Series([test_group])])
        temp_demographics = pd.concat([temp_demographics, pd.DataFrame([new_participant_processed])])

        # Calculate group statistics
        group_stats = []
        for group in groups:
            group_data = temp_demographics[temp_assignments == group]
            group_stats.append(calculate_group_stats(group_data))

        # Calculate balance score for this arrangement
        balance_score = calculate_balance_score(group_stats, new_participant_processed)

        # Update best group if this arrangement is more balanced
        if balance_score < best_balance_score:
            best_balance_score = balance_score
            best_group = test_group

    return best_group


# Example usage:
if __name__ == "__main__":
    # Example new participant data
    new_participant = {
        'age': 32,  # corresponding to '25-39' age range
        'gender': 1,  # Female
        'ai_experience': 3,  # '1-2 AI/ML courses'
        'gaming_experience': 2,  # 'A few times per year'
        'trust_score': 3.5  # Average of trust questions (1-5 scale)
    }

    # Get recommended group assignment
    recommended_group = assign_group('MAISR demographics_January 6 2025.xlsx', new_participant)
    print(f"Recommended group assignment: {recommended_group}")
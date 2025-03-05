"""
This module provides functions to calculate Average Treatment Effect (ATE),
Average Treatment Effect on the Treated (ATT),
and Average Treatment Effect on the Control (ATC)

Functions:
    calculate_measures_matching(x, t, y, n_matches=11):
        Calculate ATE, ATT, and ATC using propensity score matching.

Main Execution:
    Reads and transforms the data from a specified path,
    calculates point estimates for ATE, ATT, and ATC using
    propensity score matching, and prints the results.
    Additionally, calculates and prints the 95% confidence intervals
    for ATE, ATT, and ATC using bootstrap confidence intervals.
"""

import numpy as np

from sklearn.neighbors import NearestNeighbors

from utils import read_and_transform_data, calculate_propensity_scores, bci


def calculate_measures_matching(x, t, y, n_matches=11):
    """
    Calculate Average Treatment Effect (ATE), Average Treatment Effect on the Treated (ATT),
    and Average Treatment Effect on the Control (ATC) using propensity score matching.
    Parameters:
    x (pd.DataFrame): Covariates/features used to estimate propensity scores.
    t (pd.Series): Treatment indicator (1 for treated, 0 for control).
    y (pd.Series): Outcome variable.
    n_matches (int, optional): Number of nearest neighbors to match. Default is 11.
    Returns:
    tuple: A tuple containing:
        - ate (float): Average Treatment Effect.
        - att (float): Average Treatment Effect on the Treated.
        - atc (float): Average Treatment Effect on the Control.
    """

    propensity_scores = calculate_propensity_scores(x, t)

    # Convert to numpy arrays for indexing
    propensity_scores = propensity_scores.reshape(-1, 1)
    y = y.values
    t = t.values

    # Separate treated and control indices
    treated_indices = np.nonzero(t == 1)[0]
    control_indices = np.nonzero(t == 0)[0]

    # Build separate NN models for control and treated units
    nn_control = NearestNeighbors(n_neighbors=n_matches + 1)
    nn_control.fit(propensity_scores[control_indices])

    nn_treated = NearestNeighbors(n_neighbors=n_matches + 1)
    nn_treated.fit(propensity_scores[treated_indices])

    # Initialize lists to store effects
    att_effects = []
    atc_effects = []

    epsilon = 1e-5  # Small value to prevent division by zero

    # For treated units, find matched control units (ATT)
    for i in treated_indices:
        propensity_i = propensity_scores[i].reshape(1, -1)
        distances, indices = nn_control.kneighbors(propensity_i)
        matched_indices = control_indices[indices[0]]

        # Exclude self-match if present
        mask = matched_indices != i
        matched_indices = matched_indices[mask]
        distances = distances[0][mask]

        if len(matched_indices) >= n_matches:
            matched_indices = matched_indices[:n_matches]
            distances = distances[:n_matches]

            # Calculate weights inversely proportional to distance
            weights = 1 / (distances + epsilon)
            weights /= weights.sum()  # Normalize weights to sum to 1

            # Compute weighted average of neighbor outcomes
            weighted_mean = np.dot(weights, y[matched_indices])

            effect = y[i] - weighted_mean
            att_effects.append(effect)

    # For control units, find matched treated units (ATC)
    for i in control_indices:
        propensity_i = propensity_scores[i].reshape(1, -1)
        distances, indices = nn_treated.kneighbors(propensity_i)
        matched_indices = treated_indices[indices[0]]

        # Exclude self-match if present
        mask = matched_indices != i
        matched_indices = matched_indices[mask]
        distances = distances[0][mask]

        if len(matched_indices) >= n_matches:
            matched_indices = matched_indices[:n_matches]
            distances = distances[:n_matches]

            # Calculate weights inversely proportional to distance
            weights = 1 / (distances + epsilon)
            weights /= weights.sum()  # Normalize weights to sum to 1

            # Compute weighted average of neighbor outcomes
            weighted_mean = np.dot(weights, y[matched_indices])

            effect = weighted_mean - y[i]
            atc_effects.append(effect)

    # Combine effects for ATE calculation
    individual_effects = att_effects + atc_effects

    # Calculate ATE, ATT, and ATC
    ate = np.mean(individual_effects) if individual_effects else np.nan
    att = np.mean(att_effects) if att_effects else np.nan
    atc = np.mean(atc_effects) if atc_effects else np.nan

    return ate, att, atc

# Main execution
if __name__ == "__main__":
    DATA_PATH = 'code/data/processed/processed_age_21_outcome_strict.csv'

    # Read and transform the data
    x_data, t_data, y_data = read_and_transform_data(DATA_PATH)

    # Calculate point estimates with n_matches=3
    ate_data, att_data, atc_data = calculate_measures_matching(x_data, t_data, y_data)
    print(f'ATE (Matching): {ate_data:.4f}')
    print(f'ATT (Matching): {att_data:.4f}')
    print(f'ATC (Matching): {atc_data:.4f}')

    # Calculate confidence intervals
    ate_ci, att_ci, atc_ci, bootstrap_ate, bootstrap_att, bootstrap_atc = bci(x_data, t_data, y_data, calculate_measures_matching)

    print(f'ATE 95% CI: {ate_ci}')
    print(f'ATT 95% CI: {att_ci}')
    print(f'ATC 95% CI: {atc_ci}')

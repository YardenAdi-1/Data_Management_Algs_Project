"""
This module provides functions to estimate causal effects using Inverse Probability Weighting (IPW).

Functions:
    calculate_measures_IPW(X, t, y):
        Calculates the Average Treatment Effect (ATE),
        Average Treatment effect on the Treated (ATT),
        and Average Treatment effect on the Controls (ATC) using IPW.

        Parameters:
            X (array-like): Covariates/features.
            t (array-like): Treatment assignment (binary).
            y (array-like): Outcome variable.

        Returns:
            tuple: A tuple containing ATE, ATT, and ATC.

    read_and_transform_data(filepath):
        Reads and preprocesses the data from the given file path.

        Parameters:
            filepath (str): Path to the data file.

        Returns:
            tuple: A tuple containing preprocessed covariates (X),
            treatment assignment (t), and outcome variable (y).
"""

from utils import read_and_transform_data, calculate_propensity_scores, bci



def calculate_measures_ipw(x, t, y):
    """
    Calculate the Average Treatment Effect (ATE), Average Treatment effect on the Treated (ATT),
    and Average Treatment effect on the Controls (ATC) using Inverse Probability Weighting (IPW).
    Parameters:
    x (array-like): Covariates/features used to calculate propensity scores.
    t (array-like): Treatment assignment indicator (1 if treated, 0 if control).
    y (array-like): Outcome variable.
    Returns:
    tuple: A tuple containing ATE, ATT, and ATC.
    """

    n = len(y)
    e = calculate_propensity_scores(x, t)
    ate = sum(y * t / e) / n - sum(y * (1 - t) / (1 - e)) / n
    att = sum(y * t) / sum(t) - sum(y * (1 - t) * e / (1 - e)) / sum((1 - t) * e / (1 - e))
    atc = sum(y * t * (1 - e) / e) / sum(t * (1 - e) / e) - sum(y * (1 - t)) / sum(1 - t)

    return ate, att, atc

DATA_PATH = 'code/data/processed/processed_age_21_outcome_strict.csv'

if __name__ == '__main__':
    x_data, t_data, y_data = read_and_transform_data(DATA_PATH)

    # Calculate point estimates
    ate_data, att_data, atc_data = calculate_measures_ipw(x_data, t_data, y_data)
    print(f'ATE: {ate_data:.4f}, ATT: {att_data:.4f}, ATC: {atc_data:.4f}')

    # Calculate confidence intervals
    ate_ci, att_ci, atc_ci, bootstrap_ate, bootstrap_att, bootstrap_atc = bci(x_data, t_data, y_data, calculate_measures_ipw)
    print(f'ATE 95% CI: {ate_ci}, ATT 95% CI: {att_ci}, ATC 95% CI: {atc_ci}')

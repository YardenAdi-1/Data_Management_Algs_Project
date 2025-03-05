"""
This module implements the T-learner approach for causal inference using Gradient Boosting Classifier.

Functions:
    calculate_measures_t_learner(x, t, y):
        Calculates the Average Treatment Effect (ATE),
        Average Treatment effect on the Treated (ATT),
        and Average Treatment effect on the Control (ATC) using the T-learner approach.

        Parameters:
            x (numpy.ndarray): Feature matrix.
            t (numpy.ndarray): Treatment indicator vector.
            y (numpy.ndarray): Outcome vector.

        Returns:
            tuple: A tuple containing ATE, ATT, and ATC.

    read_and_transform_data(filepath):
        Reads and transforms the data from the given file path.

        Parameters:
            filepath (str): Path to the data file.

        Returns:
            tuple: A tuple containing the feature matrix, treatment indicator vector,
            and outcome vector.

    bci(x, t, y, estimator_func):
        Calculates bootstrap confidence intervals for the ATE, ATT, and ATC.

        Parameters:
            x (numpy.ndarray): Feature matrix.
            t (numpy.ndarray): Treatment indicator vector.
            y (numpy.ndarray): Outcome vector.
            estimator_func (function): Function to estimate ATE, ATT, and ATC.

        Returns:
            tuple: A tuple containing the confidence intervals for ATE, ATT, and ATC,
                   and the bootstrap samples for ATE, ATT, and ATC.
"""

from sklearn.ensemble import GradientBoostingClassifier
import numpy as np

from utils import read_and_transform_data, bci

def calculate_measures_t_learner(x, t, y):
    """
    Calculate the Average Treatment Effect (ATE), Average Treatment effect on the Treated (ATT),
    and Average Treatment effect on the Controls (ATC) using the T-learner approach.
    Parameters:
    x (numpy.ndarray or pandas.DataFrame): Feature matrix.
    t (numpy.ndarray or pandas.Series): Treatment indicator (1 for treated, 0 for control).
    y (numpy.ndarray or pandas.Series): Outcome variable.
    Returns:
    tuple: A tuple containing:
        - ate (float): Average Treatment Effect.
        - att (float): Average Treatment effect on the Treated.
        - atc (float): Average Treatment effect on the Controls.
    """
    # Split data into treatment and control groups
    x_1 = x[t == 1]
    x_0 = x[t == 0]
    y_1 = y[t == 1]
    y_0 = y[t == 0]

    # Fit models for each group
    model_1 = GradientBoostingClassifier(random_state=42, learning_rate=0.1)
    model_0 = GradientBoostingClassifier(random_state=42, learning_rate=0.1)

    model_1.fit(x_1, y_1)
    model_0.fit(x_0, y_0)

    # Predict outcomes for all individuals under both conditions
    y_pred_all_1 = model_1.predict_proba(x)[:, 1]
    y_pred_all_0 = model_0.predict_proba(x)[:, 1]

    # Calculate individual treatment effects
    individual_effects = y_pred_all_1 - y_pred_all_0

    # Calculate ATE
    ate = np.mean(individual_effects)

    # Calculate ATT
    att = np.mean(individual_effects[t == 1])

    # Calculate ATC
    atc = np.mean(individual_effects[t == 0])

    return ate, att, atc


DATA_PATH = 'code/data/processed/processed_age_21_outcome_strict.csv'

if __name__ == '__main__':
    data_x, data_t, data_y = read_and_transform_data(DATA_PATH)

    # Calculate point estimates
    ATE, ATT, ATC = calculate_measures_t_learner(data_x, data_t, data_y)
    print(f'ATE: {ATE:.4f}, ATT: {ATT:.4f}, ATC: {ATC:.4f}')

    # Calculate confidence intervals
    ATE_CI, ATT_CI, ATC_CI, bootstrap_ATE, bootstrap_ATT, bootstrap_ATC = bci(data_x, data_t, data_y, calculate_measures_t_learner)
    print(f'ATE 95% CI: {ATE_CI}, ATT 95% CI: {ATT_CI}, ATC 95% CI: {ATC_CI}')

"""
This module provides functions to perform doubly robust estimation for causal inference.

Functions:
    fit_outcome_models(x, t, y):
        Fits outcome models for treated and control groups using Gradient Boosting Classifier.

        Parameters:
            x (pd.DataFrame or np.ndarray): Feature matrix.
            t (pd.Series or np.ndarray): Treatment indicator (1 for treated, 0 for control).
            y (pd.Series or np.ndarray): Outcome variable.

        Returns:
            model_1 (GradientBoostingClassifier): Fitted model for the treated group.
            model_0 (GradientBoostingClassifier): Fitted model for the control group.

    calculate_measures_doubly_robust(x, t, y):
        Calculates the Average Treatment Effect (ATE), Average Treatment effect on the Treated (ATT),
        and Average Treatment effect on the Controls (ATC) using doubly robust estimation.

        Parameters:
            x (pd.DataFrame or np.ndarray): Feature matrix.
            t (pd.Series or np.ndarray): Treatment indicator (1 for treated, 0 for control).
            y (pd.Series or np.ndarray): Outcome variable.

        Returns:
            ate (float): Average Treatment Effect.
            att (float): Average Treatment effect on the Treated.
            atc (float): Average Treatment effect on the Controls.
"""

from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
from utils import read_and_transform_data, calculate_propensity_scores, bci

def fit_outcome_models(x, t, y):
    """
    Fits outcome models for treated and control groups using Gradient Boosting Classifier.
    Parameters:
    x (numpy.ndarray or pandas.DataFrame): Feature matrix.
    t (numpy.ndarray or pandas.Series): Treatment indicator (1 for treated, 0 for control).
    y (numpy.ndarray or pandas.Series): Outcome variable.
    Returns:
    tuple: A tuple containing two fitted Gradient Boosting Classifier models:
        - model_1: Fitted model for the treated group.
        - model_0: Fitted model for the control group.
    """

    x_1 = x[t == 1]
    x_0 = x[t == 0]
    y_1 = y[t == 1]
    y_0 = y[t == 0]

    model_1 = GradientBoostingClassifier(random_state=42)
    model_1.fit(x_1, y_1)

    model_0 = GradientBoostingClassifier(random_state=42)
    model_0.fit(x_0, y_0)

    return model_1, model_0


def calculate_measures_doubly_robust(x, t, y):
    """
    Calculate the Average Treatment Effect (ATE), Average Treatment effect on the Treated (ATT),
    and Average Treatment effect on the Controls (ATC) using a doubly robust estimator.
    Parameters:
    x (numpy.ndarray): Covariates/features matrix.
    t (numpy.ndarray): Treatment assignment vector (binary).
    y (numpy.ndarray): Outcome vector.
    Returns:
    tuple: A tuple containing the ATE, ATT, and ATC.
    """

    n = len(y)

    e = calculate_propensity_scores(x, t)
    model_1, model_0 = fit_outcome_models(x, t, y)

    y_pred_all_1 = model_1.predict(x)
    y_pred_all_0 = model_0.predict(x)

    g_1_score = y_pred_all_1 + (t / e) * (y - y_pred_all_1)
    g_0_score = y_pred_all_0 + ((1 - t) / (1 - e)) * (y - y_pred_all_0)

    ate = np.sum(g_1_score) / n - np.sum(g_0_score) / n
    att = np.sum(t * y - ((t - e) * y_pred_all_0 / (1 - e))) / np.sum(t)
    atc = np.sum((1 - e) * t * y / e - ((t - e) * y_pred_all_1 / e) - ((1 - t) * y)) / np.sum(1 - t)

    return ate, att, atc

DATA_PATH = 'code/data/processed/processed_age_21_outcome_strict.csv'

if __name__ == '__main__':
    x_data, t_data, y_data = read_and_transform_data(DATA_PATH)

    # Calculate point estimates
    ate_data, att_data, atc_data = calculate_measures_doubly_robust(x_data, t_data, y_data)
    print(f'ATE: {ate_data:.4f}, ATT: {att_data:.4f}, ATC: {atc_data:.4f}')

    # Calculate confidence intervals
    ate_ci, att_ci, atc_ci, bootstrap_ate, bootstrap_att, bootstrap_atc = bci(x_data, t_data, y_data, calculate_measures_doubly_robust)
    print(f'ATE 95% CI: {ate_ci}, ATT 95% CI: {att_ci}, ATC 95% CI: {atc_ci}')

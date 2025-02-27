"""
This module implements an S-Learner for causal inference using a Gradient Boosting Classifier.

Functions:
    calculate_measures_s_learner(x, t, y):
        Calculates the Average Treatment Effect (ATE),
        Average Treatment effect on the Treated (ATT),
        and Average Treatment effect on the Control (ATC) using an S-Learner approach.

    Main execution:
        Reads processed data from a CSV file, transforms it, and calculates ATE, ATT, and ATC.
        Also calculates bootstrap confidence intervals for these measures.

Args:
    x (pd.DataFrame): Features dataframe.
    t (pd.Series): Treatment indicator series.
    y (pd.Series): Outcome series.

Returns:
    tuple: A tuple containing ATE, ATT, and ATC.

Usage:
    Run the script to calculate and print ATE, ATT, ATC, and their 95% confidence intervals.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from utils import read_and_transform_data, bci

def calculate_measures_s_learner(x, t, y):
    """
    Calculate Average Treatment Effect (ATE), Average Treatment effect on the Treated (ATT),
    and Average Treatment effect on the Control (ATC) using the S-learner approach.
    Parameters:
    x (pd.DataFrame): Features dataframe.
    t (pd.Series): Treatment indicator series.
    y (pd.Series): Outcome series.
    Returns:
    tuple: A tuple containing ATE, ATT, and ATC.
    """

    model = GradientBoostingClassifier(random_state=42, learning_rate=0.1)
    model.fit(pd.concat([x, t], axis=1), y)

    # Predict outcomes for all individuals under treatment condition
    xt = pd.concat([x, t], axis=1)
    xt_treated = xt.copy()
    xt_treated['Adult'] = 1
    y_pred_treated = model.predict_proba(xt_treated)[:, 1]

    # Predict outcomes for all individuals under control condition
    xt_control = xt.copy()
    xt_control['Adult'] = 0
    y_pred_control = model.predict_proba(xt_control)[:, 1]

    # Calculate individual treatment effects
    individual_effects = y_pred_treated - y_pred_control

    # Calculate ATE
    ate = np.mean(individual_effects)

    # Calculate ATT
    att = np.mean(individual_effects[xt['Adult'] == 1])

    # Calculate ATC
    atc = np.mean(individual_effects[xt['Adult'] == 0])

    return ate, att, atc


DATA_PATH = 'code/data/processed/processed_age_21_outcome_strict.csv'

if __name__ == '__main__':
    data = pd.read_csv(DATA_PATH)
    x_data, t_data, y_data = read_and_transform_data(DATA_PATH)
    # calculate ATE, ATT, ATC
    ate_data, att_data, atc_data = calculate_measures_s_learner(x_data, t_data, y_data)
    print(f'ATE: {ate_data}, ATT: {att_data}, ATC: {atc_data}')

    # calculate confidence intervals
    ate_ci, att_ci, atc_ci, bootstrap_ate, bootstrap_att, bootstrap_atc = bci(x_data, t_data, y_data, calculate_measures_s_learner)
    print(f'ATE 95% CI: {ate_ci}, ATT 95% CI: {att_ci}, ATC 95% CI: {atc_ci}')

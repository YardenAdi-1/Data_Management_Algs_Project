"""
Utility functions for estimating treatment effects.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


def read_and_transform_data(data_path):
    """
    Reads a CSV file,
    processes the data by scaling numerical columns and one-hot encoding categorical columns,
    and returns the transformed features, treatment indicator, and target variable.
    Args:
        data_path (str): The file path to the CSV data file.
    Returns:
        tuple: A tuple containing:
            - X (pd.DataFrame): The transformed feature matrix.
            - t (pd.Series): The treatment indicator (column 'Adult').
            - y (pd.Series): The target variable (column 'Target').
    Notes:
        - Numerical columns are scaled using StandardScaler.
        - Categorical columns are one-hot encoded,
        with the first category dropped to avoid multicollinearity.
        - The column 'Application mode_39' is removed from the transformed feature matrix.
    """

    data = pd.read_csv(data_path)

    t = data['Adult']
    y = data['Target']
    x = data.drop(columns=['Adult', 'Target'])

    numerical_columns = ['Previous qualification (grade)', 'Admission grade',
                         'Unemployment rate', 'Inflation rate', 'GDP']
    categorical_columns = list(set(x.columns) - set(numerical_columns))

    # Scaling numerical columns
    scaler = StandardScaler()
    x[numerical_columns] = scaler.fit_transform(x[numerical_columns])

    # One-hot encoding for categorical columns
    x = pd.get_dummies(x, columns=categorical_columns, drop_first=True)

    # remove from data the column Application mode_39
    x = x.drop(columns=['Application mode_39'])

    return x, t, y

def calculate_propensity_scores(x, t):
    """
    Calculate propensity scores using a Gradient Boosting Classifier.
    Parameters:
    X (pd.DataFrame or np.ndarray): The feature matrix.
    t (pd.Series or np.ndarray): The treatment assignment vector.
    Returns:
    np.ndarray: The propensity scores, clipped to avoid extreme values.
    """

    propensity_model = RandomForestClassifier(random_state=42)
    propensity_model.fit(x, t)
    e = propensity_model.predict_proba(x)[:, 1]

    # Clip propensity scores to avoid extreme values
    epsilon = 1e-5
    e = np.clip(e, epsilon, 1 - epsilon)

    return e

def bci(x, t, y, calculate_measures, num_bootstrap=1000, ci_level=95, **kwargs):
    """
    Perform bootstrap resampling to calculate confidence intervals for ATE, ATT, and ATC.

    Parameters:
    X (pd.DataFrame): Feature matrix
    t (pd.Series): Treatment assignments
    y (pd.Series): Outcome variable
    calculate_measures (function): Function to calculate ATE, ATT, and ATC for a given sample
    num_bootstrap (int): Number of bootstrap iterations
    ci_level (float): Confidence interval level (e.g., 95 for 95% CI)
    **kwargs: Additional keyword arguments to pass to calculate_measures function

    Returns:
    tuple:
        - ate_ci (np.array): Confidence interval for ATE
        - att_ci (np.array): Confidence interval for ATT
        - atc_ci (np.array): Confidence interval for ATC
        - bootstrap_ate (list): List of ATE values from bootstrap samples
        - bootstrap_att (list): List of ATT values from bootstrap samples
        - bootstrap_atc (list): List of ATC values from bootstrap samples
    """
    bootstrap_ate = []
    bootstrap_att = []
    bootstrap_atc = []

    # Combine X, t, and y into a single dataframe for easier resampling
    data = pd.concat([x, t, y], axis=1)

    # Perform bootstrap sampling
    for _ in range(num_bootstrap):
        # Create a bootstrap sample (resample with replacement)
        bootstrap_sample = data.sample(n=len(data), replace=True)

        # Split the bootstrap sample back into X, t, and y
        x_bootstrap = bootstrap_sample.iloc[:, :-2]
        t_bootstrap = bootstrap_sample.iloc[:, -2]
        y_bootstrap = bootstrap_sample.iloc[:, -1]

        # Calculate measures for this bootstrap sample
        ate, att, atc = calculate_measures(x_bootstrap, t_bootstrap, y_bootstrap, **kwargs)

        # Store the results
        bootstrap_ate.append(ate)
        bootstrap_att.append(att)
        bootstrap_atc.append(atc)

    # Calculate percentiles for confidence intervals
    lower_percentile = (100 - ci_level) / 2
    upper_percentile = 100 - lower_percentile

    ate_ci = np.percentile(bootstrap_ate, [lower_percentile, upper_percentile])
    att_ci = np.percentile(bootstrap_att, [lower_percentile, upper_percentile])
    atc_ci = np.percentile(bootstrap_atc, [lower_percentile, upper_percentile])

    return ate_ci, att_ci, atc_ci, bootstrap_ate, bootstrap_att, bootstrap_atc

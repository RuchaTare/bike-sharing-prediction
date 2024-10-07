"""
This module contains functions to evaluate the model performance.
"""

import logging
import numpy as np
import sklearn
from sklearn.metrics import mean_squared_error, r2_score


def evaluate_model(model, X_train, X_test, y_train, y_test):
    """
    Evaluate the model
    """

    logging.info("Evaluating the model")

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    logging.info(f"Train RMSE: {train_rmse:.4f}")
    logging.info(f"Test RMSE: {test_rmse:.4f}")
    logging.info(f"Train R^2: {train_r2:.4f}")
    logging.info(f"Test R^2: {test_r2:.4f}")

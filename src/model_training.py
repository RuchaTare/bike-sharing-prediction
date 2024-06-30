"""
Model training module
"""

import logging

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import RFECV, VarianceThreshold
from sklearn.linear_model import ElasticNetCV, HuberRegressor, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score, train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from utils import read_csv, read_yaml


def _read_cleaned_data(file_path: str) -> pd.DataFrame:
    """
    Read the cleaned data
    """

    logging.info("Reading the cleaned data")

    try:
        data = read_csv(file_path)
        print(f"data head {data.head()}")
    except Exception as e:
        logging.error(f"Error reading the cleaned data: {str(e)}")
    return data


def _train_test_split(
    data: pd.DataFrame, test_size: float, random_state: int, target_column: str
) -> pd.DataFrame:
    """
    Split the data into training and test set
    """

    logging.info("Splitting the data into training and test set")

    try:
        X = data.drop(columns=[target_column])
        y = data[target_column]
        print(X.shape, y.shape)
        print(f"print X col and y col {X.columns, y}")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error(f"Error splitting the data into training and test set: {str(e)}")
        return None, None, None, None


def _model_selection(X: pd.DataFrame, y: pd.DataFrame, models: dict) -> sklearn.base.BaseEstimator:
    """
    Train models with CV and select the best model
    """

    logging.info("Selecting the best model")

    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    best_model = None
    min_mean_rmse = float("inf")
    cv_results = {}

    for name, model in models.items():
        logging.info(f"Training {name}")
        scores = cross_val_score(model, X, y, cv=kf, scoring="neg_mean_squared_error")
        mean_rmse = np.mean(np.sqrt(-scores))
        std_rmse = np.std(np.sqrt(-scores))
        cv_results[name] = (mean_rmse, std_rmse)
        logging.info(f"{name}: Mean RMSE = {mean_rmse:.4f}, Std RMSE = {std_rmse:.4f}")

        if mean_rmse < min_mean_rmse:
            min_mean_rmse = mean_rmse
            best_model = model

    logging.info(
        f"Best model: {best_model.__class__.__name__} with Mean RMSE = {min_mean_rmse:.4f}"
    )

    return best_model


# def _rfecv(model: sklearn.base.BaseEstimator, X: pd.DataFrame, y: pd.DataFrame) -> list:
#     """
#     Perform Recursive Feature Elimination
#     """
#     logging.info("Performing Recursive Feature Elimination with Cross-Validation (RFECV)")
#     try:
#         rfecv = RFECV(estimator=model, step=1, cv=5, scoring="neg_mean_squared_error")
#         rfecv.fit(X, y)
#         logging.info("RFECV completed successfully")
#         selected_features = X.columns[rfecv.support_]
#         return selected_features
#     except Exception as e:
#         logging.error(f"Error performing RFECV: {str(e)}")
#         return None


# def hyperparameter_tuning(model, X_train, y_train):
#     logging.info("Performing hyperparameter tuning")
#     param_grid = {
#         "n_estimators": [100, 200, 300],
#         "learning_rate": [0.01, 0.05, 0.1],
#         "max_depth": [3, 5, 7],
#     }
#     grid_search = GridSearchCV(
#         estimator=model, param_grid=param_grid, cv=5, scoring="neg_mean_squared_error"
#     )
#     grid_search.fit(X_train, y_train)
#     logging.info(f"Best parameters found: {grid_search.best_params_}")
#     return grid_search.best_estimator_


def trainer(config_data):
    """
    Train the model
    """
    logging.info("Training the model")

    data = _read_cleaned_data(config_data["cleaned_data_path"])
    logging.info(f"The shape of cleaned data is {data.shape}")

    X_train, X_test, y_train, y_test = _train_test_split(
        data, config_data["test_size"], config_data["random_state"], config_data["target_column"]
    )

    if X_train is None or X_test is None or y_train is None or y_test is None:
        logging.error("Failed to split the data")

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(),
        "Huber Regressor": HuberRegressor(max_iter=2000),
        "Elastic Net CV": ElasticNetCV(),
        "Decision Tree Regressor": DecisionTreeRegressor(),
        "Random Forest Regressor": RandomForestRegressor(),
        "Extra Trees Regressor": ExtraTreesRegressor(),
        "Gradient Boosting Regressor": GradientBoostingRegressor(),
    }

    best_model = _model_selection(X_train, y_train, models)

    # selected_features = _rfecv(best_model, X_train, y_train)
    # logging.info(f"Selected features: {selected_features}")
    # logging.info(f"Number of selected features: {len(selected_features)}")

    # X_train_filtered = X_train[selected_features]
    # X_test_filtered = X_test[selected_features]
    # X_train_filtered_scaled = scaler.fit_transform(X_train_filtered)
    # X_test_filtered_scaled = scaler.transform(X_test_filtered)

    # best_model = hyperparameter_tuning(best_model, X_train_filtered_scaled, y_train)

    # best_model.fit(X_train_filtered_scaled, y_train)
    # y_pred = best_model.predict(X_test_filtered_scaled)
    # mse = np.sqrt(mean_squared_error(y_test, y_pred))
    # mae = mean_absolute_error(y_test, y_pred)
    # r2 = r2_score(y_test, y_pred)
    # logging.info(f"Mean Squared Error: {mse}, Mean Absolute Error: {mae}, R2 Score: {r2}")

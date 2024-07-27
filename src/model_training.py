"""
Model training module
"""

import logging
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import RFE, RFECV
from sklearn.linear_model import ElasticNetCV, HuberRegressor, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeRegressor

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


def rfecv(best_model, X_train, y_train):
    """
    Recursive feature elimination with cross-validation
    """

    logging.info("Performing Recursive Feature Elimination with Cross-Validation")
    rfecv = RFECV(estimator=best_model, step=1, cv=KFold(5), scoring="neg_mean_squared_error")
    rfecv.fit(X_train, y_train)
    selected_features = rfecv.support_
    print("Optimal number of features:", rfecv.n_features_)
    print("Selected features:", selected_features)
    X_train_selected = rfecv.transform(X_train)

    return X_train_selected, selected_features


def train_final_model(best_model, X_train, y_train):
    """
    Train the final model
    """

    logging.info("Training the final model")
    trained_model = best_model.fit(X_train, y_train)

    return trained_model


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
    logging.info(f"Best model: {best_model}")

    X_train_selected, selected_features = rfecv(best_model, X_train, y_train)

    model = train_final_model(best_model, X_train_selected, y_train)
    logging.info(f"Trained model: {model} save to model.pkl")
    joblib.dump(model, "model.pkl")

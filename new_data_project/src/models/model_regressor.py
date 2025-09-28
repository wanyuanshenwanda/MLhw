import os

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import numpy as np

class MultiProjectRegressor:
    def __init__(self, model_type="LinearRegression", log_transform=False, **kwargs):
        self.model_type = model_type
        self.log_transform = log_transform
        self.model_kwargs = kwargs
        self.models = []
        self.log_flags = []  # 是否做 log1p
        self.metrics = []

    def _get_regressor(self):
        log_model = False
        mt = self.model_type
        if mt == "LinearRegression":
            model = LinearRegression(**self.model_kwargs)
        elif mt == "RandomForest":
            model = RandomForestRegressor(**self.model_kwargs)
        elif mt == "GradientBoosting":
            model = GradientBoostingRegressor(**self.model_kwargs)
        elif mt == "HistGradientBoosting":
            model = HistGradientBoostingRegressor(**self.model_kwargs)
        elif mt == "GradientBoosting_log1p":
            model = GradientBoostingRegressor(**self.model_kwargs)
            log_model = True
        else:
            raise ValueError(f"Unknown model_type: {mt}")
        return model, log_model

    def fit(self, X_train_list, y_train_list, X_test_list=None, y_test_list=None):
        self.models = []
        self.log_flags = []
        self.metrics = []

        for i, X_train in enumerate(X_train_list):
            y_train = y_train_list[i]
            X_test = X_test_list[i] if X_test_list else None
            y_test = y_test_list[i] if y_test_list else None

            model, log_flag = self._get_regressor()
            y_fit = np.log1p(y_train) if log_flag else y_train
            model.fit(X_train, y_fit)

            self.models.append(model)
            self.log_flags.append(log_flag)

            metric = {}
            if X_test is not None and y_test is not None:
                y_pred = model.predict(X_test)
                if log_flag:
                    y_pred = np.expm1(y_pred)

                # 四个指标
                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)

                metric = {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}
                print(f"Dataset {i}: MAE={mae:.4f}, MSE={mse:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")

            self.metrics.append(metric)

    # def predict(self, X_list, method="average", weights=None):
    #     preds = []
    #     for i, X in enumerate(X_list):
    #         model = self.models[i]
    #         log_flag = self.log_flags[i]
    #         y_pred = model.predict(X)
    #         if log_flag:
    #             y_pred = np.expm1(y_pred)
    #         preds.append(y_pred)
    #
    #     preds = np.array(preds)  # shape: (n_models, n_samples)
    #     if method == "average":
    #         return np.mean(preds, axis=0)
    #     elif method == "weighted":
    #         if weights is None:
    #             raise ValueError("weights must be provided for weighted method")
    #         weights = np.array(weights).reshape(-1, 1)
    #         return (preds * weights).sum(axis=0) / weights.sum()
    #     else:
    #         raise ValueError(f"Unknown method: {method}")
